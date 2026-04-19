//! Linear-scan register allocator with active/inactive sets and live-range
//! splitting on eviction.
//!
//! Inspired by Wimmer–Franz "Linear Scan Register Allocation for the Java
//! HotSpot Client Compiler" and regalloc2's model. Operates on the
//! multi-segment `LiveRange`s produced by `analysis::liveness`.
//!
//! Core idea: at each step, the "active" set is vregs whose live range
//! covers the current position; they occupy a preg right now. "Inactive"
//! vregs hold a preg but are currently in a hole — their next segment
//! hasn't started yet. A preg held by an inactive vreg is freely usable by
//! another vreg until the two ranges first intersect, which is computed
//! via two-pointer sweep over segments.
//!
//! **Live-range splitting on eviction.** When the allocator evicts a vreg
//! `u` to hand its preg to a higher-priority `v`, `u` does not go to the
//! stack for its whole life. Instead we split `u` at `v`'s first point:
//! `u` keeps its preg for `[u.first_start, split_point)` and moves to a
//! stack slot for `[split_point, u.last_end)`. A `SplitMove` is recorded
//! so the emitter can save the preg into the slot immediately before `v`'s
//! defining instruction. Later uses of `u` inside the Reg piece still load
//! from the preg (fast); uses inside the Stack piece load from the slot.
//! This dramatically cuts memory traffic compared to whole-vreg spilling.
//!
//! The allocator also does:
//!
//! * **Hint-based Copy coalescing.** On a `PseudoInstruction::Copy { dst,
//!   src }`, the dst's hint is src's preg; if available for dst's full
//!   range, assigned. Eliminates the copy in `pseudo_cleanup`.
//! * **Pre-binds enforced by eviction.** When a vreg is pre-bound (e.g. an
//!   ABI arg shim), any active or inactive vreg blocking the target preg
//!   across the pre-bound vreg's range is split/evicted.
//! * **Farthest-endpoint spill heuristic.** When choosing who to evict,
//!   prefer an interval that outlives `v` (more future uses on the stack
//!   is still cheaper than stalling `v`).

use std::collections::HashMap;

use crate::codegen::analysis::cfg::CFG;
use crate::codegen::analysis::layout::{BlockLayout, ProgramPoint};
use crate::codegen::analysis::liveness::{LiveRanges, Segment};
use crate::codegen::regalloc::{
    AllocatedSlot, Assignment, RegAllocConfig, RegAllocResult, RegAllocator, SplitMove, StackSlot,
};
use crate::codegen::tir::{Func, Inst, Instruction, PseudoInstruction, Reg, Type};
use crate::support::slotmap::SecondaryMap;

pub struct LinearScan;

impl<I: Inst> RegAllocator<I> for LinearScan {
    fn allocate(func: &Func<I>, cfg: &CFG, config: &RegAllocConfig) -> RegAllocResult {
        let layout = BlockLayout::compute(func);
        Allocator::new(func, cfg, &layout, config).run()
    }
}

struct Allocator<'a, I: Inst> {
    func: &'a Func<I>,
    config: &'a RegAllocConfig,
    ranges: LiveRanges,
    copy_src: SecondaryMap<Reg, Option<Reg>>,

    /// Merged view of `config.reg_bind` + in-stream `RegDef` pseudos.
    /// Both sources contribute whole-life pins; if a vreg is pinned from
    /// both sides, they must agree. The rest of the allocator consults
    /// this instead of `config.reg_bind` directly so frontends can use
    /// either mechanism interchangeably.
    effective_binds: HashMap<Reg, Reg>,

    /// Current slot of each vreg (None = not yet allocated). Updated on
    /// assign and on eviction. This is transient: on piece close, it's
    /// committed to `assignments[v]`.
    current_slot: Vec<Option<AllocatedSlot>>,
    /// Program point at which the current in-flight piece started.
    current_piece_start: Vec<ProgramPoint>,
    /// Accumulator for the final output.
    assignments: SecondaryMap<Reg, Assignment>,

    active: Vec<Reg>,
    inactive: Vec<Reg>,

    frame_layout: Vec<usize>,
    split_moves: Vec<SplitMove>,
}

impl<'a, I: Inst> Allocator<'a, I> {
    fn new(
        func: &'a Func<I>,
        cfg: &'a CFG,
        layout: &'a BlockLayout,
        config: &'a RegAllocConfig,
    ) -> Self {
        let ranges = LiveRanges::compute(func, cfg, layout);
        let copy_src = collect_copy_src(func);
        let effective_binds = merge_pre_binds(config, func);
        let n = func.get_regs_count();
        let mut assignments = SecondaryMap::new(n);
        assignments.fill(Assignment::default());
        // Aggregate vregs must be erased before regalloc; catch stale
        // survivors in debug builds rather than miscoloring them.
        #[cfg(debug_assertions)]
        for i in 0..n {
            let t = func.vreg_type(i as Reg);
            assert!(
                !matches!(t, Type::Agg(_)) || ranges[i as Reg].is_empty(),
                "aggregate vreg {i} reached regalloc — lower_aggregates must run first"
            );
        }
        Self {
            func,
            config,
            ranges,
            copy_src,
            effective_binds,
            current_slot: vec![None; n],
            current_piece_start: vec![0; n],
            assignments,
            active: Vec::new(),
            inactive: Vec::new(),
            frame_layout: Vec::new(),
            split_moves: Vec::new(),
        }
    }

    fn is_fp(&self, v: Reg) -> bool {
        self.func.vreg_type(v).is_fp_or_vector()
    }

    fn pool_for(&self, v: Reg) -> &[Reg] {
        if self.is_fp(v) {
            &self.config.allocatable_fp_regs
        } else {
            &self.config.allocatable_regs
        }
    }

    fn same_class(&self, a: Reg, b: Reg) -> bool {
        self.is_fp(a) == self.is_fp(b)
    }

    fn run(mut self) -> RegAllocResult {
        self.check_pre_bind_compat();
        let mut order: Vec<Reg> = (0..self.current_slot.len() as u32)
            .filter(|&v| self.ranges[v].first_start().is_some())
            .collect();
        order.sort_by(|&a, &b| {
            let sa = self.ranges[a].first_start().unwrap();
            let sb = self.ranges[b].first_start().unwrap();
            sa.cmp(&sb).then_with(|| {
                let ba = self.effective_binds.contains_key(&a);
                let bb = self.effective_binds.contains_key(&b);
                bb.cmp(&ba)
            })
        });

        for v in order {
            let position = self.ranges[v].first_start().unwrap();
            self.advance(position);
            self.allocate(v, position);
            if let Some(AllocatedSlot::Reg(_)) = self.current_slot[v as usize] {
                self.active.push(v);
            }
        }

        // Finalize: close every open piece at its vreg's last_end.
        for v_idx in 0..self.current_slot.len() {
            if self.current_slot[v_idx].is_none() {
                continue;
            }
            let v = v_idx as Reg;
            let end = self.ranges[v].last_end().unwrap();
            self.close_piece(v, end);
        }

        let frame_size = (self.frame_layout.len() * 8) as u32;
        RegAllocResult {
            assignments: self.assignments,
            frame_layout: self.frame_layout,
            frame_size,
            split_moves: self.split_moves,
        }
    }

    fn advance(&mut self, position: ProgramPoint) {
        let mut i = 0;
        while i < self.active.len() {
            let v = self.active[i];
            let r = &self.ranges[v];
            if r.last_end().unwrap() <= position {
                self.active.swap_remove(i);
            } else if !r.covers(position) {
                self.inactive.push(v);
                self.active.swap_remove(i);
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.inactive.len() {
            let v = self.inactive[i];
            let r = &self.ranges[v];
            if r.last_end().unwrap() <= position {
                self.inactive.swap_remove(i);
            } else if r.covers(position) {
                self.active.push(v);
                self.inactive.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn allocate(&mut self, v: Reg, position: ProgramPoint) {
        if let Some(&target) = self.effective_binds.get(&v) {
            self.evict_conflicts_on(target, v, position);
            self.assign_fresh_reg(v, target);
            return;
        }

        let v_end = self.ranges[v].last_end().unwrap();
        let blocked_at = self.compute_blocked_at(v, position);

        if let Some(hint) = self.copy_hint(v)
            && self.pool_for(v).contains(&hint)
            && blocked_at.get(&hint).copied().unwrap_or(0) >= v_end
        {
            self.assign_fresh_reg(v, hint);
            return;
        }

        let best = self
            .pool_for(v)
            .iter()
            .map(|&p| (p, blocked_at.get(&p).copied().unwrap_or(0)))
            .max_by_key(|&(_, fu)| fu);

        if let Some((p, fu)) = best
            && fu >= v_end
        {
            self.assign_fresh_reg(v, p);
            return;
        }

        if let Some((u, p)) = self.pick_eviction_candidate(v, position, v_end) {
            self.evict_to_stack(u, position);
            self.active.retain(|&x| x != u);
            self.inactive.retain(|&x| x != u);
            self.assign_fresh_reg(v, p);
            return;
        }

        self.assign_fresh_stack(v);
    }

    fn compute_blocked_at(&self, v: Reg, position: ProgramPoint) -> HashMap<Reg, ProgramPoint> {
        let mut blocked_at: HashMap<Reg, ProgramPoint> = HashMap::new();
        for &p in self.pool_for(v) {
            blocked_at.insert(p, ProgramPoint::MAX);
        }
        for &u in &self.active {
            if !self.same_class(u, v) {
                continue;
            }
            let p = self.current_preg(u);
            if blocked_at.contains_key(&p) {
                blocked_at.insert(p, position);
            }
        }
        let v_range = &self.ranges[v];
        for &u in &self.inactive {
            if !self.same_class(u, v) {
                continue;
            }
            let p = self.current_preg(u);
            if !blocked_at.contains_key(&p) {
                continue;
            }
            let u_range = &self.ranges[u];
            if let Some(pt) = u_range.next_intersection_at_or_after(v_range, position) {
                let prev = blocked_at[&p];
                blocked_at.insert(p, prev.min(pt));
            }
        }
        blocked_at
    }

    fn pick_eviction_candidate(
        &self,
        v: Reg,
        position: ProgramPoint,
        v_end: ProgramPoint,
    ) -> Option<(Reg, Reg)> {
        let v_range = &self.ranges[v];
        let mut best: Option<(Reg, Reg, ProgramPoint)> = None;

        for &p in self.pool_for(v) {
            // Sole-blocker detection: count and keep the last witness.
            // Any preg with 0 or >=2 blockers is skipped.
            let mut count: u32 = 0;
            let mut only: Reg = 0;
            for &w in &self.active {
                if !self.same_class(w, v) {
                    continue;
                }
                if self.current_preg(w) == p {
                    count += 1;
                    only = w;
                    if count > 1 {
                        break;
                    }
                }
            }
            if count <= 1 {
                for &w in &self.inactive {
                    if !self.same_class(w, v) {
                        continue;
                    }
                    if self.current_preg(w) == p
                        && self.ranges[w]
                            .next_intersection_at_or_after(v_range, position)
                            .is_some()
                    {
                        count += 1;
                        only = w;
                        if count > 1 {
                            break;
                        }
                    }
                }
            }
            if count != 1 {
                continue;
            }
            let u = only;
            if self.effective_binds.contains_key(&u) {
                continue;
            }
            let u_end = self.ranges[u].last_end().unwrap();
            if u_end <= v_end {
                continue;
            }
            if best.is_none() || u_end > best.unwrap().2 {
                best = Some((u, p, u_end));
            }
        }

        best.map(|(u, p, _)| (u, p))
    }

    fn evict_conflicts_on(&mut self, target: Reg, v: Reg, position: ProgramPoint) {
        let v_range = &self.ranges[v];
        let mut seen = std::collections::HashSet::new();
        let mut conflicts: Vec<Reg> = Vec::new();
        for &u in &self.active {
            if !self.same_class(u, v) {
                continue;
            }
            if self.current_preg(u) == target && seen.insert(u) {
                conflicts.push(u);
            }
        }
        for &u in &self.inactive {
            if !self.same_class(u, v) {
                continue;
            }
            if self.current_preg(u) == target
                && self.ranges[u]
                    .next_intersection_at_or_after(v_range, position)
                    .is_some()
                && seen.insert(u)
            {
                conflicts.push(u);
            }
        }
        for u in conflicts {
            assert!(
                !self.effective_binds.contains_key(&u),
                "pre-bind conflict: vreg {u} also pre-bound to preg {target}, can't evict for vreg {v}"
            );
            self.evict_to_stack(u, position);
            self.active.retain(|&x| x != u);
            self.inactive.retain(|&x| x != u);
        }
    }

    /// Walk the `Copy` chain from `v` and return the preg of the nearest
    /// ancestor already in a Reg slot; `None` if the chain bottoms out on
    /// a stack/unallocated link or cycles.
    fn copy_hint(&self, v: Reg) -> Option<Reg> {
        let mut cur = self.copy_src.get(v).copied().flatten()?;
        let mut visited = [u32::MAX; 16];
        visited[0] = v;
        let mut n = 1;
        let pool = self.pool_for(v);
        while n < visited.len() {
            if visited[..n].contains(&cur) {
                return None;
            }
            visited[n] = cur;
            n += 1;
            if let Some(AllocatedSlot::Reg(p)) = self
                .current_slot
                .get(cur as usize)
                .and_then(|s| s.as_ref())
                && pool.contains(p)
            {
                return Some(*p);
            }
            cur = self.copy_src.get(cur).copied().flatten()?;
        }
        None
    }

    // ---- Piece-lifecycle helpers ----

    /// First-time Reg assignment: open a new piece starting at v's first
    /// point.
    fn assign_fresh_reg(&mut self, v: Reg, p: Reg) {
        let start = self.ranges[v].first_start().unwrap();
        self.current_slot[v as usize] = Some(AllocatedSlot::Reg(p));
        self.current_piece_start[v as usize] = start;
    }

    /// First-time Stack assignment: the vreg never gets a preg.
    fn assign_fresh_stack(&mut self, v: Reg) {
        let start = self.ranges[v].first_start().unwrap();
        let s = self.fresh_slot();
        self.current_slot[v as usize] = Some(AllocatedSlot::Stack(s));
        self.current_piece_start[v as usize] = start;
    }

    /// Evict a vreg currently in Reg(p) to a stack slot at `split_pt`.
    /// Closes its Reg piece, opens a Stack piece, records the store
    /// moves.
    ///
    /// **Correctness across branches.** If u has multiple live-range
    /// segments (e.g. post-SSA-destruction phi destinations with
    /// Copies in several predecessors, or a value live in and live
    /// out of a diamond), placing a SplitMove only at `split_pt`
    /// leaves the stack slot uninitialized on paths that bypass
    /// `split_pt`. We emit an additional SplitMove at the start of
    /// every segment that starts on or before `split_pt`, so whichever
    /// path is taken, the preg's value is mirrored to the stack slot
    /// before any later read from the slot.
    fn evict_to_stack(&mut self, u: Reg, split_pt: ProgramPoint) {
        let AllocatedSlot::Reg(p) = self
            .current_slot[u as usize]
            .expect("evict_to_stack called on unassigned vreg")
        else {
            panic!("evict_to_stack called on vreg already on stack");
        };
        self.close_piece(u, split_pt);
        let s = self.fresh_slot();
        self.current_slot[u as usize] = Some(AllocatedSlot::Stack(s));
        self.current_piece_start[u as usize] = split_pt;
        // Primary SplitMove at split_pt.
        self.split_moves.push(SplitMove {
            at_point: split_pt,
            from_preg: p,
            to_slot: s,
        });
        // When `u` has more than one segment, the primary SplitMove
        // at `split_pt` only fires on paths that actually reach that
        // point. For segments that don't reach `split_pt` (e.g. a
        // branch-side def whose uses later read the stack slot), we
        // emit an additional SplitMove that captures the preg's value
        // immediately after the segment becomes live. Program points
        // alternate `use_pt` (even) / `def_pt` (odd), and the
        // emitter only fires SplitMoves that match an instruction's
        // `def_pt`. So we map each segment start to the next `def_pt`
        // where the preg has settled with `u`'s value:
        //
        // * segment starts at a block-entry `use_pt` (live-in): the
        //   preg already carries `u` when the block opens — save
        //   right before the first instruction runs, i.e. at that
        //   instruction's `def_pt = start + 1`.
        // * segment starts at a `def_pt` (u is produced by this
        //   instruction): the preg holds `u` *after* the instruction
        //   finishes — save just before the next instruction, i.e.
        //   at `def_pt + 2` (which is the next instruction's `def_pt`).
        let segs = self.ranges[u].segments().to_vec();
        if segs.len() > 1 {
            for seg in &segs {
                if seg.start >= split_pt {
                    break;
                }
                let save_pt = if seg.start % 2 == 0 {
                    // Live-in segment: save at this block's first
                    // instruction's def_pt.
                    seg.start + 1
                } else {
                    // Def segment: save after the defining instruction
                    // completes (i.e., at the next instruction's
                    // def_pt).
                    seg.start + 2
                };
                if save_pt >= split_pt {
                    continue;
                }
                // Don't emit past the segment's own end — if the
                // segment is just one point wide (dead def), there's
                // no later instruction to save before.
                if save_pt >= seg.end {
                    continue;
                }
                self.split_moves.push(SplitMove {
                    at_point: save_pt,
                    from_preg: p,
                    to_slot: s,
                });
            }
        }
    }

    /// Commit the in-flight piece `[current_piece_start[v], end)` to
    /// `assignments[v]`. Caller must not call this if the piece is empty.
    fn close_piece(&mut self, v: Reg, end: ProgramPoint) {
        let start = self.current_piece_start[v as usize];
        let slot = self.current_slot[v as usize].expect("close_piece: no current slot");
        if start < end {
            self.assignments
                .get_mut(v)
                .unwrap()
                .pieces
                .push((Segment { start, end }, slot));
        }
    }

    fn fresh_slot(&mut self) -> StackSlot {
        let s = self.frame_layout.len() as StackSlot;
        self.frame_layout.push(s as usize * 8);
        s
    }

    fn current_preg(&self, v: Reg) -> Reg {
        match self.current_slot[v as usize].expect("current_preg: no current slot") {
            AllocatedSlot::Reg(p) => p,
            AllocatedSlot::Stack(_) => {
                panic!("current_preg called on vreg on stack")
            }
        }
    }

    /// Panic upfront if two distinct pre-bound vregs with overlapping live
    /// ranges are pinned to the same preg.
    fn check_pre_bind_compat(&self) {
        use std::collections::HashMap;
        let mut by_preg: HashMap<Reg, Vec<Reg>> = HashMap::new();
        for (&v, &p) in &self.effective_binds {
            by_preg.entry(p).or_default().push(v);
        }
        for (p, mut vs) in by_preg {
            vs.retain(|&v| !self.ranges[v].is_empty());
            if vs.len() < 2 {
                continue;
            }
            for i in 0..vs.len() {
                for j in i + 1..vs.len() {
                    let a = &self.ranges[vs[i]];
                    let b = &self.ranges[vs[j]];
                    let start = a.first_start().unwrap().min(b.first_start().unwrap());
                    assert!(
                        a.next_intersection_at_or_after(b, start).is_none(),
                        "pre-bind conflict: vreg {} and vreg {} are both pinned to preg {} \
                         (via `reg_bind` or `RegDef`) with overlapping live ranges — one \
                         would have to be spilled, defeating the pre-bind. Fix the frontend \
                         to pick disjoint pregs for concurrently-live pre-binds.",
                        vs[i],
                        vs[j],
                        p
                    );
                }
            }
        }
    }
}

fn collect_copy_src<I: Inst>(func: &Func<I>) -> SecondaryMap<Reg, Option<Reg>> {
    let mut m = SecondaryMap::new(func.get_regs_count());
    m.fill(None);
    for (_b, bd) in func.blocks_iter() {
        for inst in bd.iter() {
            if let Instruction::Pseudo(PseudoInstruction::Copy { dst, src }) = inst {
                m.set(*dst, Some(*src));
            }
        }
    }
    m
}

/// Build the allocator's effective pre-bind map by merging `config.reg_bind`
/// with in-stream `RegDef` pseudos. Both sources pin a vreg to a preg for
/// its whole life; a vreg that appears in both must agree on the same preg.
fn merge_pre_binds<I: Inst>(config: &RegAllocConfig, func: &Func<I>) -> HashMap<Reg, Reg> {
    let mut out: HashMap<Reg, Reg> = config.reg_bind.clone();
    for (_b, bd) in func.blocks_iter() {
        for inst in bd.iter() {
            if let Instruction::Pseudo(PseudoInstruction::RegDef { vreg, preg }) = inst {
                match out.insert(*vreg, *preg) {
                    Some(prev) if prev != *preg => panic!(
                        "vreg {vreg} is pre-bound to both preg {prev} (via reg_bind) \
                         and preg {preg} (via RegDef). Sources must agree."
                    ),
                    _ => {}
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::isa::x64::inst::X64Inst;
    use crate::codegen::isa::x64::regs::*;
    use crate::codegen::tir::PseudoInstruction;
    use std::collections::HashMap;

    fn cfg4(reg_bind: HashMap<Reg, Reg>) -> RegAllocConfig {
        RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RAX, RBX, RCX, RDX],
            scratch_regs: vec![R12, R13],
            allocatable_fp_regs: Vec::new(),
            scratch_fp_regs: Vec::new(),
            reg_bind,
        }
    }

    fn uniform(res: &RegAllocResult, v: Reg) -> AllocatedSlot {
        res.assignments[v]
            .uniform_slot()
            .expect("vreg was split unexpectedly")
    }

    #[test]
    fn copy_hint_coalesces_dst_onto_src_preg_when_src_dies() {
        let mut func = Func::<X64Inst>::new("t".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let mut reg_bind = HashMap::new();
        reg_bind.insert(v0, RDI);
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: v0, idx: 0 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: v1, src: v0 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v1 });
        }
        let cfg = CFG::compute(&func).unwrap();
        let cfg_cfg = RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RDI, RAX, RBX, RCX],
            scratch_regs: vec![R12, R13],
            allocatable_fp_regs: Vec::new(),
            scratch_fp_regs: Vec::new(),
            reg_bind,
        };
        let res = LinearScan::allocate(&func, &cfg, &cfg_cfg);
        assert_eq!(uniform(&res, v0), AllocatedSlot::Reg(RDI));
        assert_eq!(uniform(&res, v1), AllocatedSlot::Reg(RDI));
    }

    #[test]
    fn transitive_copy_hint_coalesces_a_three_link_chain_onto_one_preg() {
        let mut func = Func::<X64Inst>::new("chain".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();
        let mut reg_bind = HashMap::new();
        reg_bind.insert(v0, RDI);
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: v0, idx: 0 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: v1, src: v0 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: v2, src: v1 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: v3, src: v2 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v3 });
        }
        let cfg = CFG::compute(&func).unwrap();
        let cfg_cfg = RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RDI, RAX, RBX, RCX],
            scratch_regs: vec![R12, R13],
            allocatable_fp_regs: Vec::new(),
            scratch_fp_regs: Vec::new(),
            reg_bind,
        };
        let res = LinearScan::allocate(&func, &cfg, &cfg_cfg);
        assert_eq!(uniform(&res, v0), AllocatedSlot::Reg(RDI));
        assert_eq!(uniform(&res, v1), AllocatedSlot::Reg(RDI));
        assert_eq!(uniform(&res, v2), AllocatedSlot::Reg(RDI));
        assert_eq!(uniform(&res, v3), AllocatedSlot::Reg(RDI));
    }

    #[test]
    fn pre_bound_vregs_on_the_same_preg_with_disjoint_ranges_both_land_there() {
        let mut func = Func::<X64Inst>::new("t".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v3 = func.new_vreg();
        let mut reg_bind = HashMap::new();
        reg_bind.insert(v0, RAX);
        reg_bind.insert(v3, RAX);
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: v0, idx: 0 });
            bd.push_target_inst(X64Inst::Mov64ri { dst: v3, imm: 1 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v3 });
        }
        let cfg = CFG::compute(&func).unwrap();
        let res = LinearScan::allocate(&func, &cfg, &cfg4(reg_bind));
        assert_eq!(uniform(&res, v0), AllocatedSlot::Reg(RAX));
        assert_eq!(uniform(&res, v3), AllocatedSlot::Reg(RAX));
    }

    #[test]
    fn pre_bind_eviction_splits_the_incumbent_live_range() {
        // v0 defined first, grabs RAX (only allocatable). Then v1 is
        // pre-bound to RAX while v0 is still live. v0's life should be
        // split: Reg(RAX) for the prefix up to v1's first point, Stack(_)
        // for the rest. A SplitMove must be recorded.
        let mut func = Func::<X64Inst>::new("evict".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let mut reg_bind = HashMap::new();
        reg_bind.insert(v1, RAX);
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_target_inst(X64Inst::Mov64ri { dst: v0, imm: 1 });
            bd.push_target_inst(X64Inst::Mov64ri { dst: v1, imm: 2 });
            bd.push_target_inst(X64Inst::Add64rr { dst: v0, src: v1 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: v2, src: v0 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v2 });
        }
        let cfg = CFG::compute(&func).unwrap();
        let cfg_cfg = RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RAX],
            scratch_regs: vec![RBX, R12, R13],
            allocatable_fp_regs: Vec::new(),
            scratch_fp_regs: Vec::new(),
            reg_bind,
        };
        let res = LinearScan::allocate(&func, &cfg, &cfg_cfg);
        assert_eq!(uniform(&res, v1), AllocatedSlot::Reg(RAX));
        let v0_pieces = &res.assignments[v0].pieces;
        assert_eq!(
            v0_pieces.len(),
            2,
            "v0 should be split into Reg-then-Stack pieces"
        );
        assert!(matches!(v0_pieces[0].1, AllocatedSlot::Reg(RAX)));
        assert!(matches!(v0_pieces[1].1, AllocatedSlot::Stack(_)));
        assert_eq!(res.split_moves.len(), 1);
        assert_eq!(res.split_moves[0].from_preg, RAX);
    }

    #[test]
    fn under_pressure_farthest_endpoint_gets_evicted_and_split() {
        let mut func = Func::<X64Inst>::new("p".into());
        let b0 = func.add_empty_block();
        let vs: Vec<Reg> = (0..4).map(|_| func.new_vreg()).collect();
        {
            let bd = func.get_block_data_mut(b0);
            for (i, &v) in vs.iter().enumerate() {
                bd.push_target_inst(X64Inst::Mov64ri { dst: v, imm: i as i64 });
            }
            bd.push_target_inst(X64Inst::Add64rr { dst: vs[3], src: vs[0] });
            bd.push_target_inst(X64Inst::Add64rr { dst: vs[3], src: vs[1] });
            bd.push_target_inst(X64Inst::Add64rr { dst: vs[3], src: vs[2] });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: vs[3] });
        }
        let cfg = CFG::compute(&func).unwrap();
        let cfg_cfg = RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RAX, RBX],
            scratch_regs: vec![R12, R13],
            allocatable_fp_regs: Vec::new(),
            scratch_fp_regs: Vec::new(),
            reg_bind: HashMap::new(),
        };
        let res = LinearScan::allocate(&func, &cfg, &cfg_cfg);
        // At least one vreg should end up with a Stack piece somewhere.
        let any_on_stack = vs.iter().any(|&v| {
            res.assignments[v]
                .slots()
                .any(|s| matches!(s, AllocatedSlot::Stack(_)))
        });
        assert!(any_on_stack, "expected at least one spill under 2-reg pressure");
    }

    #[test]
    fn in_stream_regdef_pins_vreg_same_as_reg_bind() {
        // Same behavior as pre_bind_eviction_splits_the_incumbent_live_range,
        // but the pin is expressed via `RegDef` instead of `reg_bind`.
        let mut func = Func::<X64Inst>::new("regdef-pin".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_target_inst(X64Inst::Mov64ri { dst: v0, imm: 1 });
            bd.push_pseudo_inst(PseudoInstruction::RegDef { vreg: v1, preg: RDI });
            bd.push_target_inst(X64Inst::Mov64ri { dst: v1, imm: 2 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v1 });
        }
        let cfg = CFG::compute(&func).unwrap();
        let res = LinearScan::allocate(&func, &cfg, &cfg4(HashMap::new()));
        assert_eq!(uniform(&res, v1), AllocatedSlot::Reg(RDI));
    }

    #[test]
    fn regdef_agreeing_with_reg_bind_is_fine() {
        let mut func = Func::<X64Inst>::new("agree".into());
        let b0 = func.add_empty_block();
        let v = func.new_vreg();
        let mut reg_bind = HashMap::new();
        reg_bind.insert(v, RDI);
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::RegDef { vreg: v, preg: RDI });
            bd.push_target_inst(X64Inst::Mov64ri { dst: v, imm: 1 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v });
        }
        let cfg = CFG::compute(&func).unwrap();
        let res = LinearScan::allocate(&func, &cfg, &cfg4(reg_bind));
        assert_eq!(uniform(&res, v), AllocatedSlot::Reg(RDI));
    }

    #[test]
    #[should_panic(expected = "pre-bound to both")]
    fn regdef_disagreeing_with_reg_bind_panics() {
        let mut func = Func::<X64Inst>::new("disagree".into());
        let b0 = func.add_empty_block();
        let v = func.new_vreg();
        let mut reg_bind = HashMap::new();
        reg_bind.insert(v, RDI);
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::RegDef { vreg: v, preg: RSI });
            bd.push_target_inst(X64Inst::Mov64ri { dst: v, imm: 1 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v });
        }
        let cfg = CFG::compute(&func).unwrap();
        let _ = LinearScan::allocate(&func, &cfg, &cfg4(reg_bind));
    }

    #[test]
    fn dead_vreg_gets_no_assignment_and_frame_stays_zero() {
        let mut func = Func::<X64Inst>::new("dead".into());
        let b0 = func.add_empty_block();
        let _never = func.new_vreg();
        let live = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_target_inst(X64Inst::Mov64ri { dst: live, imm: 7 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: live });
        }
        let cfg = CFG::compute(&func).unwrap();
        let res = LinearScan::allocate(&func, &cfg, &cfg4(HashMap::new()));
        assert_eq!(res.frame_size, 0);
        assert!(matches!(uniform(&res, live), AllocatedSlot::Reg(_)));
    }
}
