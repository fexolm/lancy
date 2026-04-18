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
use crate::codegen::tir::{Func, Inst, Instruction, PseudoInstruction, Reg};
use crate::support::slotmap::SecondaryMap;

pub struct LinearScan;

impl<I: Inst> RegAllocator<I> for LinearScan {
    fn allocate(func: &Func<I>, cfg: &CFG, config: &RegAllocConfig) -> RegAllocResult {
        let layout = BlockLayout::compute(func);
        Allocator::new(func, cfg, &layout, config).run()
    }
}

struct Allocator<'a> {
    config: &'a RegAllocConfig,
    ranges: LiveRanges,
    copy_src: SecondaryMap<Reg, Option<Reg>>,

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

impl<'a> Allocator<'a> {
    fn new<I: Inst>(
        func: &'a Func<I>,
        cfg: &'a CFG,
        layout: &'a BlockLayout,
        config: &'a RegAllocConfig,
    ) -> Self {
        let ranges = LiveRanges::compute(func, cfg, layout);
        let copy_src = collect_copy_src(func);
        let n = func.get_regs_count();
        let mut assignments = SecondaryMap::new(n);
        assignments.fill(Assignment::default());
        Self {
            config,
            ranges,
            copy_src,
            current_slot: vec![None; n],
            current_piece_start: vec![0; n],
            assignments,
            active: Vec::new(),
            inactive: Vec::new(),
            frame_layout: Vec::new(),
            split_moves: Vec::new(),
        }
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
                let ba = self.config.reg_bind.contains_key(&a);
                let bb = self.config.reg_bind.contains_key(&b);
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
        if let Some(&target) = self.config.reg_bind.get(&v) {
            self.evict_conflicts_on(target, v, position);
            self.assign_fresh_reg(v, target);
            return;
        }

        let v_end = self.ranges[v].last_end().unwrap();
        let blocked_at = self.compute_blocked_at(v, position);

        if let Some(hint) = self.copy_hint(v)
            && blocked_at.get(&hint).copied().unwrap_or(0) >= v_end
        {
            self.assign_fresh_reg(v, hint);
            return;
        }

        let best = self
            .config
            .allocatable_regs
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
        for &p in &self.config.allocatable_regs {
            blocked_at.insert(p, ProgramPoint::MAX);
        }
        for &u in &self.active {
            let p = self.current_preg(u);
            if blocked_at.contains_key(&p) {
                blocked_at.insert(p, position);
            }
        }
        let v_range = &self.ranges[v];
        for &u in &self.inactive {
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

        for &p in &self.config.allocatable_regs {
            let mut blockers: Vec<Reg> = Vec::new();
            for &w in &self.active {
                if self.current_preg(w) == p {
                    blockers.push(w);
                }
            }
            for &w in &self.inactive {
                if self.current_preg(w) == p
                    && self.ranges[w]
                        .next_intersection_at_or_after(v_range, position)
                        .is_some()
                {
                    blockers.push(w);
                }
            }
            if blockers.len() != 1 {
                continue;
            }
            let u = blockers[0];
            if self.config.reg_bind.contains_key(&u) {
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
            if self.current_preg(u) == target && seen.insert(u) {
                conflicts.push(u);
            }
        }
        for &u in &self.inactive {
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
                !self.config.reg_bind.contains_key(&u),
                "pre-bind conflict: vreg {u} also pre-bound to preg {target}, can't evict for vreg {v}"
            );
            self.evict_to_stack(u, position);
            self.active.retain(|&x| x != u);
            self.inactive.retain(|&x| x != u);
        }
    }

    fn copy_hint(&self, v: Reg) -> Option<Reg> {
        let s = self.copy_src.get(v).copied().flatten()?;
        match self.current_slot.get(s as usize)?.as_ref()? {
            AllocatedSlot::Reg(p) if self.config.allocatable_regs.contains(p) => Some(*p),
            _ => None,
        }
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
    /// Closes its Reg piece, opens a Stack piece, records the store move.
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
        self.split_moves.push(SplitMove {
            at_point: split_pt,
            from_preg: p,
            to_slot: s,
        });
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
        for (&v, &p) in &self.config.reg_bind {
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
                        "config.reg_bind pins both vreg {} and vreg {} to preg {} \
                         with overlapping live ranges — one would have to be spilled, \
                         defeating the pre-bind. Fix the frontend to pick disjoint \
                         pregs for concurrently-live pre-binds.",
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
            reg_bind,
        };
        let res = LinearScan::allocate(&func, &cfg, &cfg_cfg);
        assert_eq!(uniform(&res, v0), AllocatedSlot::Reg(RDI));
        assert_eq!(uniform(&res, v1), AllocatedSlot::Reg(RDI));
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
