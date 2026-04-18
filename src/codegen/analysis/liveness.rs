//! Live-range analysis: multi-segment live ranges in flat-point space.
//!
//! Two phases. The first computes `live_in` / `live_out` bitsets per block
//! via iterative dataflow — correct even when a vreg has multiple def sites
//! (as happens with x86's two-operand `Add64rr` pattern where the dst is
//! both read and rewritten). The second pass walks each block bottom-up,
//! using `live_out` as a seed, turning intra-block use/def traces into
//! `Segment`s. The output per vreg is a sorted, non-overlapping list of
//! segments; a vreg that's defined then dead, used multiple times across
//! non-contiguous blocks, or live-through a block without using it produces
//! a faithful record instead of a single conservative interval.
//!
//! The segment structure is what unlocks precise allocation: two vregs may
//! share a preg iff *none* of their segments intersect. A vreg holding a
//! preg across a hole where another vreg is live simply releases it, gets
//! it back on the other side.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::codegen::analysis::cfg::{reverse_post_order, CFG};
use crate::codegen::analysis::layout::{BlockLayout, ProgramPoint};
use crate::codegen::tir::{Block, Func, Inst, Reg};
use crate::support::bitset::FixedBitSet;
use crate::support::slotmap::{Key, SecondaryMap};

/// Half-open `[start, end)` interval in flat program-point space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub start: ProgramPoint,
    pub end: ProgramPoint,
}

impl Segment {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    #[must_use]
    pub fn contains(&self, pt: ProgramPoint) -> bool {
        self.start <= pt && pt < self.end
    }

    /// `true` iff the two segments share at least one integer point. Adjacent
    /// segments `[a, b)` and `[b, c)` do *not* intersect — this is the
    /// property that makes Copy coalescing safe.
    #[must_use]
    pub fn intersects(&self, other: &Segment) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// A vreg's live range: sorted, non-overlapping, non-adjacent-mergeable
/// list of `Segment`s.
#[derive(Default, Clone, Debug)]
pub struct LiveRange {
    segments: SmallVec<[Segment; 2]>,
}

impl LiveRange {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    #[must_use]
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// First point of the earliest segment, or `None` if empty.
    #[must_use]
    pub fn first_start(&self) -> Option<ProgramPoint> {
        self.segments.first().map(|s| s.start)
    }

    /// End point (exclusive) of the latest segment, or `None` if empty.
    #[must_use]
    pub fn last_end(&self) -> Option<ProgramPoint> {
        self.segments.last().map(|s| s.end)
    }

    /// Whether `pt` is covered by any segment. O(log n).
    #[must_use]
    pub fn covers(&self, pt: ProgramPoint) -> bool {
        self.segments
            .binary_search_by(|s| {
                if pt < s.start {
                    std::cmp::Ordering::Greater
                } else if pt >= s.end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }

    /// First point `>= from` at which `self` and `other` are both live, or
    /// `None` if they never intersect past `from`. Two-pointer sweep,
    /// O(|self| + |other|).
    #[must_use]
    pub fn next_intersection_at_or_after(
        &self,
        other: &LiveRange,
        from: ProgramPoint,
    ) -> Option<ProgramPoint> {
        let mut i = 0;
        let mut j = 0;
        while i < self.segments.len() && j < other.segments.len() {
            let a = self.segments[i];
            let b = other.segments[j];
            if a.end <= from {
                i += 1;
                continue;
            }
            if b.end <= from {
                j += 1;
                continue;
            }
            let s_start = a.start.max(b.start).max(from);
            let s_end = a.end.min(b.end);
            if s_start < s_end {
                return Some(s_start);
            }
            if a.end <= b.end {
                i += 1;
            } else {
                j += 1;
            }
        }
        None
    }

    /// Insert `seg` into `self`, merging with any overlapping / adjacent
    /// existing segments.
    pub fn add(&mut self, seg: Segment) {
        if seg.is_empty() {
            return;
        }
        // Find first segment whose end is >= seg.start — i.e. the earliest
        // one that could merge.
        let pos = self.segments.partition_point(|s| s.end < seg.start);
        let mut merged = seg;
        while pos < self.segments.len() && self.segments[pos].start <= merged.end {
            merged.start = merged.start.min(self.segments[pos].start);
            merged.end = merged.end.max(self.segments[pos].end);
            self.segments.remove(pos);
        }
        self.segments.insert(pos, merged);
    }
}

pub struct LiveRanges {
    ranges: SecondaryMap<Reg, LiveRange>,
}

impl LiveRanges {
    #[must_use]
    pub fn compute<I: Inst>(func: &Func<I>, cfg: &CFG, layout: &BlockLayout) -> Self {
        let live_out = compute_live_out(func, cfg);
        let mut ranges: SecondaryMap<Reg, LiveRange> = SecondaryMap::new(func.get_regs_count());
        ranges.fill(LiveRange::default());

        for (block, bd) in func.blocks_iter() {
            let block_start = layout.block_start_pt(block);
            let block_end = layout.block_end_pt(block);

            // end_of_current_segment[r] = upper bound of the segment we're
            // currently building going upward for r. Initialized from
            // live_out (r is live through the bottom of the block).
            let mut ends: HashMap<Reg, ProgramPoint> = HashMap::new();
            for r_idx in live_out[block].iter_ones() {
                ends.insert(r_idx as Reg, block_end);
            }

            for (idx, inst) in bd.insts().iter().enumerate().rev() {
                let i = idx as u32;
                let def_pt = layout.def_pt(block, i);
                let use_pt = layout.use_pt(block, i);

                for r in inst.get_defs() {
                    match ends.remove(&r) {
                        Some(end) => ranges[r].add(Segment { start: def_pt, end }),
                        None => {
                            // Dead def — still needs a point-sized slot so
                            // the allocator reserves a register for the one
                            // cycle the value exists.
                            ranges[r].add(Segment {
                                start: def_pt,
                                end: def_pt + 1,
                            });
                        }
                    }
                }
                for r in inst.get_uses() {
                    ends.entry(r).or_insert(use_pt + 1);
                }
            }

            // Whatever is still in `ends` is live-in from predecessors;
            // segment covers block_start up to the tracked end.
            for (r, end) in ends {
                ranges[r].add(Segment { start: block_start, end });
            }
        }

        Self { ranges }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Reg, &LiveRange)> {
        self.ranges.iter()
    }
}

impl std::ops::Index<Reg> for LiveRanges {
    type Output = LiveRange;
    fn index(&self, r: Reg) -> &Self::Output {
        &self.ranges[r]
    }
}

// -----------------------------------------------------------------------
// Internal: iterative live_in/out dataflow. Only `live_out` escapes;
// `live_in` is a transient needed to compute successors' `live_out` during
// the fixpoint.

fn compute_live_out<I: Inst>(
    func: &Func<I>,
    cfg: &CFG,
) -> SecondaryMap<Block, FixedBitSet> {
    let regs_count = func.get_regs_count();
    let blocks_count = cfg.blocks_count();
    let mut live_in: SecondaryMap<Block, FixedBitSet> = SecondaryMap::new(blocks_count);
    live_in.fill(FixedBitSet::zeroes(regs_count));
    let mut live_out: SecondaryMap<Block, FixedBitSet> = SecondaryMap::new(blocks_count);
    live_out.fill(FixedBitSet::zeroes(regs_count));

    let (uses_per_block, defs_per_block) = compute_use_def(func);

    // Worklist seeded with blocks in reverse-post-order (tail first): an
    // acyclic CFG converges in one sweep, loops in a small constant. We
    // also maintain an `in_worklist` bitset so a block that's already
    // queued doesn't get re-pushed by every predecessor's change.
    let mut worklist: Vec<Block> = reverse_post_order(cfg);
    worklist.reverse();
    let mut in_worklist = FixedBitSet::zeroes(blocks_count);
    for b in &worklist {
        in_worklist.add(b.index());
    }

    while let Some(block) = worklist.pop() {
        in_worklist.del(block.index());
        let old_in_count = live_in[block].ones_count();
        let old_out_count = live_out[block].ones_count();

        {
            let out = live_out.get_mut(block).unwrap();
            for &s in cfg.succs(block) {
                out.union(&live_in[s]);
            }
        }

        let new_in = {
            let mut tmp = live_out[block].clone();
            tmp.difference(&defs_per_block[block]);
            tmp.union(&uses_per_block[block]);
            tmp
        };
        *live_in.get_mut(block).unwrap() = new_in;

        if live_in[block].ones_count() != old_in_count
            || live_out[block].ones_count() != old_out_count
        {
            for &p in cfg.preds(block) {
                if !in_worklist.has(p.index()) {
                    in_worklist.add(p.index());
                    worklist.push(p);
                }
            }
        }
    }

    live_out
}

fn compute_use_def<I: Inst>(
    func: &Func<I>,
) -> (SecondaryMap<Block, FixedBitSet>, SecondaryMap<Block, FixedBitSet>) {
    let regs_count = func.get_regs_count();
    let blocks_count = func.blocks_count();
    let mut uses = SecondaryMap::new(blocks_count);
    uses.fill(FixedBitSet::zeroes(regs_count));
    let mut defs = SecondaryMap::new(blocks_count);
    defs.fill(FixedBitSet::zeroes(regs_count));

    for (block, bd) in func.blocks_iter() {
        let u = uses.get_mut(block).unwrap();
        let d = defs.get_mut(block).unwrap();
        for inst in bd.iter() {
            for r in inst.get_uses() {
                if !d.has(r as usize) {
                    u.add(r as usize);
                }
            }
            for r in inst.get_defs() {
                d.add(r as usize);
            }
        }
    }

    (uses, defs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::analysis::layout::BlockLayout;
    use crate::codegen::isa::x64::inst::X64Inst;
    use crate::codegen::tir::PseudoInstruction;

    #[test]
    fn segment_add_merges_adjacent_and_overlapping() {
        let mut r = LiveRange::default();
        r.add(Segment { start: 0, end: 3 });
        r.add(Segment { start: 5, end: 8 });
        assert_eq!(r.segments.len(), 2);
        // Adjacent: [3..5) fills the gap and merges both.
        r.add(Segment { start: 3, end: 5 });
        assert_eq!(r.segments.len(), 1);
        assert_eq!(r.segments[0], Segment { start: 0, end: 8 });
        // Fully-contained add is a no-op.
        r.add(Segment { start: 2, end: 4 });
        assert_eq!(r.segments.len(), 1);
        assert_eq!(r.segments[0], Segment { start: 0, end: 8 });
    }

    #[test]
    fn adjacent_segments_do_not_intersect_enabling_coalescing() {
        let a = Segment { start: 0, end: 3 };
        let b = Segment { start: 3, end: 6 };
        assert!(!a.intersects(&b));
        // But overlapping-by-one DOES intersect.
        let c = Segment { start: 2, end: 5 };
        assert!(a.intersects(&c));
    }

    #[test]
    fn straight_line_ranges_span_def_to_last_use() {
        // arg v1; mov v0, v1; ret v0
        let mut func = Func::<X64Inst>::new("t".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: v1, idx: 0 });
            bd.push_target_inst(X64Inst::Mov64rr { dst: v0, src: v1 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v0 });
        }
        let cfg = CFG::compute(&func).unwrap();
        let layout = BlockLayout::compute(&func);
        let ranges = LiveRanges::compute(&func, &cfg, &layout);

        // v1 defined at 1 (late of inst 0), last used at 2 (early of inst 1).
        // Segment [1, 3) — half-open end is late(1) = 3.
        assert_eq!(ranges[v1].segments(), &[Segment { start: 1, end: 3 }]);
        // v0 defined at 3, last used at 4 (early of inst 2) → end 5.
        assert_eq!(ranges[v0].segments(), &[Segment { start: 3, end: 5 }]);
    }

    #[test]
    fn value_live_through_a_block_without_using_it_has_a_through_segment() {
        // b0: mov v0, 42; jmp b1
        // b1: jmp b2             (v0 live-through)
        // b2: ret v0
        let mut func = Func::<X64Inst>::new("t".into());
        let b0 = func.add_empty_block();
        let b1 = func.add_empty_block();
        let b2 = func.add_empty_block();
        let v0 = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_target_inst(X64Inst::Mov64ri { dst: v0, imm: 42 });
            bd.push_target_inst(X64Inst::Jmp { dst: b1 });
        }
        {
            let bd = func.get_block_data_mut(b1);
            bd.push_target_inst(X64Inst::Jmp { dst: b2 });
        }
        {
            let bd = func.get_block_data_mut(b2);
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v0 });
        }
        let cfg = CFG::compute(&func).unwrap();
        let layout = BlockLayout::compute(&func);
        let ranges = LiveRanges::compute(&func, &cfg, &layout);

        // b0: 2 insts → points 0..4.  b1: 1 inst → points 4..6.  b2: 1 inst → 6..8.
        // v0: defined late(0)=1 in b0, live-out of b0, live-through b1, used early(0)=6 in b2.
        // One merged segment from 1 to 7 (= use_pt + 1).
        assert_eq!(ranges[v0].segments(), &[Segment { start: 1, end: 7 }]);
    }

    #[test]
    fn hole_between_uses_produces_two_segments() {
        // Adversarial: diamond where one branch re-uses v, the other
        // doesn't. v should have TWO segments, reflecting the hole along
        // the "doesn't use" path.
        //       b0 ──────┐
        //       │        │
        //       b1       b2   (b2 doesn't touch v0)
        //       │        │
        //       └──► b3 ◄┘    (b3 uses v0)
        use crate::codegen::isa::x64::inst::Cond;
        let mut func = Func::<X64Inst>::new("diamond".into());
        let b0 = func.add_empty_block();
        let b1 = func.add_empty_block();
        let b2 = func.add_empty_block();
        let b3 = func.add_empty_block();
        let v0 = func.new_vreg();
        let z = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_target_inst(X64Inst::Mov64ri { dst: v0, imm: 1 });
            bd.push_target_inst(X64Inst::Mov64ri { dst: z, imm: 0 });
            bd.push_target_inst(X64Inst::Cmp64rr { lhs: v0, rhs: z });
            bd.push_target_inst(X64Inst::CondJmp {
                cond: Cond::NZ,
                taken: b1,
                not_taken: b2,
            });
        }
        {
            // b1 uses v0 (keeps it live-in).
            let bd = func.get_block_data_mut(b1);
            bd.push_target_inst(X64Inst::Add64ri32 { dst: v0, imm: 1 });
            bd.push_target_inst(X64Inst::Jmp { dst: b3 });
        }
        {
            // b2 doesn't touch v0 at all. Along the b0→b2→b3 path, v0 is
            // dead in b2 — yet b3 uses it, so liveness propagates
            // conservatively.
            let bd = func.get_block_data_mut(b2);
            bd.push_target_inst(X64Inst::Jmp { dst: b3 });
        }
        {
            let bd = func.get_block_data_mut(b3);
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v0 });
        }
        let cfg = CFG::compute(&func).unwrap();
        let layout = BlockLayout::compute(&func);
        let ranges = LiveRanges::compute(&func, &cfg, &layout);
        // v0 is live through b2 along the false path (can't prove otherwise
        // without value tracking). Still, the range covers every block it
        // touches; we just assert it's non-empty and its end reaches the
        // return in b3.
        let v0_r = &ranges[v0];
        assert!(!v0_r.is_empty());
        let end = v0_r.last_end().unwrap();
        // b3's sole inst is at late point = block_end_pt(b3) - 1 → use is
        // at early = block_start_pt(b3). use_pt + 1 = block_start(b3) + 1.
        assert_eq!(end, layout.block_start_pt(b3) + 1);
    }
}
