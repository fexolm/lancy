//! Flat program-point numbering for a function.
//!
//! Each instruction gets two points: an *early* point (program reads its
//! operands here) and a *late* point (program writes its defs here). A
//! half-open interval `[start, end)` with these points then expresses every
//! intra-block liveness question:
//!
//! * A use at inst `i` keeps its operand alive through the early point — end
//!   of that operand's live segment is `early(i) + 1` = `late(i)`.
//! * A def at inst `i` produces its result at the late point — start of that
//!   vreg's live segment is `late(i)`.
//!
//! With this convention, a copy `v2 = copy v1` at inst `i` where `v1` dies
//! here produces segments that *just touch* (`v1.end == v2.start`), which is
//! what enables coalescing: they never overlap at any integer point.
//!
//! Block order is `func.blocks_iter()` insertion order — the same order the
//! MC emitter walks. Any consistent order works; we just need one.

use crate::codegen::tir::{Block, Func, Inst};
use crate::support::slotmap::SecondaryMap;

pub type ProgramPoint = u32;

pub const POINTS_PER_INST: u32 = 2;

/// Precomputed block → (first global inst, one-past-last) mapping.
///
/// For a block with 3 insts and `first_inst[B] = 10`, the instructions have
/// global indices 10, 11, 12. Early points are 20, 22, 24; late points are
/// 21, 23, 25. `last_inst[B] = 13`. `end_point(B) = 26`.
pub struct BlockLayout {
    pub order: Vec<Block>,
    first_inst: SecondaryMap<Block, u32>,
    last_inst: SecondaryMap<Block, u32>, // exclusive
    total_insts: u32,
}

impl BlockLayout {
    #[must_use]
    pub fn compute<I: Inst>(func: &Func<I>) -> Self {
        let n = func.blocks_count();
        let mut first_inst = SecondaryMap::new(n);
        let mut last_inst = SecondaryMap::new(n);
        let mut order = Vec::with_capacity(n);
        let mut cursor: u32 = 0;
        for (b, bd) in func.blocks_iter() {
            order.push(b);
            first_inst.set(b, cursor);
            cursor += bd.len() as u32;
            last_inst.set(b, cursor);
        }
        Self {
            order,
            first_inst,
            last_inst,
            total_insts: cursor,
        }
    }

    #[must_use]
    pub fn total_insts(&self) -> u32 {
        self.total_insts
    }

    /// First program point of the block — the early point of its first inst.
    #[must_use]
    pub fn block_start_pt(&self, b: Block) -> ProgramPoint {
        self.first_inst[b] * POINTS_PER_INST
    }

    /// One-past-the-last program point of the block — the late point of its
    /// last inst, plus one. Equals `block_start_pt(B_next)` if B is followed
    /// immediately in the layout.
    #[must_use]
    pub fn block_end_pt(&self, b: Block) -> ProgramPoint {
        self.last_inst[b] * POINTS_PER_INST
    }

    /// Early (use) point of the `inst_idx`-th instruction in block `b`.
    #[must_use]
    pub fn use_pt(&self, b: Block, inst_idx: u32) -> ProgramPoint {
        (self.first_inst[b] + inst_idx) * POINTS_PER_INST
    }

    /// Late (def) point of the `inst_idx`-th instruction in block `b`.
    #[must_use]
    pub fn def_pt(&self, b: Block, inst_idx: u32) -> ProgramPoint {
        (self.first_inst[b] + inst_idx) * POINTS_PER_INST + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::isa::x64::inst::X64Inst;
    use crate::codegen::tir::PseudoInstruction;

    #[test]
    fn point_layout_is_half_open_and_copy_segments_touch() {
        let mut func = Func::<X64Inst>::new("t".into());
        let b = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_target_inst(X64Inst::Mov64ri { dst: v0, imm: 1 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: v1, src: v0 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v1 });
        }
        let layout = BlockLayout::compute(&func);
        // v0 def at inst 0 late point = 1.
        // v0 last-use at inst 1 early point = 2. Segment [1, 3).
        // v1 def at inst 1 late point = 3. v1 last-use at inst 2 early = 4.
        // Segment [3, 5). Note 3 is the shared boundary — half-open
        // semantics means segments don't overlap at any integer point, so
        // coalescing onto the same preg is safe.
        assert_eq!(layout.def_pt(b, 0), 1);
        assert_eq!(layout.use_pt(b, 1), 2);
        assert_eq!(layout.def_pt(b, 1), 3);
        assert_eq!(layout.use_pt(b, 2), 4);
    }

    #[test]
    fn block_start_and_end_bracket_all_point_numbers() {
        let mut func = Func::<X64Inst>::new("t".into());
        let a = func.add_empty_block();
        let b = func.add_empty_block();
        let v = func.new_vreg();
        {
            let bd = func.get_block_data_mut(a);
            bd.push_target_inst(X64Inst::Mov64ri { dst: v, imm: 1 });
            bd.push_target_inst(X64Inst::Jmp { dst: b });
        }
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v });
        }
        let layout = BlockLayout::compute(&func);
        assert_eq!(layout.block_start_pt(a), 0);
        assert_eq!(layout.block_end_pt(a), 4); // 2 insts * 2 points
        assert_eq!(layout.block_start_pt(b), 4);
        assert_eq!(layout.block_end_pt(b), 6);
    }
}
