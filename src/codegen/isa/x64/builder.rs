//! IR builder (v0 frontend) for the x64 target. Produces `Func<X64Inst>` in
//! SSA shape using `PseudoInstruction` for ABI-neutral concerns.
//!
//! Each builder method emits into the current block the minimal set of
//! instructions to implement the described operation. Arithmetic ops emit a
//! `Copy` before the two-operand target instruction so the frontend sees a
//! three-operand illusion.

use crate::codegen::isa::x64::inst::{Cond, X64Inst};
use crate::codegen::tir::{Block, Func, PseudoInstruction, Reg};

pub struct FuncBuilder {
    func: Func<X64Inst>,
    entry: Block,
    current: Block,
    arg_count: u32,
}

impl FuncBuilder {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        let mut func = Func::<X64Inst>::new(name.into());
        let entry = func.add_empty_block();
        Self {
            func,
            entry,
            current: entry,
            arg_count: 0,
        }
    }

    #[must_use]
    pub fn entry_block(&self) -> Block {
        self.entry
    }

    #[must_use]
    pub fn current_block(&self) -> Block {
        self.current
    }

    pub fn switch_to_block(&mut self, block: Block) {
        self.current = block;
    }

    pub fn new_block(&mut self) -> Block {
        self.func.add_empty_block()
    }

    /// Define the next incoming argument. Must be called on the entry block.
    /// Returns a fresh vreg that holds the argument value.
    pub fn arg(&mut self) -> Reg {
        assert_eq!(
            self.current, self.entry,
            "arg() must be called while positioned on the entry block"
        );
        let dst = self.func.new_vreg();
        let idx = self.arg_count;
        self.arg_count += 1;
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::Arg { dst, idx });
        dst
    }

    pub fn iconst64(&mut self, imm: i64) -> Reg {
        let dst = self.func.new_vreg();
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Mov64ri { dst, imm });
        dst
    }

    fn binop_rr<F>(&mut self, a: Reg, b: Reg, make_inst: F) -> Reg
    where
        F: FnOnce(Reg, Reg) -> X64Inst,
    {
        let dst = self.func.new_vreg();
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst, src: a });
        bd.push_target_inst(make_inst(dst, b));
        dst
    }

    pub fn add(&mut self, a: Reg, b: Reg) -> Reg {
        self.binop_rr(a, b, |dst, src| X64Inst::Add64rr { dst, src })
    }

    pub fn sub(&mut self, a: Reg, b: Reg) -> Reg {
        self.binop_rr(a, b, |dst, src| X64Inst::Sub64rr { dst, src })
    }

    pub fn imul(&mut self, a: Reg, b: Reg) -> Reg {
        self.binop_rr(a, b, |dst, src| X64Inst::Imul64rr { dst, src })
    }

    /// Emit `cmp a, b` and a `CondJmp` that terminates the current block.
    /// After this call the builder is not positioned on any block; the caller
    /// must `switch_to_block` before emitting further instructions.
    pub fn branch_icmp(&mut self, cond: Cond, a: Reg, b: Reg, taken: Block, not_taken: Block) {
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_target_inst(X64Inst::Cmp64rr { lhs: a, rhs: b });
        bd.push_target_inst(X64Inst::CondJmp { cond, taken, not_taken });
    }

    pub fn jmp(&mut self, dst: Block) {
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Jmp { dst });
    }

    pub fn ret(&mut self, src: Reg) {
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::Return { src });
    }

    #[must_use]
    pub fn build(self) -> Func<X64Inst> {
        self.func
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::tir::Inst;

    #[test]
    fn add_emits_copy_then_add_and_returns_fresh_vreg() {
        let mut b = FuncBuilder::new("t");
        let a = b.arg();
        let c = b.arg();
        let s = b.add(a, c);
        assert_ne!(s, a);
        assert_ne!(s, c);
        let entry = b.entry_block();
        let f = b.build();
        let insts: Vec<_> = f.get_block_data(entry).iter().copied().collect();
        assert_eq!(insts.len(), 4);
        let copy = &insts[2];
        let add = &insts[3];
        assert_eq!(copy.get_defs().as_slice(), &[s]);
        assert_eq!(copy.get_uses().as_slice(), &[a]);
        assert_eq!(add.get_defs().as_slice(), &[s]);
        assert_eq!(add.get_uses().as_slice(), &[s, c]);
    }

    #[test]
    fn arg_indices_increase_monotonically() {
        let mut b = FuncBuilder::new("t");
        let _a0 = b.arg();
        let _a1 = b.arg();
        let _a2 = b.arg();
        let entry = b.entry_block();
        let f = b.build();
        let mut seen_idx = Vec::new();
        for inst in f.get_block_data(entry).iter() {
            if let crate::codegen::tir::Instruction::Pseudo(PseudoInstruction::Arg { idx, .. }) =
                inst
            {
                seen_idx.push(*idx);
            }
        }
        assert_eq!(seen_idx, vec![0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "arg() must be called while positioned on the entry block")]
    fn arg_not_on_entry_panics() {
        let mut b = FuncBuilder::new("t");
        let other = b.new_block();
        b.switch_to_block(other);
        let _ = b.arg();
    }
}
