//! ABI lowering pass for the x64 target (SysV AMD64).
//!
//! **Requires:** SSA-shaped target IR with `PseudoInstruction::Arg { dst, idx }`
//! in the entry block and `PseudoInstruction::Return { src }` as the terminator
//! of return blocks. No checked constraint — contract between passes.
//!
//! **Preserves:** CFG shape. Block indices unchanged.
//!
//! **Invalidates:** The pre-lowering pseudo set (no more `Return`; `Arg` stays
//! as the def-site of a pinned shim, erased later by `pseudo_cleanup`).
//!
//! **Effect:**
//!
//! * `Arg { dst, idx }` → a fresh shim vreg pinned to the ABI arg register
//!   plus `Copy { dst, src: shim }`. The `Arg` pseudo remains as the shim's
//!   def with no use, so regalloc sees a definition at entry.
//! * `Return { src }` → `Copy { dst: ret_vreg, src }; X64Inst::RawRet`, with
//!   `ret_vreg` pinned to the ABI return register.
//!
//! The shim-plus-copy pattern lets regalloc coalesce when possible without
//! constraining the user vreg's entire live range to a specific preg.

use std::collections::HashMap;

use crate::codegen::isa::x64::inst::X64Inst;
use crate::codegen::isa::x64::sysv::SysVAmd64;
use crate::codegen::tir::{Func, Instruction, PseudoInstruction, Reg};

pub struct AbiLowerResult {
    /// `vreg -> preg` for every vreg the ABI pass pinned to a physical
    /// register. Consumed directly by the regalloc config.
    pub reg_bind: HashMap<Reg, Reg>,
}

pub fn lower(func: &mut Func<X64Inst>, cc: SysVAmd64) -> AbiLowerResult {
    let mut reg_bind: HashMap<Reg, Reg> = HashMap::new();
    let block_ids: Vec<_> = func.blocks_iter().map(|(b, _)| b).collect();

    for block in block_ids {
        let old = func.get_block_data_mut(block).take_insts();
        let mut new: Vec<Instruction<X64Inst>> = Vec::with_capacity(old.len());
        for inst in old {
            match inst {
                Instruction::Pseudo(PseudoInstruction::Arg { dst, idx }) => {
                    let preg = cc.int_arg_reg(idx).unwrap_or_else(|| {
                        panic!(
                            "arg idx {idx} exceeds register-passed arg count ({}) — stack args not supported",
                            cc.max_int_args_in_regs()
                        )
                    });
                    let shim = func.new_vreg();
                    reg_bind.insert(shim, preg);
                    new.push(Instruction::Pseudo(PseudoInstruction::Arg { dst: shim, idx }));
                    new.push(Instruction::Pseudo(PseudoInstruction::Copy { dst, src: shim }));
                }
                Instruction::Pseudo(PseudoInstruction::Return { src }) => {
                    let ret_vreg = func.new_vreg();
                    reg_bind.insert(ret_vreg, cc.int_ret_reg());
                    new.push(Instruction::Pseudo(PseudoInstruction::Copy { dst: ret_vreg, src }));
                    new.push(Instruction::Target(X64Inst::RawRet));
                }
                other => new.push(other),
            }
        }
        func.get_block_data_mut(block).set_insts(new);
    }

    AbiLowerResult { reg_bind }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::isa::x64::regs::{RAX, RDI, RSI};
    use crate::codegen::tir::Inst;

    fn build_simple_add() -> Func<X64Inst> {
        let mut func = Func::<X64Inst>::new("add".to_string());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let bd = func.get_block_data_mut(b0);
        bd.push_pseudo_inst(PseudoInstruction::Arg { dst: v0, idx: 0 });
        bd.push_pseudo_inst(PseudoInstruction::Arg { dst: v1, idx: 1 });
        bd.push_target_inst(X64Inst::Add64rr { dst: v0, src: v1 });
        bd.push_pseudo_inst(PseudoInstruction::Return { src: v0 });
        func
    }

    #[test]
    fn lowers_args_to_pinned_shims_and_copies() {
        let mut func = build_simple_add();
        let res = lower(&mut func, SysVAmd64);

        let mut pinned_pregs: Vec<_> = res.reg_bind.values().copied().collect();
        pinned_pregs.sort_unstable();
        assert!(pinned_pregs.contains(&RDI));
        assert!(pinned_pregs.contains(&RSI));
        assert!(pinned_pregs.contains(&RAX));
    }

    #[test]
    fn lowers_return_to_copy_plus_rawret() {
        let mut func = build_simple_add();
        lower(&mut func, SysVAmd64);

        let b0 = func.get_entry_block().unwrap();
        let last_two: Vec<_> = func
            .get_block_data(b0)
            .insts()
            .iter()
            .rev()
            .take(2)
            .collect();
        let term = last_two[0];
        assert!(matches!(term, Instruction::Target(X64Inst::RawRet)));
        assert!(term.is_ret());
        assert!(matches!(
            last_two[1],
            Instruction::Pseudo(PseudoInstruction::Copy { .. })
        ));
    }

    #[test]
    fn no_return_pseudo_remains_after_lowering() {
        let mut func = build_simple_add();
        lower(&mut func, SysVAmd64);
        for (_b, bd) in func.blocks_iter() {
            for inst in bd.iter() {
                assert!(
                    !matches!(inst, Instruction::Pseudo(PseudoInstruction::Return { .. })),
                    "return pseudo should have been lowered: {inst:?}"
                );
            }
        }
    }

    #[test]
    fn arg_pseudo_targets_a_pinned_shim_not_the_original_vreg() {
        let mut func = build_simple_add();
        let res = lower(&mut func, SysVAmd64);

        let b0 = func.get_entry_block().unwrap();
        let mut shim_targets = Vec::new();
        for inst in func.get_block_data(b0).iter() {
            if let Instruction::Pseudo(PseudoInstruction::Arg { dst, .. }) = inst {
                shim_targets.push(*dst);
            }
        }
        assert_eq!(shim_targets.len(), 2);
        for shim in shim_targets {
            assert!(
                res.reg_bind.contains_key(&shim),
                "arg shim vreg {shim} must be pinned"
            );
        }
    }

    #[test]
    #[should_panic(expected = "arg idx 6 exceeds register-passed arg count (6)")]
    fn lowering_panics_on_more_than_six_args() {
        let mut func = Func::<X64Inst>::new("too_many_args".to_string());
        let b0 = func.add_empty_block();
        let bd = func.get_block_data_mut(b0);
        for i in 0..7 {
            let v = 0; // placeholder vreg; lowering only reads `idx`
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: v, idx: i });
        }
        bd.push_target_inst(X64Inst::RawRet);
        lower(&mut func, SysVAmd64);
    }
}
