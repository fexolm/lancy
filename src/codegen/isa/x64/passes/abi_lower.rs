//! ABI lowering for the x64 target. One `AbiLowering<X64Inst>` implementor
//! per calling convention — today, `SysVAmd64Lowering`.
//!
//! **Requires:** SSA-shaped target IR with `PseudoInstruction::Arg { dst, idx }`
//! in the entry block and `PseudoInstruction::Return { src }` as the terminator
//! of return blocks.
//!
//! **Preserves:** CFG shape.
//!
//! **Invalidates:** The pre-lowering pseudo set (no more `Return`; `Arg` stays
//! as the def-site of a pinned shim, erased later by `pseudo_cleanup`).
//!
//! **Effect:**
//!
//! * `Arg { dst, idx }` → a fresh shim vreg pinned to the ABI arg register
//!   plus `Copy { dst, src: shim }`. The shim's short life plus the dangling
//!   `Copy` make coalescing in regalloc straightforward.
//! * `Return { src }` → `Copy { dst: ret_vreg, src }; X64Inst::RawRet` with
//!   `ret_vreg` pinned to the ABI return register.

use std::collections::HashMap;

use crate::codegen::isa::x64::inst::X64Inst;
use crate::codegen::isa::x64::regs::{R10, R11, RAX};
use crate::codegen::isa::x64::sysv::{INT_ARG_REGS, SysVAmd64};
use crate::codegen::passes::{AbiLowering, AbiLowerResult, CallSite};
use crate::codegen::tir::{CallTarget, Func, Instruction, PseudoInstruction, Reg};

pub struct SysVAmd64Lowering;

impl AbiLowering<X64Inst> for SysVAmd64Lowering {
    fn lower(&self, func: &mut Func<X64Inst>) -> AbiLowerResult {
        let cc = SysVAmd64;
        let mut reg_bind: HashMap<Reg, Reg> = HashMap::new();
        let mut call_sites: Vec<CallSite> = Vec::new();
        let block_ids: Vec<_> = func.blocks_iter().map(|(b, _)| b).collect();

        for block in block_ids {
            let old = func.get_block_data_mut(block).take_insts();
            let mut new: Vec<Instruction<X64Inst>> = Vec::with_capacity(old.len());
            for inst in old {
                match inst {
                    Instruction::Pseudo(PseudoInstruction::Arg { dst, idx }) => {
                        let preg = cc.int_arg_reg(idx).unwrap_or_else(|| {
                            panic!(
                                "arg idx {idx} exceeds register-passed arg count ({}) \
                                 — stack args not supported",
                                cc.max_int_args_in_regs()
                            )
                        });
                        let shim = func.new_vreg();
                        reg_bind.insert(shim, preg);
                        new.push(Instruction::Pseudo(PseudoInstruction::Arg {
                            dst: shim,
                            idx,
                        }));
                        new.push(Instruction::Pseudo(PseudoInstruction::Copy {
                            dst,
                            src: shim,
                        }));
                    }
                    Instruction::Pseudo(PseudoInstruction::Return { src }) => {
                        let ret_vreg = func.new_vreg();
                        reg_bind.insert(ret_vreg, cc.int_ret_reg());
                        new.push(Instruction::Pseudo(PseudoInstruction::Copy {
                            dst: ret_vreg,
                            src,
                        }));
                        new.push(Instruction::Target(X64Inst::RawRet));
                    }
                    Instruction::Pseudo(PseudoInstruction::CallPseudo { id }) => {
                        lower_call(
                            id,
                            func,
                            &mut new,
                            &mut reg_bind,
                            &mut call_sites,
                        );
                    }
                    other => new.push(other),
                }
            }
            func.get_block_data_mut(block).set_insts(new);
        }

        AbiLowerResult { reg_bind, call_sites }
    }
}

fn lower_call(
    id: crate::codegen::tir::CallId,
    func: &mut Func<X64Inst>,
    new: &mut Vec<Instruction<X64Inst>>,
    reg_bind: &mut HashMap<Reg, Reg>,
    call_sites: &mut Vec<CallSite>,
) {
    // Snapshot the CallData's fields we need; the side-table might be
    // mutated below if we ever add spill vregs.
    let call_data = func.call_operands(id).clone();
    let args = call_data.args;
    let rets = call_data.rets;
    assert!(
        args.len() <= INT_ARG_REGS.len(),
        "stack-passed args not yet supported ({} args > {})",
        args.len(),
        INT_ARG_REGS.len()
    );
    assert!(
        rets.len() <= 1,
        "multiple-return calls not yet supported"
    );

    // Copy each user-arg vreg into a fresh shim vreg pinned to the
    // SysV arg preg. The shim's life is from Copy to Call64r, so it
    // occupies the arg preg across that window.
    let mut arg_shims: Vec<Reg> = Vec::with_capacity(args.len());
    for (i, user_arg) in args.iter().copied().enumerate() {
        let shim = func.new_vreg();
        reg_bind.insert(shim, INT_ARG_REGS[i]);
        new.push(Instruction::Pseudo(PseudoInstruction::Copy {
            dst: shim,
            src: user_arg,
        }));
        arg_shims.push(shim);
    }

    // Clobber markers for caller-saved pregs NOT holding the first
    // `args.len()` arg regs. Each clobber vreg is defined and pinned
    // to its preg at a point right before the call's target load —
    // any user vreg still live in that preg must get evicted /
    // split-spilled ahead of this point so the JIT-callee can stomp
    // on the preg freely.
    //
    for &preg in &[R10, R11] {
        emit_clobber(func, new, reg_bind, preg);
    }
    for (i, &arg_preg) in INT_ARG_REGS.iter().enumerate() {
        if i >= args.len() {
            emit_clobber(func, new, reg_bind, arg_preg);
        }
    }
    emit_clobber(func, new, reg_bind, RAX);

    // Load callee address. The immediate is a placeholder the loader
    // patches at Module::load time.
    let addr_vreg = func.new_vreg();
    new.push(Instruction::Target(X64Inst::Mov64ri {
        dst: addr_vreg,
        imm: 0, // placeholder, patched at load time
    }));

    // Emit the call.
    new.push(Instruction::Target(X64Inst::Call64r { target: addr_vreg }));

    // Extract the return value: define ret_shim pinned to RAX, copy
    // into the user's return vreg.
    if let Some(&user_ret) = rets.first() {
        let ret_shim = func.new_vreg();
        reg_bind.insert(ret_shim, RAX);
        new.push(Instruction::Pseudo(PseudoInstruction::RegDef {
            vreg: ret_shim,
            preg: RAX,
        }));
        new.push(Instruction::Pseudo(PseudoInstruction::Copy {
            dst: user_ret,
            src: ret_shim,
        }));
    }

    // Record this call's symbol-patch request.
    let symbol = match call_data.callee {
        CallTarget::Symbol(s) => s,
        CallTarget::Indirect(_) => String::new(), // no patching needed
    };
    call_sites.push(CallSite { addr_vreg, symbol });
}

fn emit_clobber(
    func: &mut Func<X64Inst>,
    new: &mut Vec<Instruction<X64Inst>>,
    reg_bind: &mut HashMap<Reg, Reg>,
    preg: Reg,
) {
    let v = func.new_vreg();
    reg_bind.insert(v, preg);
    new.push(Instruction::Pseudo(PseudoInstruction::ImplicitDef { dst: v }));
    new.push(Instruction::Pseudo(PseudoInstruction::RegDef {
        vreg: v,
        preg,
    }));
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
        let res = SysVAmd64Lowering.lower(&mut func);

        let mut pinned_pregs: Vec<_> = res.reg_bind.values().copied().collect();
        pinned_pregs.sort_unstable();
        assert!(pinned_pregs.contains(&RDI));
        assert!(pinned_pregs.contains(&RSI));
        assert!(pinned_pregs.contains(&RAX));
    }

    #[test]
    fn lowers_return_to_copy_plus_rawret() {
        let mut func = build_simple_add();
        SysVAmd64Lowering.lower(&mut func);

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
        SysVAmd64Lowering.lower(&mut func);
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
        let res = SysVAmd64Lowering.lower(&mut func);

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
            let v = 0;
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: v, idx: i });
        }
        bd.push_target_inst(X64Inst::RawRet);
        SysVAmd64Lowering.lower(&mut func);
    }
}
