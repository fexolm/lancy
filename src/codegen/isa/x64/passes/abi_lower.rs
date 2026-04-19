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
use crate::codegen::isa::x64::regs::{
    R10, R11, RAX, XMM0, XMM1, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, XMM2, XMM3, XMM4, XMM5,
    XMM6, XMM7, XMM8, XMM9,
};
use crate::codegen::isa::x64::sysv::{FP_ARG_REGS, INT_ARG_REGS, SysVAmd64};
use crate::codegen::passes::{AbiLowering, AbiLowerResult, CallSite};
use crate::codegen::tir::{CallTarget, Func, Instruction, PseudoInstruction, Reg, Type};

pub struct SysVAmd64Lowering;

impl AbiLowering<X64Inst> for SysVAmd64Lowering {
    fn lower(&self, func: &mut Func<X64Inst>) -> AbiLowerResult {
        let cc = SysVAmd64;
        let mut reg_bind: HashMap<Reg, Reg> = HashMap::new();
        let mut call_sites: Vec<CallSite> = Vec::new();
        let block_ids: Vec<_> = func.blocks_iter().map(|(b, _)| b).collect();

        // SysV gives integer and FP args separate position counters.
        // Walk entry-block Arg pseudos in declaration order and map
        // each `idx` into its class-relative counter.
        let mut int_pos: u32 = 0;
        let mut fp_pos: u32 = 0;

        for block in block_ids {
            let old = func.get_block_data_mut(block).take_insts();
            let mut new: Vec<Instruction<X64Inst>> = Vec::with_capacity(old.len());
            for inst in old {
                match inst {
                    Instruction::Pseudo(PseudoInstruction::Arg { dst, idx }) => {
                        let is_fp = func.vreg_type(dst).is_fp_or_vector();
                        if is_fp {
                            if let Some(preg) = cc.fp_arg_reg(fp_pos) {
                                let shim = func.new_typed_vreg(func.vreg_type(dst));
                                reg_bind.insert(shim, preg);
                                new.push(Instruction::Pseudo(PseudoInstruction::Arg {
                                    dst: shim,
                                    idx,
                                }));
                                new.push(Instruction::Pseudo(PseudoInstruction::Copy {
                                    dst,
                                    src: shim,
                                }));
                                fp_pos += 1;
                            } else {
                                // Stack-passed FP args — unsupported today;
                                // the frontend should cap FP arg count at 8.
                                panic!(
                                    "more than {} FP args: stack-passed FP args unimplemented",
                                    cc.max_fp_args_in_regs()
                                );
                            }
                        } else if let Some(preg) = cc.int_arg_reg(int_pos) {
                            let shim = func.new_typed_vreg(func.vreg_type(dst));
                            reg_bind.insert(shim, preg);
                            new.push(Instruction::Pseudo(PseudoInstruction::Arg {
                                dst: shim,
                                idx,
                            }));
                            new.push(Instruction::Pseudo(PseudoInstruction::Copy {
                                dst,
                                src: shim,
                            }));
                            int_pos += 1;
                        } else {
                            // Stack-passed integer argument.
                            let stack_idx = int_pos - cc.max_int_args_in_regs();
                            new.push(Instruction::Target(X64Inst::LoadArgFromStack {
                                dst,
                                stack_idx,
                            }));
                            int_pos += 1;
                        }
                    }
                    Instruction::Pseudo(PseudoInstruction::Return { src }) => {
                        let src_ty = func.vreg_type(src);
                        let (ret_preg, ret_ty) = if src_ty.is_fp_or_vector() {
                            (cc.fp_ret_reg(), src_ty)
                        } else {
                            (cc.int_ret_reg(), Type::I64)
                        };
                        let ret_vreg = func.new_typed_vreg(ret_ty);
                        reg_bind.insert(ret_vreg, ret_preg);
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
        rets.len() <= 1,
        "multiple-return calls not yet supported"
    );

    // Walk args once, partitioning into int vs FP class and assigning
    // class-relative register indices. Each class gets its own pool
    // and its own position counter, matching SysV. Stack-passed args
    // (int past 6, FP past 8) go into the stack area below.
    #[derive(Copy, Clone)]
    enum ArgSlot {
        IntReg(Reg),
        FpReg(Reg),
        /// Stack arg index, zero-based from the first stack-passed arg.
        IntStack(u32),
    }

    let mut slots: Vec<(Reg, ArgSlot)> = Vec::with_capacity(args.len());
    let mut int_pos: u32 = 0;
    let mut fp_pos: u32 = 0;
    let mut stack_idx_counter: u32 = 0;
    for &user_arg in &args {
        let is_fp = func.vreg_type(user_arg).is_fp_or_vector();
        if is_fp {
            if let Some(preg) = FP_ARG_REGS.get(fp_pos as usize).copied() {
                slots.push((user_arg, ArgSlot::FpReg(preg)));
                fp_pos += 1;
            } else {
                panic!(
                    "more than {} FP call args: stack-passed FP args unimplemented",
                    FP_ARG_REGS.len()
                );
            }
        } else if let Some(preg) = INT_ARG_REGS.get(int_pos as usize).copied() {
            slots.push((user_arg, ArgSlot::IntReg(preg)));
            int_pos += 1;
        } else {
            slots.push((user_arg, ArgSlot::IntStack(stack_idx_counter)));
            stack_idx_counter += 1;
        }
    }

    let stack_arg_count = stack_idx_counter as usize;

    // Reserve a 16-byte-aligned outgoing-args area. Rsp is 16-aligned
    // on entry to this call (the function prologue established that,
    // and no dynamic RSP motion happens between calls); a padded
    // region keeps the CALL at a 16-aligned Rsp.
    let raw_bytes = (stack_arg_count * 8) as i32;
    let reserved = (raw_bytes + 15) & !15; // round up to multiple of 16
    if reserved > 0 {
        new.push(Instruction::Target(X64Inst::AdjustRsp { delta: -reserved }));
    }

    // Emit stack-arg stores (writes to `[rsp + 8*stack_idx]`).
    for (user_arg, slot) in &slots {
        if let ArgSlot::IntStack(stack_idx) = slot {
            new.push(Instruction::Target(X64Inst::StoreStackArg {
                src: *user_arg,
                stack_idx: *stack_idx,
            }));
        }
    }

    // Copy each user-arg vreg into a fresh shim pinned to the
    // appropriate arg preg; the shim lives from Copy to Call64r.
    for (user_arg, slot) in &slots {
        let preg = match *slot {
            ArgSlot::IntReg(p) | ArgSlot::FpReg(p) => p,
            ArgSlot::IntStack(_) => continue,
        };
        let shim = func.new_typed_vreg(func.vreg_type(*user_arg));
        reg_bind.insert(shim, preg);
        new.push(Instruction::Pseudo(PseudoInstruction::Copy {
            dst: shim,
            src: *user_arg,
        }));
    }

    // Clobber every caller-saved preg NOT holding an arg shim.
    // R10/R11/RAX are always caller-saved and never int arg regs.
    // Int arg regs past `int_pos` and all XMMs past `fp_pos` are free.
    for &preg in &[R10, R11] {
        emit_clobber(func, new, reg_bind, preg, Type::I64);
    }
    for &arg_preg in &INT_ARG_REGS[int_pos as usize..] {
        emit_clobber(func, new, reg_bind, arg_preg, Type::I64);
    }
    let all_xmms = [
        XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13,
        XMM14, XMM15,
    ];
    for &preg in &all_xmms[fp_pos as usize..] {
        emit_clobber(func, new, reg_bind, preg, Type::F64);
    }
    emit_clobber(func, new, reg_bind, RAX, Type::I64);

    // Callee address: for direct (symbol) calls we materialize a
    // placeholder `Mov64ri 0` that the loader patches at load time;
    // for indirect calls we already have the address in the user's
    // `fn_ptr` vreg and just thread it into `Call64r` unchanged.
    let addr_vreg = match &call_data.callee {
        CallTarget::Symbol(_) => {
            let v = func.new_vreg();
            new.push(Instruction::Target(X64Inst::Mov64ri {
                dst: v,
                imm: 0, // placeholder, patched at load time
            }));
            v
        }
        CallTarget::Indirect(fn_ptr) => *fn_ptr,
    };

    // Emit the call.
    new.push(Instruction::Target(X64Inst::Call64r { target: addr_vreg }));

    // Reclaim the outgoing-args area before touching RAX / the ret
    // shim so post-call IR sees a canonical RSP.
    if reserved > 0 {
        new.push(Instruction::Target(X64Inst::AdjustRsp { delta: reserved }));
    }

    // Extract the return value: define ret_shim pinned to RAX (int) or
    // XMM0 (FP), copy into the user's return vreg.
    if let Some(&user_ret) = rets.first() {
        let user_ret_ty = func.vreg_type(user_ret);
        let (ret_preg, shim_ty) = if user_ret_ty.is_fp_or_vector() {
            (XMM0, user_ret_ty)
        } else {
            (RAX, user_ret_ty)
        };
        let ret_shim = func.new_typed_vreg(shim_ty);
        reg_bind.insert(ret_shim, ret_preg);
        new.push(Instruction::Pseudo(PseudoInstruction::RegDef {
            vreg: ret_shim,
            preg: ret_preg,
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

/// Emit a caller-saved clobber marker. Caller passes `ty` so the
/// allocator routes the clobber into the right class pool (`I64` for
/// GPR, `F64` for XMM).
fn emit_clobber(
    func: &mut Func<X64Inst>,
    new: &mut Vec<Instruction<X64Inst>>,
    reg_bind: &mut HashMap<Reg, Reg>,
    preg: Reg,
    ty: Type,
) {
    let v = func.new_typed_vreg(ty);
    reg_bind.insert(v, preg);
    new.push(Instruction::Pseudo(PseudoInstruction::ImplicitDef { dst: v }));
    new.push(Instruction::Pseudo(PseudoInstruction::RegDef { vreg: v, preg }));
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
    fn stack_passed_args_lower_to_load_from_stack() {
        // fn(a0..a7) — a0..a5 are reg-passed, a6 and a7 are stack-passed.
        let mut func = Func::<X64Inst>::new("many_args".to_string());
        let b0 = func.add_empty_block();
        let mut dsts = Vec::new();
        for i in 0..8 {
            let v = func.new_vreg();
            dsts.push(v);
            func.get_block_data_mut(b0)
                .push_pseudo_inst(PseudoInstruction::Arg { dst: v, idx: i });
        }
        func.get_block_data_mut(b0)
            .push_pseudo_inst(PseudoInstruction::Return { src: dsts[6] });
        SysVAmd64Lowering.lower(&mut func);

        // The two stack-passed Args must have turned into LoadArgFromStack
        // with stack_idx = 0 and 1.
        let mut seen: Vec<(Reg, u32)> = Vec::new();
        for inst in func.get_block_data(b0).iter() {
            if let Instruction::Target(X64Inst::LoadArgFromStack { dst, stack_idx }) = inst {
                seen.push((*dst, *stack_idx));
            }
        }
        assert_eq!(seen.len(), 2, "two stack-passed args expected");
        assert_eq!(seen[0], (dsts[6], 0));
        assert_eq!(seen[1], (dsts[7], 1));
    }

    #[test]
    fn call_with_stack_args_emits_store_and_rsp_adjusts() {
        use crate::codegen::tir::{CallData, CallTarget};
        let mut func = Func::<X64Inst>::new("caller".to_string());
        let b0 = func.add_empty_block();
        let args: Vec<Reg> = (0..8).map(|_| func.new_vreg()).collect();
        let ret = func.new_vreg();
        let id = func.new_call(CallData {
            callee: CallTarget::Symbol("callee".into()),
            args: args.clone(),
            rets: vec![ret],
        });
        func.get_block_data_mut(b0)
            .push_pseudo_inst(PseudoInstruction::CallPseudo { id });
        func.get_block_data_mut(b0)
            .push_pseudo_inst(PseudoInstruction::Return { src: ret });
        SysVAmd64Lowering.lower(&mut func);

        let insts: Vec<_> = func.get_block_data(b0).iter().copied().collect();
        // Expect: AdjustRsp(-16), then two StoreStackArg, then reg-arg
        // copies, clobbers, Mov64ri, Call64r, AdjustRsp(+16), ret
        // shim/copy, then the original RawRet-pair (emitted by Return
        // lowering).
        let adj_neg = insts.iter().find_map(|i| match i {
            Instruction::Target(X64Inst::AdjustRsp { delta }) if *delta < 0 => Some(*delta),
            _ => None,
        });
        assert_eq!(adj_neg, Some(-16), "reserve 16 bytes for 2 stack args");

        let stack_stores: Vec<_> = insts
            .iter()
            .filter_map(|i| match i {
                Instruction::Target(X64Inst::StoreStackArg { src, stack_idx }) => {
                    Some((*src, *stack_idx))
                }
                _ => None,
            })
            .collect();
        assert_eq!(stack_stores, vec![(args[6], 0), (args[7], 1)]);

        let adj_pos = insts.iter().find_map(|i| match i {
            Instruction::Target(X64Inst::AdjustRsp { delta }) if *delta > 0 => Some(*delta),
            _ => None,
        });
        assert_eq!(adj_pos, Some(16), "reclaim 16 bytes after call");
    }

    #[test]
    fn call_with_exactly_six_args_emits_no_rsp_motion() {
        use crate::codegen::tir::{CallData, CallTarget};
        let mut func = Func::<X64Inst>::new("caller6".to_string());
        let b0 = func.add_empty_block();
        let args: Vec<Reg> = (0..6).map(|_| func.new_vreg()).collect();
        let ret = func.new_vreg();
        let id = func.new_call(CallData {
            callee: CallTarget::Symbol("callee".into()),
            args,
            rets: vec![ret],
        });
        func.get_block_data_mut(b0)
            .push_pseudo_inst(PseudoInstruction::CallPseudo { id });
        func.get_block_data_mut(b0)
            .push_pseudo_inst(PseudoInstruction::Return { src: ret });
        SysVAmd64Lowering.lower(&mut func);

        for inst in func.get_block_data(b0).iter() {
            assert!(
                !matches!(inst, Instruction::Target(X64Inst::AdjustRsp { .. })),
                "no rsp motion expected: {inst:?}"
            );
            assert!(
                !matches!(inst, Instruction::Target(X64Inst::StoreStackArg { .. })),
                "no stack args expected: {inst:?}"
            );
        }
    }

    #[test]
    fn call_with_odd_stack_arg_count_reserves_aligned_pad() {
        use crate::codegen::tir::{CallData, CallTarget};
        let mut func = Func::<X64Inst>::new("caller7".to_string());
        let b0 = func.add_empty_block();
        // 7 args = 1 stack-passed → reserve 16 (8 + 8 pad) to keep
        // rsp 16-aligned at the CALL instruction.
        let args: Vec<Reg> = (0..7).map(|_| func.new_vreg()).collect();
        let ret = func.new_vreg();
        let id = func.new_call(CallData {
            callee: CallTarget::Symbol("callee".into()),
            args,
            rets: vec![ret],
        });
        func.get_block_data_mut(b0)
            .push_pseudo_inst(PseudoInstruction::CallPseudo { id });
        func.get_block_data_mut(b0)
            .push_pseudo_inst(PseudoInstruction::Return { src: ret });
        SysVAmd64Lowering.lower(&mut func);

        let adj_neg = func.get_block_data(b0).iter().find_map(|i| match i {
            Instruction::Target(X64Inst::AdjustRsp { delta }) if *delta < 0 => Some(*delta),
            _ => None,
        });
        assert_eq!(adj_neg, Some(-16));
    }
}
