//! LLVM IR -> lancy IR conversion.

use std::collections::HashMap;

use inkwell::IntPredicate;
use inkwell::basic_block::BasicBlock;
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::values::{
    AsValueRef, BasicValueEnum, CallSiteValue, FunctionValue, InstructionOpcode,
    InstructionValue, PhiValue,
};

use lancy::codegen::isa::x64::builder::FuncBuilder;
use lancy::codegen::isa::x64::inst::{Cond, X64Inst};
use lancy::codegen::isa::x64::pipeline;
use lancy::codegen::jit::Module as JitModule;
use lancy::codegen::tir::{Block, Func, PhiId, Reg};

use crate::error::ConvertError;

/// Pointer-identity key for an LLVM value / basic block handle.
type PtrKey = usize;

fn val_key<V: AsValueRef>(v: &V) -> PtrKey {
    v.as_value_ref() as usize
}

fn bb_key(bb: BasicBlock) -> PtrKey {
    // BasicBlock is a newtype over LLVMBasicBlockRef; `as_mut_ptr` exposes
    // that raw pointer, which is stable for the lifetime of the Context.
    bb.as_mut_ptr() as usize
}

/// Parse `ir` (textual LLVM IR or LLVM bitcode) and lower the function
/// named `name` into a lancy `Func<X64Inst>`.
pub fn convert_function(ir: &[u8], name: &str) -> Result<Func<X64Inst>, ConvertError> {
    let ctx = Context::create();
    // inkwell's memory-buffer APIs require a trailing nul byte; callers
    // typically pass slices from `Vec<u8>` or string literals without
    // one, so we paper over the difference here.
    let owned: Vec<u8> = if ir.last() == Some(&0) {
        ir.to_vec()
    } else {
        let mut v = Vec::with_capacity(ir.len() + 1);
        v.extend_from_slice(ir);
        v.push(0);
        v
    };
    let mem_buf = MemoryBuffer::create_from_memory_range_copy(&owned, "lancy_llvm_input");
    let module = ctx
        .create_module_from_ir(mem_buf)
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    let func = module
        .get_function(name)
        .ok_or_else(|| ConvertError::FunctionNotFound(name.to_string()))?;
    let mut cv = Converter::new(name);
    cv.lower(func)?;
    Ok(cv.finish())
}

/// Parse `ir`, lower `name`, run the x64 pipeline, and load into a JIT
/// module. The caller owns the `JitModule` and can call `entry()` to get
/// a `sysv64` function pointer.
pub fn compile_ir_to_jit(ir: &[u8], name: &str) -> Result<JitModule, ConvertError> {
    let func = convert_function(ir, name)?;
    if std::env::var("LANCY_DEBUG_IR").is_ok() {
        eprintln!("=== lancy IR for `{name}` ===\n{func}");
    }
    let m = pipeline::jit(func)?;
    Ok(m)
}

struct Converter {
    builder: FuncBuilder,
    vals: HashMap<PtrKey, Reg>,
    blocks: HashMap<PtrKey, Block>,
    /// `(lancy_phi_id, llvm_phi_value_ref)` pairs pending incoming-operand
    /// resolution. Back-edge sources may reference vregs defined later in
    /// the function, so we finalize after every block is lowered.
    pending_phis: Vec<(PhiId, llvm_sys::prelude::LLVMValueRef)>,
}

impl Converter {
    fn new(name: &str) -> Self {
        Self {
            builder: FuncBuilder::new(name),
            vals: HashMap::new(),
            blocks: HashMap::new(),
            pending_phis: Vec::new(),
        }
    }

    fn finish(self) -> Func<X64Inst> {
        self.builder.build()
    }

    fn lower(&mut self, func: FunctionValue<'_>) -> Result<(), ConvertError> {
        let bbs = func.get_basic_blocks();
        if bbs.is_empty() {
            return Err(ConvertError::Malformed(
                "function has no basic blocks (declaration only?)".into(),
            ));
        }

        // Map LLVM blocks to lancy blocks. The first LLVM block is the
        // entry and maps to `FuncBuilder`'s pre-created entry block.
        self.blocks.insert(bb_key(bbs[0]), self.builder.entry_block());
        for bb in &bbs[1..] {
            let nb = self.builder.new_block();
            self.blocks.insert(bb_key(*bb), nb);
        }

        // Emit `arg` pseudos in entry-block order for each function param.
        self.builder.switch_to_block(self.builder.entry_block());
        for param in func.get_param_iter() {
            check_int64(param)?;
            let r = self.builder.arg();
            self.vals.insert(val_key(&param), r);
        }

        for bb in &bbs {
            let lancy_bb = self.blocks[&bb_key(*bb)];
            self.builder.switch_to_block(lancy_bb);
            self.lower_block(*bb)?;
        }

        // Phi incoming constants must be materialized in the predecessor
        // block so SSA destruction finds them live at the pred-end Copy.
        let pending = std::mem::take(&mut self.pending_phis);
        for (phi_id, raw_phi) in pending {
            let phi = unsafe { PhiValue::new(raw_phi) };
            let count = phi.count_incoming();
            let mut incoming = Vec::with_capacity(count as usize);
            for i in 0..count {
                let (val, bb) = phi.get_incoming(i).ok_or_else(|| {
                    ConvertError::Malformed(format!("phi missing incoming at {i}"))
                })?;
                let pred_block = *self.blocks.get(&bb_key(bb)).ok_or_else(|| {
                    ConvertError::Malformed("phi references unknown predecessor block".into())
                })?;
                // Re-materialize constants per-edge; a cached vreg from an
                // unrelated block wouldn't be live at this pred-end.
                let src_reg = if is_int_const(val) {
                    self.materialize_phi_const_in_pred(val, pred_block)?
                } else {
                    *self.vals.get(&val_key(&val)).ok_or_else(|| {
                        ConvertError::Malformed(format!(
                            "phi operand {val:?} has no mapped vreg"
                        ))
                    })?
                };
                incoming.push((pred_block, src_reg));
            }
            self.builder.set_phi_incoming(phi_id, incoming);
        }
        Ok(())
    }

    fn materialize_phi_const_in_pred(
        &mut self,
        v: BasicValueEnum<'_>,
        pred_block: Block,
    ) -> Result<Reg, ConvertError> {
        let BasicValueEnum::IntValue(iv) = v else {
            return Err(ConvertError::Unsupported(format!(
                "non-integer unmapped phi operand: {v:?}"
            )));
        };
        if !iv.is_const() {
            return Err(ConvertError::Unsupported(format!(
                "non-constant unmapped phi operand: {v:?}"
            )));
        }
        let imm = iv.get_sign_extended_constant().ok_or_else(|| {
            ConvertError::Unsupported(format!("int constant wider than 64 bits: {v:?}"))
        })?;
        let saved = self.builder.current_block();
        self.builder.switch_to_block(pred_block);
        let r = self.builder.iconst64(imm);
        // `iconst64` appends; rotate the new MOV in front of the
        // terminator that may already be there.
        self.builder.move_last_before_terminator(pred_block);
        self.builder.switch_to_block(saved);
        Ok(r)
    }

    fn lower_block(&mut self, bb: BasicBlock<'_>) -> Result<(), ConvertError> {
        let insts: Vec<InstructionValue> = bb.get_instructions().collect();
        if insts.is_empty() {
            return Err(ConvertError::Malformed(
                "basic block has no terminator".into(),
            ));
        }

        // Detect icmp+br fusion: if the terminator is a conditional br
        // whose condition is the immediately-preceding icmp, translate
        // the pair as a single `branch_icmp` and skip the icmp's own
        // lowering. This is the pattern clang emits at every opt level
        // for `if (a OP b)`-style control flow.
        let fuse_idx = fused_icmp_index(&insts);

        for (i, inst) in insts.iter().enumerate() {
            if Some(i) == fuse_idx {
                continue;
            }
            if let Some(icmp_idx) = fuse_idx
                && inst.get_opcode() == InstructionOpcode::Br
            {
                self.lower_fused_icmp_br(insts[icmp_idx], *inst)?;
                continue;
            }
            self.lower_inst(*inst)?;
        }
        Ok(())
    }

    fn lower_inst(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        use InstructionOpcode as Op;
        match inst.get_opcode() {
            Op::Add => self.lower_binop(inst, BinOp::Add),
            Op::Sub => self.lower_binop(inst, BinOp::Sub),
            Op::Mul => self.lower_binop(inst, BinOp::Mul),
            Op::SDiv => self.lower_binop(inst, BinOp::SDiv),
            Op::UDiv => self.lower_binop(inst, BinOp::UDiv),
            Op::SRem => self.lower_binop(inst, BinOp::SRem),
            Op::URem => self.lower_binop(inst, BinOp::URem),
            Op::And => self.lower_binop(inst, BinOp::And),
            Op::Or => self.lower_binop(inst, BinOp::Or),
            Op::Xor => self.lower_binop(inst, BinOp::Xor),
            Op::Shl => self.lower_shift(inst, ShiftKind::Shl),
            Op::LShr => self.lower_shift(inst, ShiftKind::LShr),
            Op::AShr => self.lower_shift(inst, ShiftKind::AShr),
            Op::ICmp => self.lower_icmp_value(inst),
            Op::Select => self.lower_select(inst),
            Op::Phi => self.lower_phi(inst),
            Op::Return => self.lower_ret(inst),
            Op::Br => self.lower_br(inst),
            Op::SExt => self.lower_sext(inst),
            Op::ZExt => self.lower_zext(inst),
            Op::Trunc => self.lower_trunc(inst),
            Op::BitCast | Op::PtrToInt | Op::IntToPtr => self.lower_passthrough(inst),
            Op::Call => self.lower_call(inst),
            Op::Alloca => self.lower_alloca(inst),
            Op::Load => self.lower_load(inst),
            Op::Store => self.lower_store(inst),
            Op::GetElementPtr => self.lower_gep(inst),
            Op::Switch => self.lower_switch(inst),
            Op::IndirectBr => self.lower_indirectbr(inst),
            Op::Unreachable => {
                self.builder.unreachable();
                Ok(())
            }
            Op::Freeze => self.lower_passthrough(inst),
            Op::Fence => {
                self.builder.mfence();
                Ok(())
            }
            Op::ExtractValue
            | Op::InsertValue
            | Op::AtomicRMW
            | Op::AtomicCmpXchg
            | Op::ExtractElement
            | Op::InsertElement
            | Op::ShuffleVector
            | Op::FAdd
            | Op::FSub
            | Op::FMul
            | Op::FDiv
            | Op::FRem
            | Op::FNeg
            | Op::FCmp
            | Op::FPToSI
            | Op::FPToUI
            | Op::SIToFP
            | Op::UIToFP
            | Op::FPExt
            | Op::FPTrunc => Err(ConvertError::Unsupported(format!(
                "opcode {:?} not modeled yet",
                inst.get_opcode()
            ))),
            other => Err(ConvertError::Unsupported(format!(
                "instruction opcode {other:?}"
            ))),
        }
    }

    fn lower_binop(
        &mut self,
        inst: InstructionValue<'_>,
        op: BinOp,
    ) -> Result<(), ConvertError> {
        let lhs = self.operand_reg(inst, 0)?;
        let rhs = self.operand_reg(inst, 1)?;
        let dst = match op {
            BinOp::Add => self.builder.add(lhs, rhs),
            BinOp::Sub => self.builder.sub(lhs, rhs),
            BinOp::Mul => self.builder.imul(lhs, rhs),
            BinOp::SDiv => self.builder.sdiv(lhs, rhs),
            BinOp::UDiv => self.builder.udiv(lhs, rhs),
            BinOp::SRem => self.builder.srem(lhs, rhs),
            BinOp::URem => self.builder.urem(lhs, rhs),
            BinOp::And => self.builder.and(lhs, rhs),
            BinOp::Or => self.builder.or(lhs, rhs),
            BinOp::Xor => self.builder.xor(lhs, rhs),
        };
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    /// Shifts: emit the immediate-count form when the count is a constant
    /// that fits in `[0, 63]`; otherwise pin the count to CL.
    fn lower_shift(
        &mut self,
        inst: InstructionValue<'_>,
        kind: ShiftKind,
    ) -> Result<(), ConvertError> {
        let val = self.operand_reg(inst, 0)?;
        let count_operand = Self::operand_value(inst, 1)?;
        let imm_count = if let BasicValueEnum::IntValue(iv) = count_operand
            && iv.is_const()
        {
            iv.get_sign_extended_constant()
                .filter(|c| (0..64).contains(c))
                .and_then(|c| u8::try_from(c).ok())
        } else {
            None
        };

        let dst = if let Some(imm) = imm_count {
            match kind {
                ShiftKind::Shl => self.builder.shl_imm(val, imm),
                ShiftKind::LShr => self.builder.shr_imm(val, imm),
                ShiftKind::AShr => self.builder.sar_imm(val, imm),
            }
        } else {
            let count = self.resolve_value(count_operand)?;
            match kind {
                ShiftKind::Shl => self.builder.shl(val, count),
                ShiftKind::LShr => self.builder.shr(val, count),
                ShiftKind::AShr => self.builder.sar(val, count),
            }
        };
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    /// A free-standing `icmp` (not fused with a conditional br). Emit
    /// `cmp + setcc + movzx` so the result is a proper `{0, 1}` i64.
    fn lower_icmp_value(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let pred = inst
            .get_icmp_predicate()
            .ok_or_else(|| ConvertError::Malformed("icmp without predicate".into()))?;
        let cond = cond_from_predicate(pred);
        let a = self.operand_reg(inst, 0)?;
        let b = self.operand_reg(inst, 1)?;
        let dst = self.builder.icmp_to_i64(cond, a, b);
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    /// `select i1 %cond, i64 %t, i64 %f`. Fuses into `cmp + cmov` when
    /// `%cond` is a just-emitted icmp; otherwise tests the materialized
    /// i64 condition against zero.
    fn lower_select(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let cond_operand = Self::operand_value(inst, 0)?;
        let true_val = self.operand_reg(inst, 1)?;
        let false_val = self.operand_reg(inst, 2)?;

        if let BasicValueEnum::IntValue(iv) = cond_operand
            && let Some(cond_inst) = iv.as_instruction()
            && cond_inst.get_opcode() == InstructionOpcode::ICmp
        {
            let pred = cond_inst.get_icmp_predicate().ok_or_else(|| {
                ConvertError::Malformed("icmp without predicate".into())
            })?;
            let cond = cond_from_predicate(pred);
            let a = self.operand_reg(cond_inst, 0)?;
            let b = self.operand_reg(cond_inst, 1)?;
            let dst = self.builder.select(cond, a, b, true_val, false_val);
            self.vals.insert(val_key(&inst), dst);
            return Ok(());
        }

        let cond_reg = self.resolve_value(cond_operand)?;
        let zero = self.builder.iconst64(0);
        let dst =
            self.builder
                .select(crate::Cond::NZ, cond_reg, zero, true_val, false_val);
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    /// Lower an LLVM `call`. Integer intrinsics we recognize inline;
    /// everything else becomes a direct `call_sym` that the ABI pass
    /// expands into arg-register copies plus an indirect call.
    fn lower_call(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        // InstructionValue exposes no `get_called_fn_value`; wrap the raw
        // value ref as a `CallSiteValue` and ask that instead.
        let cs = unsafe { CallSiteValue::new(inst.as_value_ref()) };
        let callee = cs.get_called_fn_value().ok_or_else(|| {
            ConvertError::Unsupported(
                "indirect call (function pointer not supported yet)".into(),
            )
        })?;
        let name_cow = callee.get_name().to_string_lossy();
        let name: &str = &name_cow;

        if name.starts_with("llvm.lifetime.")
            || name.starts_with("llvm.invariant.")
            || name == "llvm.donothing"
            || name == "llvm.assume"
        {
            return Ok(());
        }

        // Integer min/max/abs intrinsics — identical shape, CMP + CMOV.
        if let Some(cond) = minmax_cond(name) {
            let a = self.operand_reg(inst, 0)?;
            let b = self.operand_reg(inst, 1)?;
            let dst = self.builder.select(cond, a, b, a, b);
            self.vals.insert(val_key(&inst), dst);
            return Ok(());
        }
        if name == "llvm.abs.i64" {
            let x = self.operand_reg(inst, 0)?;
            let neg = self.builder.neg(x);
            let zero = self.builder.iconst64(0);
            let dst = self.builder.select(crate::Cond::GE, x, zero, x, neg);
            self.vals.insert(val_key(&inst), dst);
            return Ok(());
        }

        // Memory intrinsics redirect to libc. The is-volatile operand
        // is ignored — our emitter doesn't do volatile optimizations.
        if let Some(sym) = libc_mem_sym(name) {
            let a = self.operand_reg(inst, 0)?;
            let b = self.operand_reg(inst, 1)?;
            let c = self.operand_reg(inst, 2)?;
            self.builder.call_sym(sym, &[a, b, c]);
            return Ok(());
        }

        let n_args = callee.count_params();
        let mut args: Vec<_> = Vec::with_capacity(n_args as usize);
        for i in 0..n_args {
            args.push(self.operand_reg(inst, i)?);
        }
        let ret = self.builder.call_sym(name, &args);
        self.vals.insert(val_key(&inst), ret);
        Ok(())
    }

    fn lower_ret(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        if inst.get_num_operands() == 0 {
            return Err(ConvertError::Unsupported(
                "void return — lancy requires an i64 return value".into(),
            ));
        }
        let r = self.operand_reg(inst, 0)?;
        self.builder.ret(r);
        Ok(())
    }

    fn lower_br(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        match inst.get_num_operands() {
            1 => {
                let target = Self::operand_block(inst, 0)?;
                let blk = self.lookup_block(target)?;
                self.builder.jmp(blk);
                Ok(())
            }
            3 => Err(ConvertError::Unsupported(
                "conditional br not preceded by icmp (likely a non-fused boolean path)"
                    .into(),
            )),
            n => Err(ConvertError::Malformed(format!(
                "br with {n} operands"
            ))),
        }
    }

    fn lower_passthrough(
        &mut self,
        inst: InstructionValue<'_>,
    ) -> Result<(), ConvertError> {
        // lancy only carries i64 right now — integer bitcasts become
        // identity copies at the value level. The emitted bytes of
        // upstream ops already leave the register in the low-64 form
        // lancy uses.
        let src = self.operand_reg(inst, 0)?;
        self.vals.insert(val_key(&inst), src);
        Ok(())
    }

    fn int_bit_width(v: BasicValueEnum<'_>) -> Result<u32, ConvertError> {
        match v {
            BasicValueEnum::IntValue(iv) => Ok(iv.get_type().get_bit_width()),
            other => Err(ConvertError::Unsupported(format!(
                "expected integer value, got {other:?}"
            ))),
        }
    }

    fn lower_sext(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let src_val = Self::operand_value(inst, 0)?;
        let src_bw = Self::int_bit_width(src_val)?;
        let src = self.resolve_value(src_val)?;
        let dst = match src_bw {
            1 => {
                // sext i1 → i64: `x == 0 ? 0 : -1`.
                let zero = self.builder.iconst64(0);
                let minus_one = self.builder.iconst64(-1);
                self.builder
                    .select(crate::Cond::Z, src, zero, zero, minus_one)
            }
            8 => self.builder.sext_i8_to_i64(src),
            16 => self.builder.sext_i16_to_i64(src),
            32 => self.builder.sext_i32_to_i64(src),
            64 => src,
            other => {
                return Err(ConvertError::Unsupported(format!(
                    "sext from i{other} (only i1/i8/i16/i32 to i64 supported)"
                )));
            }
        };
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    fn lower_zext(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let src_val = Self::operand_value(inst, 0)?;
        let src_bw = Self::int_bit_width(src_val)?;
        let src = self.resolve_value(src_val)?;
        // i1 values are already 0/1 (from setcc+movzx), so zext i1 is a no-op.
        let dst = match src_bw {
            1 | 64 => src,
            8 => self.builder.zext_i8_to_i64(src),
            16 => self.builder.zext_i16_to_i64(src),
            32 => self.builder.zext_i32_to_i64(src),
            other => {
                return Err(ConvertError::Unsupported(format!(
                    "zext from i{other} (only i1/i8/i16/i32 to i64 supported)"
                )));
            }
        };
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    /// Lower `switch i64 %cond, label %default [i64 V0, label %L0, …]`
    /// as a chain of `cmp/je`: each case `i` branches to its target or
    /// falls through to the next comparison; the last miss targets
    /// `default` directly.
    fn lower_switch(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let cond = self.operand_reg(inst, 0)?;
        let default_blk = self.lookup_block(Self::operand_block(inst, 1)?)?;

        // Cases are interleaved `(value, label)` pairs after the first
        // two operands. Collect them so we know which iteration is last.
        let n = inst.get_num_operands();
        let mut cases: Vec<(BasicValueEnum<'_>, Block)> = Vec::new();
        let mut idx = 2;
        while idx + 1 < n {
            let val_op = Self::operand_value(inst, idx)?;
            let label_bb = Self::operand_block(inst, idx + 1)?;
            cases.push((val_op, self.lookup_block(label_bb)?));
            idx += 2;
        }

        if cases.is_empty() {
            self.builder.jmp(default_blk);
            return Ok(());
        }

        let last_idx = cases.len() - 1;
        for (i, (val_op, case_target)) in cases.into_iter().enumerate() {
            let case_val_reg = self.resolve_value(val_op)?;
            // The last miss falls through directly to the default; every
            // earlier miss falls through to a fresh scratch block that
            // hosts the next comparison.
            let miss_target = if i == last_idx {
                default_blk
            } else {
                let next_tail = self.builder.new_block();
                self.builder
                    .branch_icmp(Cond::Z, cond, case_val_reg, case_target, next_tail);
                self.builder.switch_to_block(next_tail);
                continue;
            };
            self.builder
                .branch_icmp(Cond::Z, cond, case_val_reg, case_target, miss_target);
        }
        Ok(())
    }

    /// Lower `indirectbr ptr %tgt, [...]`. The set of possible target
    /// labels is advisory; we just jump to the address in the register.
    fn lower_indirectbr(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let tgt = self.operand_reg(inst, 0)?;
        self.builder.jmp_indirect(tgt);
        Ok(())
    }

    fn lower_alloca(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let elem_ty = inst
            .get_allocated_type()
            .map_err(|e| ConvertError::Malformed(format!("alloca: {e}")))?;
        let (elem_size, elem_align) = type_size_align(elem_ty)?;
        let n_val = inst
            .get_operand(0)
            .and_then(inkwell::values::Operand::value);
        let count: u64 = match n_val {
            Some(BasicValueEnum::IntValue(iv)) if iv.is_const() => iv
                .get_zero_extended_constant()
                .ok_or_else(|| ConvertError::Unsupported("alloca count not fitting in u64".into()))?,
            _ => 1,
        };
        let total = elem_size.saturating_mul(count);
        if total > u64::from(u32::MAX) {
            return Err(ConvertError::Unsupported(
                "alloca size larger than u32".into(),
            ));
        }
        let dst = self.builder.stack_alloc(total as u32, elem_align);
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    fn lower_load(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let ptr = self.operand_reg(inst, 0)?;
        let ty = match inst.get_type() {
            inkwell::types::AnyTypeEnum::IntType(t) => Some(t.get_bit_width()),
            _ => None,
        };
        let dst = match ty {
            Some(64) => self.builder.load_i64(ptr, 0),
            Some(32) => self.builder.load_i32(ptr, 0),
            Some(16) => self.builder.load_i16(ptr, 0),
            Some(8) => self.builder.load_i8(ptr, 0),
            Some(1) => self.builder.load_i8(ptr, 0),
            Some(w) => {
                return Err(ConvertError::Unsupported(format!(
                    "load of i{w} — only i1/i8/i16/i32/i64 supported"
                )));
            }
            None => {
                // Pointer load. Treat as i64 on x86-64.
                self.builder.load_i64(ptr, 0)
            }
        };
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    fn lower_store(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        // LLVM store operand order: (value, pointer).
        let val = self.operand_reg(inst, 0)?;
        let ptr = self.operand_reg(inst, 1)?;
        let val_op = Self::operand_value(inst, 0)?;
        let ty_bits = match val_op {
            BasicValueEnum::IntValue(iv) => Some(iv.get_type().get_bit_width()),
            _ => None,
        };
        match ty_bits {
            Some(64) => self.builder.store_i64(ptr, 0, val),
            Some(32) => self.builder.store_i32(ptr, 0, val),
            Some(16) => self.builder.store_i16(ptr, 0, val),
            Some(1 | 8) => self.builder.store_i8(ptr, 0, val),
            Some(w) => {
                return Err(ConvertError::Unsupported(format!(
                    "store of i{w} — only i1/i8/i16/i32/i64 supported"
                )));
            }
            None => {
                // Pointer store: 8-byte.
                self.builder.store_i64(ptr, 0, val);
            }
        }
        Ok(())
    }

    fn lower_gep(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        use inkwell::types::AnyType;
        // GEP walks the type tree: for arrays, each index advances by
        // element stride; for structs, the index selects a field offset.
        let base = self.operand_reg(inst, 0)?;
        let n = inst.get_num_operands();
        let source_ty = inst
            .get_gep_source_element_type()
            .map_err(|e| ConvertError::Malformed(format!("GEP: {e}")))?;
        let mut cur_ty: inkwell::types::AnyTypeEnum<'_> = source_ty.as_any_type_enum();
        // Accumulated constant displacement, and optional (index,
        // stride) for the final variable index — we fuse the last
        // variable index into `lea` when possible.
        let mut disp: i64 = 0;
        let mut var_index: Option<(Reg, u64)> = None;
        for idx in 1..n {
            let op = inst.get_operand(idx).and_then(inkwell::values::Operand::value);
            match cur_ty {
                inkwell::types::AnyTypeEnum::ArrayType(at) => {
                    let elem_basic = at.get_element_type();
                    let elem_any = elem_basic.as_any_type_enum();
                    let (elem_size, _) = any_type_size_align(elem_any)?;
                    (disp, var_index, cur_ty) =
                        step_gep_linear(self, op, elem_size, disp, var_index, elem_any)?;
                }
                inkwell::types::AnyTypeEnum::StructType(st) => {
                    // Struct index must be a constant i32/i64.
                    let Some(BasicValueEnum::IntValue(iv)) = op else {
                        return Err(ConvertError::Unsupported(
                            "GEP into struct with non-constant index".into(),
                        ));
                    };
                    let field_idx = iv
                        .get_zero_extended_constant()
                        .ok_or_else(|| ConvertError::Unsupported("struct GEP index overflow".into()))?
                        as u32;
                    let (off, field_ty) = struct_field_offset(st, field_idx)?;
                    disp = disp.saturating_add(off as i64);
                    cur_ty = field_ty;
                }
                other => {
                    // First index walks elements of the source type —
                    // after that, PointerType shouldn't recur. But
                    // LLVM typed pointers have changed in recent
                    // versions; handle it like an array for the first
                    // step.
                    let (elem_size, _) = any_type_size_align(other)?;
                    (disp, var_index, cur_ty) = step_gep_linear(
                        self, op, elem_size, disp, var_index, other,
                    )?;
                }
            }
        }
        // Fit disp into i32 for lea. Larger accumulations become an
        // add-before-lea.
        let disp32: i32 = i32::try_from(disp).map_err(|_| {
            ConvertError::Unsupported(format!("GEP displacement {disp} doesn't fit in i32"))
        })?;
        let dst = if let Some((idx, stride)) = var_index {
            // We can fold strides of 1, 2, 4, or 8 directly into LEA.
            let scale = match stride {
                1 => Some(1),
                2 => Some(2),
                4 => Some(4),
                8 => Some(8),
                _ => None,
            };
            if let Some(s) = scale {
                self.builder.gep_indexed(base, idx, s, disp32)
            } else {
                // stride = non-1248: emit `tmp = idx * stride; base +
                // tmp` as a two-step computation. Using imul + lea.
                let stride_reg = self.builder.iconst64(stride as i64);
                let scaled = self.builder.imul(idx, stride_reg);
                let sum = self.builder.add(base, scaled);
                if disp32 != 0 {
                    self.builder.gep_const(sum, disp32)
                } else {
                    sum
                }
            }
        } else {
            self.builder.gep_const(base, disp32)
        };
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    // Defers incoming-operand resolution so back-edge sources are
    // guaranteed to be in `self.vals` by the time we need them.
    #[allow(clippy::unnecessary_wraps)]
    fn lower_phi(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let (dst, id) = self.builder.phi_with_id(Vec::new());
        self.vals.insert(val_key(&inst), dst);
        self.pending_phis.push((id, inst.as_value_ref()));
        Ok(())
    }

    fn lower_trunc(&mut self, inst: InstructionValue<'_>) -> Result<(), ConvertError> {
        let src = self.operand_reg(inst, 0)?;
        // Destination width comes from the instruction's result type.
        let dst_bw = match inst.get_type() {
            inkwell::types::AnyTypeEnum::IntType(t) => t.get_bit_width(),
            other => {
                return Err(ConvertError::Unsupported(format!(
                    "trunc to non-integer type: {other:?}"
                )));
            }
        };
        let dst = match dst_bw {
            1 => self.builder.trunc_to_i1(src),
            8 => self.builder.trunc_to_i8(src),
            16 => self.builder.trunc_to_i16(src),
            32 => self.builder.trunc_to_i32(src),
            64 => src,
            other => {
                return Err(ConvertError::Unsupported(format!(
                    "trunc to i{other} (only i1/i8/i16/i32/i64 supported)"
                )));
            }
        };
        self.vals.insert(val_key(&inst), dst);
        Ok(())
    }

    fn lower_fused_icmp_br(
        &mut self,
        icmp: InstructionValue<'_>,
        br: InstructionValue<'_>,
    ) -> Result<(), ConvertError> {
        let pred = icmp
            .get_icmp_predicate()
            .ok_or_else(|| ConvertError::Malformed("icmp without predicate".into()))?;
        let cond = cond_from_predicate(pred);
        let a = self.operand_reg(icmp, 0)?;
        let b = self.operand_reg(icmp, 1)?;
        // LLVM C API conditional br operand order is (cond, false, true).
        let false_bb = Self::operand_block(br, 1)?;
        let true_bb = Self::operand_block(br, 2)?;
        let taken = self.lookup_block(true_bb)?;
        let not_taken = self.lookup_block(false_bb)?;
        self.builder.branch_icmp(cond, a, b, taken, not_taken);
        Ok(())
    }

    /// Operand `idx` of `inst` as a `BasicValueEnum`. Errors if the
    /// operand slot is absent or isn't a value (e.g., a label).
    fn operand_value(
        inst: InstructionValue<'_>,
        idx: u32,
    ) -> Result<BasicValueEnum<'_>, ConvertError> {
        inst.get_operand(idx)
            .and_then(inkwell::values::Operand::value)
            .ok_or_else(|| {
                ConvertError::Malformed(format!(
                    "instruction {:?} missing value operand {idx}",
                    inst.get_opcode()
                ))
            })
    }

    fn operand_block(
        inst: InstructionValue<'_>,
        idx: u32,
    ) -> Result<BasicBlock<'_>, ConvertError> {
        inst.get_operand(idx)
            .and_then(inkwell::values::Operand::block)
            .ok_or_else(|| {
                ConvertError::Malformed(format!(
                    "instruction {:?} missing label operand {idx}",
                    inst.get_opcode()
                ))
            })
    }

    fn lookup_block(&self, bb: BasicBlock<'_>) -> Result<Block, ConvertError> {
        self.blocks
            .get(&bb_key(bb))
            .copied()
            .ok_or_else(|| ConvertError::Malformed("reference to unknown basic block".into()))
    }

    /// Resolve operand `idx` of `inst` to a lancy vreg. If the operand is
    /// an as-yet-unmaterialized integer constant, materialize it with
    /// `iconst64` now and cache the result.
    fn operand_reg(
        &mut self,
        inst: InstructionValue<'_>,
        idx: u32,
    ) -> Result<Reg, ConvertError> {
        let operand = Self::operand_value(inst, idx)?;
        self.resolve_value(operand)
    }

    fn resolve_value(&mut self, v: BasicValueEnum<'_>) -> Result<Reg, ConvertError> {
        let key = val_key(&v);
        if let Some(&r) = self.vals.get(&key) {
            return Ok(r);
        }
        // Null and global-address pointer constants become placeholder 0.
        // TODO: real global relocation for pointer constants that are
        // actually dereferenced at runtime.
        if let BasicValueEnum::PointerValue(pv) = v
            && (pv.is_null() || pv.is_const())
        {
            let r = self.builder.iconst64(0);
            self.vals.insert(key, r);
            return Ok(r);
        }
        let BasicValueEnum::IntValue(iv) = v else {
            return Err(ConvertError::Unsupported(format!(
                "non-integer value operand: {v:?}"
            )));
        };
        if !iv.is_const() {
            return Err(ConvertError::Unsupported(format!(
                "non-constant unmapped integer value: {v:?}"
            )));
        }
        // Constants wider than i64 (e.g. the i65 temporaries clang emits at
        // -O1 for overflow-free multiply) don't surface via the safe API;
        // fall back to LLVMConstIntGetSExtValue for the low 64 bits.
        let imm: i64 = if let Some(v) = iv.get_sign_extended_constant() {
            v
        } else {
            // SAFETY: `iv` is a valid constant int value owned by the LLVM
            // context; the call returns the low 64 bits signed.
            unsafe { llvm_sys::core::LLVMConstIntGetSExtValue(iv.as_value_ref()) }
        };
        let r = self.builder.iconst64(imm);
        self.vals.insert(key, r);
        Ok(r)
    }
}

#[derive(Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy)]
enum ShiftKind {
    Shl,
    LShr,
    AShr,
}

fn is_int_const(v: BasicValueEnum<'_>) -> bool {
    matches!(v, BasicValueEnum::IntValue(iv) if iv.is_const())
}

/// Map an i64 min/max intrinsic name to the CMP condition that selects
/// the first operand. `None` means the name isn't one of these.
fn minmax_cond(name: &str) -> Option<Cond> {
    match name {
        "llvm.smax.i64" => Some(Cond::G),
        "llvm.smin.i64" => Some(Cond::L),
        "llvm.umax.i64" => Some(Cond::A),
        "llvm.umin.i64" => Some(Cond::B),
        _ => None,
    }
}

/// Map an `llvm.mem{cpy,move,set}.*` intrinsic name to its libc symbol.
fn libc_mem_sym(name: &str) -> Option<&'static str> {
    if name.starts_with("llvm.memcpy.") {
        Some("memcpy")
    } else if name.starts_with("llvm.memmove.") {
        Some("memmove")
    } else if name.starts_with("llvm.memset.") {
        Some("memset")
    } else {
        None
    }
}

fn type_size_align(ty: inkwell::types::BasicTypeEnum<'_>) -> Result<(u64, u32), ConvertError> {
    use inkwell::types::AnyType;
    any_type_size_align(ty.as_any_type_enum())
}

fn any_type_size_align(
    ty: inkwell::types::AnyTypeEnum<'_>,
) -> Result<(u64, u32), ConvertError> {
    use inkwell::types::AnyType;
    use inkwell::types::AnyTypeEnum as T;
    match ty {
        T::IntType(it) => {
            let bits = it.get_bit_width();
            let bytes = bits.div_ceil(8).max(1);
            let aligned = bytes.next_power_of_two();
            Ok((u64::from(aligned), aligned.min(8)))
        }
        T::PointerType(_) => Ok((8, 8)),
        T::FloatType(ft) => {
            let bits = match ft.print_to_string().to_string().as_str() {
                "float" => 32,
                "double" => 64,
                _ => {
                    return Err(ConvertError::Unsupported(format!(
                        "unsupported float type: {ft:?}"
                    )));
                }
            };
            let bytes = bits / 8;
            Ok((u64::from(bytes), bytes))
        }
        T::ArrayType(at) => {
            let (esz, ea) = any_type_size_align(at.get_element_type().as_any_type_enum())?;
            let len = u64::from(at.len());
            Ok((esz.saturating_mul(len), ea))
        }
        T::StructType(st) => {
            let mut size: u64 = 0;
            let mut max_align: u32 = 1;
            for i in 0..st.count_fields() {
                let f = st.get_field_type_at_index(i).ok_or_else(|| {
                    ConvertError::Malformed(format!("struct missing field at {i}"))
                })?;
                let (fs, fa) = type_size_align(f)?;
                size = size.next_multiple_of(u64::from(fa));
                size += fs;
                if fa > max_align {
                    max_align = fa;
                }
            }
            size = size.next_multiple_of(u64::from(max_align));
            Ok((size, max_align))
        }
        T::VectorType(_) | T::ScalableVectorType(_) => Err(ConvertError::Unsupported(
            "vector types in alloca/memory not yet supported".into(),
        )),
        T::VoidType(_) | T::FunctionType(_) => Err(ConvertError::Malformed(format!(
            "cannot size non-value type {ty:?}"
        ))),
    }
}

fn struct_field_offset(
    st: inkwell::types::StructType<'_>,
    field: u32,
) -> Result<(u64, inkwell::types::AnyTypeEnum<'_>), ConvertError> {
    use inkwell::types::AnyType;
    let mut offset: u64 = 0;
    for i in 0..st.count_fields() {
        let ty = st
            .get_field_type_at_index(i)
            .ok_or_else(|| ConvertError::Malformed(format!("struct missing field at {i}")))?;
        let (fs, fa) = type_size_align(ty)?;
        offset = offset.next_multiple_of(u64::from(fa));
        if i == field {
            return Ok((offset, ty.as_any_type_enum()));
        }
        offset += fs;
    }
    Err(ConvertError::Malformed(format!(
        "struct field index {field} out of range"
    )))
}

/// `(constant_displacement, optional (var_index_reg, stride), next_type)`
/// — the state threaded through GEP walking.
type GepStep<'a> = (i64, Option<(Reg, u64)>, inkwell::types::AnyTypeEnum<'a>);

/// Walk one step of a GEP through a "linear" type (array or, at the
/// top, the source element type): either add a constant offset (when
/// the index is constant) or remember a single variable index that
/// we'll fuse into the final `lea`.
fn step_gep_linear<'a>(
    cv: &mut Converter,
    op: Option<BasicValueEnum<'_>>,
    elem_size: u64,
    mut disp: i64,
    mut var_index: Option<(Reg, u64)>,
    next_ty: inkwell::types::AnyTypeEnum<'a>,
) -> Result<GepStep<'a>, ConvertError> {
    let op = op.ok_or_else(|| ConvertError::Malformed("GEP missing index".into()))?;
    if let BasicValueEnum::IntValue(iv) = op
        && iv.is_const()
    {
        let c = iv
            .get_sign_extended_constant()
            .ok_or_else(|| ConvertError::Unsupported("GEP index wider than i64".into()))?;
        disp = disp.saturating_add(
            i128::from(c)
                .saturating_mul(i128::from(elem_size))
                .try_into()
                .unwrap_or(i64::MAX),
        );
    } else if var_index.is_some() {
        return Err(ConvertError::Unsupported(
            "GEP with multiple variable indices not supported yet".into(),
        ));
    } else {
        let idx_reg = cv.resolve_value(op)?;
        var_index = Some((idx_reg, elem_size));
    }
    Ok((disp, var_index, next_ty))
}

fn cond_from_predicate(p: IntPredicate) -> Cond {
    match p {
        IntPredicate::EQ => Cond::Z,
        IntPredicate::NE => Cond::NZ,
        IntPredicate::SLT => Cond::L,
        IntPredicate::SLE => Cond::LE,
        IntPredicate::SGT => Cond::G,
        IntPredicate::SGE => Cond::GE,
        IntPredicate::ULT => Cond::B,
        IntPredicate::ULE => Cond::BE,
        IntPredicate::UGT => Cond::A,
        IntPredicate::UGE => Cond::AE,
    }
}

/// Return the index of an icmp that should be fused with the block's
/// conditional-br terminator. We fuse only when the icmp immediately
/// precedes the br and the br consumes its result — the common pattern
/// clang emits. Any other layout (intervening instructions, icmp with
/// multiple uses, icmp in a different block) is translated as-is, which
/// currently fails because the builder has no way to materialize an i1.
fn fused_icmp_index(insts: &[InstructionValue<'_>]) -> Option<usize> {
    if insts.len() < 2 {
        return None;
    }
    let n = insts.len();
    let br = insts[n - 1];
    if br.get_opcode() != InstructionOpcode::Br || br.get_num_operands() != 3 {
        return None;
    }
    let cond_val = br.get_operand(0).and_then(inkwell::values::Operand::value)?;
    let cond_key = val_key(&cond_val);
    let prev = insts[n - 2];
    if prev.get_opcode() != InstructionOpcode::ICmp {
        return None;
    }
    if val_key(&prev) != cond_key {
        return None;
    }
    Some(n - 2)
}

fn check_int64(v: BasicValueEnum<'_>) -> Result<(), ConvertError> {
    match v {
        BasicValueEnum::IntValue(iv) => {
            let bw = iv.get_type().get_bit_width();
            if bw == 64 {
                Ok(())
            } else {
                Err(ConvertError::Unsupported(format!(
                    "function argument is i{bw}; only i64 is supported"
                )))
            }
        }
        other => Err(ConvertError::Unsupported(format!(
            "non-integer function argument: {other:?}"
        ))),
    }
}
