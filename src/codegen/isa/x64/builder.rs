//! IR builder (v0 frontend) for the x64 target. Produces `Func<X64Inst>` in
//! SSA shape using `PseudoInstruction` for ABI-neutral concerns.
//!
//! Each builder method emits into the current block the minimal set of
//! instructions to implement the described operation. Arithmetic ops emit a
//! `Copy` before the two-operand target instruction so the frontend sees a
//! three-operand illusion.

use crate::codegen::isa::x64::inst::{Cond, Mem, X64Inst};
use crate::codegen::isa::x64::regs::{RAX, RCX, RDX};
use crate::codegen::tir::{
    AggregateId, Block, CallData, CallTarget, Func, Inst, PhiId, PseudoInstruction, Reg, Type,
};

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

    pub fn new_vreg(&mut self) -> Reg {
        self.func.new_vreg()
    }

    /// Allocate a fresh vreg of the given type. Use for FP / vector /
    /// non-I64 values so the allocator and emitter can route correctly.
    pub fn new_typed_vreg(&mut self, ty: Type) -> Reg {
        self.func.new_typed_vreg(ty)
    }

    pub fn copy_into(&mut self, dst: Reg, src: Reg) {
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::Copy { dst, src });
    }

    /// Move the last instruction of `block` to just before its terminator.
    /// Used when a frontend appends into a block that already has a
    /// terminator; on x64 splicing MOV-class instructions between a
    /// flag-setting op and its consumer is safe because MOV preserves flags.
    pub fn move_last_before_terminator(&mut self, block: Block) {
        let insts = self.func.get_block_data_mut(block).insts_mut();
        if insts.len() < 2 {
            return;
        }
        let Some(pos) = insts.iter().rposition(Inst::is_term) else {
            return;
        };
        let last = insts.pop().expect("len >= 2");
        insts.insert(pos, last);
    }

    /// Define the next incoming argument. Must be called on the entry block.
    /// Returns a fresh vreg that holds the argument value.
    pub fn arg(&mut self) -> Reg {
        self.arg_typed(Type::I64)
    }

    /// Typed variant of `arg` — stamps the given `Type` on the returned
    /// vreg so ABI lowering can route it into the FP argument-register
    /// pool when `ty.is_fp_or_vector()`.
    pub fn arg_typed(&mut self, ty: Type) -> Reg {
        assert_eq!(
            self.current, self.entry,
            "arg() must be called while positioned on the entry block"
        );
        let dst = self.func.new_typed_vreg(ty);
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

    /// Emit a phi at the current position. `incoming` is `(pred, src)`
    /// pairs — one per predecessor edge. SSA destruction rewrites this
    /// to parallel copies in predecessors before regalloc.
    pub fn phi(&mut self, incoming: Vec<(Block, Reg)>) -> Reg {
        self.phi_with_id(incoming).0
    }

    /// Variant of `phi` that also returns the phi's `PhiId` so callers
    /// can populate back-edge incoming pairs via `set_phi_incoming`
    /// once the predecessor's vreg exists.
    pub fn phi_with_id(&mut self, incoming: Vec<(Block, Reg)>) -> (Reg, PhiId) {
        let dst = self.func.new_vreg();
        let id = self.func.new_phi(incoming);
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::Phi { dst, id });
        (dst, id)
    }

    pub fn set_phi_incoming(&mut self, id: PhiId, incoming: Vec<(Block, Reg)>) {
        self.func.phi_operands_mut(id).incoming = incoming;
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

    pub fn and(&mut self, a: Reg, b: Reg) -> Reg {
        self.binop_rr(a, b, |dst, src| X64Inst::And64rr { dst, src })
    }

    pub fn or(&mut self, a: Reg, b: Reg) -> Reg {
        self.binop_rr(a, b, |dst, src| X64Inst::Or64rr { dst, src })
    }

    pub fn xor(&mut self, a: Reg, b: Reg) -> Reg {
        self.binop_rr(a, b, |dst, src| X64Inst::Xor64rr { dst, src })
    }

    fn unary(&mut self, a: Reg, make_inst: impl FnOnce(Reg) -> X64Inst) -> Reg {
        let dst = self.func.new_vreg();
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst, src: a });
        bd.push_target_inst(make_inst(dst));
        dst
    }

    pub fn not(&mut self, a: Reg) -> Reg {
        self.unary(a, |dst| X64Inst::Not64r { dst })
    }

    pub fn neg(&mut self, a: Reg) -> Reg {
        self.unary(a, |dst| X64Inst::Neg64r { dst })
    }

    fn shift_ri(&mut self, a: Reg, imm: u8, make_inst: impl FnOnce(Reg, u8) -> X64Inst) -> Reg {
        let dst = self.func.new_vreg();
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst, src: a });
        bd.push_target_inst(make_inst(dst, imm));
        dst
    }

    pub fn shl_imm(&mut self, a: Reg, imm: u8) -> Reg {
        self.shift_ri(a, imm, |dst, imm| X64Inst::Shl64ri8 { dst, imm })
    }

    pub fn shr_imm(&mut self, a: Reg, imm: u8) -> Reg {
        self.shift_ri(a, imm, |dst, imm| X64Inst::Shr64ri8 { dst, imm })
    }

    pub fn sar_imm(&mut self, a: Reg, imm: u8) -> Reg {
        self.shift_ri(a, imm, |dst, imm| X64Inst::Sar64ri8 { dst, imm })
    }

    fn shift_rr(&mut self, a: Reg, count: Reg, make_inst: impl FnOnce(Reg, Reg) -> X64Inst) -> Reg {
        // x86 requires the shift count in CL; pin a shim vreg to RCX.
        let count_shim = self.func.new_vreg();
        self.func.pre_bind(count_shim, RCX);
        let dst = self.func.new_vreg();
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst: count_shim, src: count });
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst, src: a });
        bd.push_target_inst(make_inst(dst, count_shim));
        dst
    }

    pub fn shl(&mut self, a: Reg, count: Reg) -> Reg {
        self.shift_rr(a, count, |dst, count| X64Inst::Shl64rcl { dst, count })
    }

    pub fn shr(&mut self, a: Reg, count: Reg) -> Reg {
        self.shift_rr(a, count, |dst, count| X64Inst::Shr64rcl { dst, count })
    }

    pub fn sar(&mut self, a: Reg, count: Reg) -> Reg {
        self.shift_rr(a, count, |dst, count| X64Inst::Sar64rcl { dst, count })
    }

    /// Signed 64-bit divide/remainder. Returns `(quotient_vreg, remainder_vreg)`,
    /// pinned to RAX and RDX respectively. The dividend is sign-extended
    /// into RDX via `sar 63`.
    fn idiv_impl(&mut self, a: Reg, b: Reg) -> (Reg, Reg) {
        let hi_in = self.func.new_vreg();
        self.func.pre_bind(hi_in, RDX);
        let lo_in = self.func.new_vreg();
        self.func.pre_bind(lo_in, RAX);
        let quotient = self.func.new_vreg();
        self.func.pre_bind(quotient, RAX);
        let remainder = self.func.new_vreg();
        self.func.pre_bind(remainder, RDX);

        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst: lo_in, src: a });
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst: hi_in, src: a });
        bd.push_target_inst(X64Inst::Sar64ri8 { dst: hi_in, imm: 63 });
        bd.push_target_inst(X64Inst::Idiv64r {
            divisor: b,
            hi_in,
            lo_in,
            quotient,
            remainder,
        });
        (quotient, remainder)
    }

    /// Unsigned 64-bit divide/remainder. Returns `(quotient_vreg, remainder_vreg)`,
    /// pinned to RAX and RDX. The dividend is zero-extended into RDX via a
    /// 64-bit MOV immediate 0.
    fn div_impl(&mut self, a: Reg, b: Reg) -> (Reg, Reg) {
        let hi_in = self.func.new_vreg();
        self.func.pre_bind(hi_in, RDX);
        let lo_in = self.func.new_vreg();
        self.func.pre_bind(lo_in, RAX);
        let quotient = self.func.new_vreg();
        self.func.pre_bind(quotient, RAX);
        let remainder = self.func.new_vreg();
        self.func.pre_bind(remainder, RDX);

        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst: lo_in, src: a });
        bd.push_target_inst(X64Inst::Mov64ri { dst: hi_in, imm: 0 });
        bd.push_target_inst(X64Inst::Div64r {
            divisor: b,
            hi_in,
            lo_in,
            quotient,
            remainder,
        });
        (quotient, remainder)
    }

    /// Kill `discard`, then materialize a fresh vreg holding a copy of `keep`.
    /// Used to extract one of IDIV/DIV's two pinned outputs.
    fn take_div_result(&mut self, keep: Reg, discard: Reg) -> Reg {
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::Kill { src: discard });
        let out = self.func.new_vreg();
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::Copy { dst: out, src: keep });
        out
    }

    pub fn sdiv(&mut self, a: Reg, b: Reg) -> Reg {
        let (q, r) = self.idiv_impl(a, b);
        self.take_div_result(q, r)
    }

    pub fn srem(&mut self, a: Reg, b: Reg) -> Reg {
        let (q, r) = self.idiv_impl(a, b);
        self.take_div_result(r, q)
    }

    pub fn udiv(&mut self, a: Reg, b: Reg) -> Reg {
        let (q, r) = self.div_impl(a, b);
        self.take_div_result(q, r)
    }

    pub fn urem(&mut self, a: Reg, b: Reg) -> Reg {
        let (q, r) = self.div_impl(a, b);
        self.take_div_result(r, q)
    }

    /// `dst = if (a OP b) { true_val } else { false_val }` via CMP + CMOV.
    /// The starting value is `false_val` (copied), then CMOV on `cond`
    /// replaces it with `true_val` if the flag is set.
    pub fn select(
        &mut self,
        cond: Cond,
        cmp_lhs: Reg,
        cmp_rhs: Reg,
        true_val: Reg,
        false_val: Reg,
    ) -> Reg {
        let dst = self.func.new_vreg();
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst, src: false_val });
        bd.push_target_inst(X64Inst::Cmp64rr { lhs: cmp_lhs, rhs: cmp_rhs });
        bd.push_target_inst(X64Inst::Cmov64rr { cond, dst, src: true_val });
        dst
    }

    /// Emit a single target instruction shaped `{ dst, src }` into a fresh
    /// vreg. Shared tail for the zero-cost rr-form extends (Mov32rr, Movsx,
    /// Movzx) that define `dst` purely from `src`.
    fn emit_rr_def<F>(&mut self, src: Reg, make_inst: F) -> Reg
    where
        F: FnOnce(Reg, Reg) -> X64Inst,
    {
        let dst = self.func.new_vreg();
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(make_inst(dst, src));
        dst
    }

    pub fn sext_i32_to_i64(&mut self, src: Reg) -> Reg {
        self.emit_rr_def(src, |dst, src| X64Inst::Movsxd64r32 { dst, src })
    }

    pub fn sext_i16_to_i64(&mut self, src: Reg) -> Reg {
        self.emit_rr_def(src, |dst, src| X64Inst::Movsx64r16 { dst, src })
    }

    pub fn sext_i8_to_i64(&mut self, src: Reg) -> Reg {
        self.emit_rr_def(src, |dst, src| X64Inst::Movsx64r8 { dst, src })
    }

    pub fn zext_i32_to_i64(&mut self, src: Reg) -> Reg {
        self.emit_rr_def(src, |dst, src| X64Inst::Mov32rr { dst, src })
    }

    pub fn zext_i16_to_i64(&mut self, src: Reg) -> Reg {
        self.emit_rr_def(src, |dst, src| X64Inst::Movzx64r16 { dst, src })
    }

    pub fn zext_i8_to_i64(&mut self, src: Reg) -> Reg {
        self.emit_rr_def(src, |dst, src| X64Inst::Movzx64r8 { dst, src })
    }

    pub fn trunc_to_i32(&mut self, src: Reg) -> Reg {
        self.emit_rr_def(src, |dst, src| X64Inst::Mov32rr { dst, src })
    }

    /// Truncate by masking the low `mask` bits into a fresh vreg.
    fn trunc_mask(&mut self, src: Reg, mask: i32) -> Reg {
        let dst = self.func.new_vreg();
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst, src });
        bd.push_target_inst(X64Inst::And64ri32 { dst, imm: mask });
        dst
    }

    pub fn trunc_to_i16(&mut self, src: Reg) -> Reg {
        self.trunc_mask(src, 0xFFFF)
    }

    pub fn trunc_to_i8(&mut self, src: Reg) -> Reg {
        self.trunc_mask(src, 0xFF)
    }

    pub fn trunc_to_i1(&mut self, src: Reg) -> Reg {
        self.trunc_mask(src, 0x1)
    }

    /// Materialize an `icmp` result as an i64 in `{0, 1}` via CMP + SETCC +
    /// MOVZX. Used when the comparison feeds something other than a branch
    /// (stored, returned, etc).
    pub fn icmp_to_i64(&mut self, cond: Cond, a: Reg, b: Reg) -> Reg {
        let byte = self.func.new_vreg();
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_target_inst(X64Inst::Cmp64rr { lhs: a, rhs: b });
        bd.push_target_inst(X64Inst::Setcc8r { cond, dst: byte });
        let dst = self.func.new_vreg();
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Movzx64r8 { dst, src: byte });
        dst
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

    /// Unconditional indirect jump through a register. Used for LLVM's
    /// `indirectbr` — the register holds the target address.
    pub fn jmp_indirect(&mut self, target: Reg) {
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Jmp64r { target });
    }

    /// Emit `ud2` — traps the process. Used for LLVM's `unreachable`
    /// so reaching this point yields a defined SIGILL rather than
    /// silently falling through to garbage.
    pub fn unreachable(&mut self) {
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Ud2);
    }

    /// Emit `mfence` — a full memory fence. Used for LLVM's `fence`.
    pub fn mfence(&mut self) {
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Mfence);
    }

    pub fn ret(&mut self, src: Reg) {
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::Return { src });
    }

    /// Allocate `size` bytes of stack with `align`-byte alignment.
    /// Returns a vreg holding the base pointer of the allocation.
    /// The allocation lives for the rest of the function.
    pub fn stack_alloc(&mut self, size: u32, align: u32) -> Reg {
        assert!(align.is_power_of_two(), "alloca align must be a power of two");
        assert!(align >= 1, "alloca align must be at least 1");
        let dst = self.func.new_vreg();
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::StackAlloc { dst, size, align });
        dst
    }

    fn emit_load(
        &mut self,
        base: Reg,
        disp: i32,
        make: impl FnOnce(Reg, Mem) -> X64Inst,
    ) -> Reg {
        let dst = self.func.new_vreg();
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(make(dst, Mem::base_disp(base, disp)));
        dst
    }

    fn emit_store(
        &mut self,
        base: Reg,
        disp: i32,
        val: Reg,
        make: impl FnOnce(Mem, Reg) -> X64Inst,
    ) {
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(make(Mem::base_disp(base, disp), val));
    }

    pub fn load_i64(&mut self, base: Reg, disp: i32) -> Reg {
        self.emit_load(base, disp, |dst, src| X64Inst::Mov64rm { dst, src })
    }

    pub fn load_i32(&mut self, base: Reg, disp: i32) -> Reg {
        self.emit_load(base, disp, |dst, src| X64Inst::Mov32rm { dst, src })
    }

    pub fn load_i16(&mut self, base: Reg, disp: i32) -> Reg {
        self.emit_load(base, disp, |dst, src| X64Inst::Mov16rm { dst, src })
    }

    pub fn load_i8(&mut self, base: Reg, disp: i32) -> Reg {
        self.emit_load(base, disp, |dst, src| X64Inst::Mov8rm { dst, src })
    }

    pub fn store_i64(&mut self, base: Reg, disp: i32, val: Reg) {
        self.emit_store(base, disp, val, |dst, src| X64Inst::Mov64mr { dst, src });
    }

    pub fn store_i32(&mut self, base: Reg, disp: i32, val: Reg) {
        self.emit_store(base, disp, val, |dst, src| X64Inst::Mov32mr { dst, src });
    }

    pub fn store_i16(&mut self, base: Reg, disp: i32, val: Reg) {
        self.emit_store(base, disp, val, |dst, src| X64Inst::Mov16mr { dst, src });
    }

    pub fn store_i8(&mut self, base: Reg, disp: i32, val: Reg) {
        self.emit_store(base, disp, val, |dst, src| X64Inst::Mov8mr { dst, src });
    }

    /// Compute `base + disp` into a fresh vreg — integer-constant GEP.
    /// For `disp == 0` this is just `base`; otherwise a `lea` absorbs
    /// the arithmetic in one instruction.
    pub fn gep_const(&mut self, base: Reg, disp: i32) -> Reg {
        if disp == 0 {
            return base;
        }
        let dst = self.func.new_vreg();
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Lea64rm {
                dst,
                src: Mem::base_disp(base, disp),
            });
        dst
    }

    /// Compute `base + index*scale + disp`. `scale` must be 1, 2, 4, or
    /// 8 (the hardware-supported scales for `lea`).
    pub fn gep_indexed(&mut self, base: Reg, index: Reg, scale: u8, disp: i32) -> Reg {
        assert!(
            matches!(scale, 1 | 2 | 4 | 8),
            "lea supports scale ∈ {{1,2,4,8}}, got {scale}"
        );
        let dst = self.func.new_vreg();
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Lea64rm {
                dst,
                src: Mem {
                    base,
                    index: Some(index),
                    scale,
                    disp,
                },
            });
        dst
    }

    /// Emit a direct call to a named symbol. `args` are the user vregs
    /// holding argument values. Returns the user vreg that will hold
    /// the 64-bit return value after the call.
    pub fn call_sym(&mut self, symbol: &str, args: &[Reg]) -> Reg {
        let user_ret = self.func.new_vreg();
        let id = self.func.new_call(CallData {
            callee: CallTarget::Symbol(symbol.to_string()),
            args: args.to_vec(),
            rets: vec![user_ret],
        });
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::CallPseudo { id });
        user_ret
    }

    /// Emit an indirect call through a register holding a function
    /// pointer. Returns an integer-typed return vreg.
    pub fn call_indirect(&mut self, fn_ptr: Reg, args: &[Reg]) -> Reg {
        self.call_indirect_typed(fn_ptr, args, Type::I64)
    }

    /// Variant of `call_indirect` that stamps the given `Type` on the
    /// return vreg. Use for calls whose return belongs to a different
    /// class (FP via `Type::F64` / `Type::F32`) so ABI lowering pins
    /// the return shim to XMM0 rather than RAX.
    pub fn call_indirect_typed(&mut self, fn_ptr: Reg, args: &[Reg], ret_ty: Type) -> Reg {
        let user_ret = self.func.new_typed_vreg(ret_ty);
        let id = self.func.new_call(CallData {
            callee: CallTarget::Indirect(fn_ptr),
            args: args.to_vec(),
            rets: vec![user_ret],
        });
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::CallPseudo { id });
        user_ret
    }

    // ---- Floating-point helpers. ----

    fn fp_binop<F>(&mut self, ty: Type, a: Reg, b: Reg, make_inst: F) -> Reg
    where
        F: FnOnce(Reg, Reg) -> X64Inst,
    {
        let dst = self.func.new_typed_vreg(ty);
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst, src: a });
        bd.push_target_inst(make_inst(dst, b));
        dst
    }

    pub fn fadd_f32(&mut self, a: Reg, b: Reg) -> Reg {
        self.fp_binop(Type::F32, a, b, |dst, src| X64Inst::Addssrr { dst, src })
    }
    pub fn fsub_f32(&mut self, a: Reg, b: Reg) -> Reg {
        self.fp_binop(Type::F32, a, b, |dst, src| X64Inst::Subssrr { dst, src })
    }
    pub fn fmul_f32(&mut self, a: Reg, b: Reg) -> Reg {
        self.fp_binop(Type::F32, a, b, |dst, src| X64Inst::Mulssrr { dst, src })
    }
    pub fn fdiv_f32(&mut self, a: Reg, b: Reg) -> Reg {
        self.fp_binop(Type::F32, a, b, |dst, src| X64Inst::Divssrr { dst, src })
    }

    pub fn fadd_f64(&mut self, a: Reg, b: Reg) -> Reg {
        self.fp_binop(Type::F64, a, b, |dst, src| X64Inst::Addsdrr { dst, src })
    }
    pub fn fsub_f64(&mut self, a: Reg, b: Reg) -> Reg {
        self.fp_binop(Type::F64, a, b, |dst, src| X64Inst::Subsdrr { dst, src })
    }
    pub fn fmul_f64(&mut self, a: Reg, b: Reg) -> Reg {
        self.fp_binop(Type::F64, a, b, |dst, src| X64Inst::Mulsdrr { dst, src })
    }
    pub fn fdiv_f64(&mut self, a: Reg, b: Reg) -> Reg {
        self.fp_binop(Type::F64, a, b, |dst, src| X64Inst::Divsdrr { dst, src })
    }

    /// `f32` load from `[base + disp]`.
    pub fn load_f32(&mut self, base: Reg, disp: i32) -> Reg {
        let dst = self.func.new_typed_vreg(Type::F32);
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Movssrm {
                dst,
                src: Mem::base_disp(base, disp),
            });
        dst
    }

    /// `f64` load from `[base + disp]`.
    pub fn load_f64(&mut self, base: Reg, disp: i32) -> Reg {
        let dst = self.func.new_typed_vreg(Type::F64);
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Movsdrm {
                dst,
                src: Mem::base_disp(base, disp),
            });
        dst
    }

    pub fn store_f32(&mut self, base: Reg, disp: i32, val: Reg) {
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Movssmr {
                dst: Mem::base_disp(base, disp),
                src: val,
            });
    }

    pub fn store_f64(&mut self, base: Reg, disp: i32, val: Reg) {
        self.func
            .get_block_data_mut(self.current)
            .push_target_inst(X64Inst::Movsdmr {
                dst: Mem::base_disp(base, disp),
                src: val,
            });
    }

    // ---- Atomic helpers. ----

    /// Atomic fetch-and-add on a 64-bit memory location. Returns the
    /// pre-add memory value. xadd's register operand is both used and
    /// redefined, so we shim `delta` into a fresh vreg.
    pub fn atomic_fetch_add_i64(&mut self, base: Reg, disp: i32, delta: Reg) -> Reg {
        let shim = self.func.new_typed_vreg(Type::I64);
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst: shim, src: delta });
        bd.push_target_inst(X64Inst::LockXadd64mr {
            dst: Mem::base_disp(base, disp),
            src: shim,
        });
        shim
    }

    /// Atomic compare-and-swap on a 64-bit memory location. Returns
    /// `(witness, success)`: `witness` is the pre-op memory value,
    /// `success` is `1` on swap, `0` on mismatch.
    pub fn atomic_cmpxchg_i64(
        &mut self,
        base: Reg,
        disp: i32,
        expected: Reg,
        desired: Reg,
    ) -> (Reg, Reg) {
        let rax_in = self.func.new_typed_vreg(Type::I64);
        let rax_out = self.func.new_typed_vreg(Type::I64);
        self.func.pre_bind(rax_in, RAX);
        self.func.pre_bind(rax_out, RAX);
        let success_byte = self.func.new_typed_vreg(Type::I8);
        let success = self.func.new_typed_vreg(Type::I64);
        let bd = self.func.get_block_data_mut(self.current);
        bd.push_pseudo_inst(PseudoInstruction::Copy { dst: rax_in, src: expected });
        bd.push_target_inst(X64Inst::LockCmpxchg64mr {
            dst: Mem::base_disp(base, disp),
            src: desired,
            rax_in,
            rax_out,
        });
        // SETZ must run immediately after CMPXCHG — any intervening
        // flag-writing op would clobber ZF.
        bd.push_target_inst(X64Inst::Setcc8r {
            cond: Cond::Z,
            dst: success_byte,
        });
        bd.push_target_inst(X64Inst::Movzx64r8 {
            dst: success,
            src: success_byte,
        });
        // Unpin the witness so downstream users don't inherit RAX.
        let out = self.func.new_typed_vreg(Type::I64);
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::Copy { dst: out, src: rax_out });
        (out, success)
    }

    // ---- Aggregate helpers. ----

    /// Register an aggregate with initial `elems` and return its
    /// handle vreg (erased by `lower_aggregates` before regalloc).
    pub fn make_aggregate(&mut self, elems: Vec<Reg>) -> Reg {
        let id: AggregateId = self.func.new_aggregate(elems);
        let dst = self.func.new_typed_vreg(Type::Agg(id));
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::MakeAggregate { dst, id });
        dst
    }

    /// Extract element `idx` of aggregate `agg` as a fresh `elem_ty`
    /// vreg.
    pub fn extract_value(&mut self, agg: Reg, idx: u32, elem_ty: Type) -> Reg {
        let dst = self.func.new_typed_vreg(elem_ty);
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::ExtractValue { dst, agg, idx });
        dst
    }

    /// Produce a new aggregate matching `agg` except element `idx` is
    /// replaced by `val`.
    pub fn insert_value(&mut self, agg: Reg, val: Reg, idx: u32) -> Reg {
        let ty = self.func.vreg_type(agg);
        let dst = self.func.new_typed_vreg(ty);
        self.func
            .get_block_data_mut(self.current)
            .push_pseudo_inst(PseudoInstruction::InsertValue {
                dst,
                agg,
                val,
                idx,
            });
        dst
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
