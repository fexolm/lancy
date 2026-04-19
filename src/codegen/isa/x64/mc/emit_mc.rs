//! Machine-code emission for the x86-64 target.
//!
//! **Requires:** Regalloc complete. Every vreg that appears in an emitted
//! operand has an `Assignment` in the result. Pseudos in the input stream
//! are tolerated: `Arg` is erased, `Copy` is either coalesced (erased) or
//! lowered to a target MOV. `Return` must have been lowered to `RawRet`
//! already.
//!
//! **Effect:** Emits a flat `Vec<u8>` of x86-64 machine code via iced-x86.
//! Inserts the prologue (`push rbp; mov rbp, rsp; sub rsp, N`) and
//! epilogue (`add rsp, N; pop rbp; ret`) around the user body. Injects
//! spill-store moves at each `SplitMove` point so an evicted value lands
//! in its stack slot before the new owner takes the preg.
//!
//! **Spill handling:** when an operand is stack-allocated at the point of
//! use, we load into / store out of a scratch register around the
//! instruction. Scratch registers must be disjoint from the allocatable
//! pool — the frontend's `RegAllocConfig` is responsible for that.

use std::collections::HashMap;

use crate::codegen::analysis::layout::{BlockLayout, ProgramPoint};
use crate::codegen::isa::x64::inst::{Cond, X64Inst};
use crate::codegen::isa::x64::regs::{
    R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP,
};
use crate::codegen::isa::x64::sysv::CALLEE_SAVED;
use crate::codegen::regalloc::{
    AllocatedSlot, RegAllocConfig, RegAllocResult, SplitMove, StackSlot,
};
use crate::codegen::tir::{Func, Instruction, PseudoInstruction, Reg};
use crate::support::slotmap::Key;
use iced_x86::code_asm::registers::{
    cl, r10, r10b, r10d, r10w, r11, r11b, r11d, r11w, r12, r12b, r12d, r12w, r13, r13b, r13d,
    r13w, r14, r14b, r14d, r14w, r15, r15b, r15d, r15w, r8, r8b, r8d, r8w, r9, r9b, r9d, r9w,
    rax, rbp, rbx, rcx, rdi, rdx, rsi, rsp, al, ax, bl, bp, bpl, bx, cx, di, dil, dl, dx, eax,
    ebp, ebx, ecx, edi, edx, esi, esp, si, sil, sp, spl,
};
use iced_x86::code_asm::{
    AsmRegister16, AsmRegister32, AsmRegister64, AsmRegister8, CodeAssembler, CodeLabel,
};
use std::collections::BTreeSet;

/// Maximum simultaneous scratch registers this instruction can demand in the
/// worst case (all operand vregs spilled). Stays in sync with `emit_inst`.
fn scratch_demand_of(inst: &X64Inst) -> usize {
    match inst {
        // Memory ops with index take 3 scratches in the worst case (base,
        // index, and either dst-load or src).
        X64Inst::Mov64rm { src, .. }
        | X64Inst::Mov32rm { src, .. }
        | X64Inst::Mov16rm { src, .. }
        | X64Inst::Mov8rm { src, .. }
        | X64Inst::Lea64rm { src, .. }
            if src.index.is_some() =>
        {
            3
        }
        X64Inst::Mov64mr { dst: m, .. }
        | X64Inst::Mov32mr { dst: m, .. }
        | X64Inst::Mov16mr { dst: m, .. }
        | X64Inst::Mov8mr { dst: m, .. }
            if m.index.is_some() =>
        {
            3
        }
        X64Inst::Mov64rm { .. }
        | X64Inst::Mov32rm { .. }
        | X64Inst::Mov16rm { .. }
        | X64Inst::Mov8rm { .. }
        | X64Inst::Lea64rm { .. }
        | X64Inst::Mov64mr { .. }
        | X64Inst::Mov32mr { .. }
        | X64Inst::Mov16mr { .. }
        | X64Inst::Mov8mr { .. } => 2,
        X64Inst::Add64rr { .. }
        | X64Inst::Sub64rr { .. }
        | X64Inst::Imul64rr { .. }
        | X64Inst::And64rr { .. }
        | X64Inst::Or64rr { .. }
        | X64Inst::Xor64rr { .. }
        | X64Inst::Cmp64rr { .. }
        | X64Inst::Test64rr { .. }
        | X64Inst::Cmov64rr { .. }
        | X64Inst::Mov64rr { .. }
        | X64Inst::Mov32rr { .. }
        | X64Inst::Mov16rr { .. }
        | X64Inst::Mov8rr { .. }
        | X64Inst::Movsx64r8 { .. }
        | X64Inst::Movsx64r16 { .. }
        | X64Inst::Movsxd64r32 { .. }
        | X64Inst::Movzx64r8 { .. }
        | X64Inst::Movzx64r16 { .. }
        | X64Inst::Shl64rcl { .. }
        | X64Inst::Shr64rcl { .. }
        | X64Inst::Sar64rcl { .. } => 2,
        X64Inst::Mov64ri { .. }
        | X64Inst::Mov32ri { .. }
        | X64Inst::Mov16ri { .. }
        | X64Inst::Mov8ri { .. }
        | X64Inst::Add64ri32 { .. }
        | X64Inst::Sub64ri32 { .. }
        | X64Inst::And64ri32 { .. }
        | X64Inst::Or64ri32 { .. }
        | X64Inst::Xor64ri32 { .. }
        | X64Inst::Cmp64ri32 { .. }
        | X64Inst::Test64ri32 { .. }
        | X64Inst::Not64r { .. }
        | X64Inst::Neg64r { .. }
        | X64Inst::Shl64ri8 { .. }
        | X64Inst::Shr64ri8 { .. }
        | X64Inst::Sar64ri8 { .. }
        | X64Inst::Setcc8r { .. }
        | X64Inst::Call64r { .. }
        | X64Inst::Jmp64r { .. } => 1,
        // Div/IDiv: divisor is the only vreg that isn't pre-bound. The
        // rest are pinned to RAX/RDX and physically live there at the
        // call site.
        X64Inst::Idiv64r { .. } | X64Inst::Div64r { .. } => 1,
        X64Inst::Jmp { .. }
        | X64Inst::CondJmp { .. }
        | X64Inst::RawRet
        | X64Inst::Ud2
        | X64Inst::Mfence => 0,
    }
}

fn to_ice_reg(r: Reg) -> AsmRegister64 {
    match r {
        RAX => rax,
        RBX => rbx,
        RCX => rcx,
        RDX => rdx,
        RSI => rsi,
        RDI => rdi,
        RSP => rsp,
        RBP => rbp,
        R8 => r8,
        R9 => r9,
        R10 => r10,
        R11 => r11,
        R12 => r12,
        R13 => r13,
        R14 => r14,
        R15 => r15,
        other => panic!("to_ice_reg: unsupported reg index {other}"),
    }
}

fn to_ice_reg32(r: Reg) -> AsmRegister32 {
    match r {
        RAX => eax,
        RBX => ebx,
        RCX => ecx,
        RDX => edx,
        RSI => esi,
        RDI => edi,
        RSP => esp,
        RBP => ebp,
        R8 => r8d,
        R9 => r9d,
        R10 => r10d,
        R11 => r11d,
        R12 => r12d,
        R13 => r13d,
        R14 => r14d,
        R15 => r15d,
        other => panic!("to_ice_reg32: unsupported reg index {other}"),
    }
}

fn to_ice_reg16(r: Reg) -> AsmRegister16 {
    match r {
        RAX => ax,
        RBX => bx,
        RCX => cx,
        RDX => dx,
        RSI => si,
        RDI => di,
        RSP => sp,
        RBP => bp,
        R8 => r8w,
        R9 => r9w,
        R10 => r10w,
        R11 => r11w,
        R12 => r12w,
        R13 => r13w,
        R14 => r14w,
        R15 => r15w,
        other => panic!("to_ice_reg16: unsupported reg index {other}"),
    }
}

fn to_ice_reg8(r: Reg) -> AsmRegister8 {
    // REX-prefixed 8-bit encoding: the low byte of any of the 16 GPRs.
    match r {
        RAX => al,
        RBX => bl,
        RCX => cl,
        RDX => dl,
        RSI => sil,
        RDI => dil,
        RSP => spl,
        RBP => bpl,
        R8 => r8b,
        R9 => r9b,
        R10 => r10b,
        R11 => r11b,
        R12 => r12b,
        R13 => r13b,
        R14 => r14b,
        R15 => r15b,
        other => panic!("to_ice_reg8: unsupported reg index {other}"),
    }
}

pub struct FnMCWriter<'i> {
    asm: CodeAssembler,
    func: &'i Func<X64Inst>,
    ra_cfg: &'i RegAllocConfig,
    ra_res: &'i RegAllocResult,
    layout: BlockLayout,
    frame_adjust: u32,
    saved_callee_regs: Vec<Reg>,
    splits_by_point: HashMap<ProgramPoint, Vec<SplitMove>>,
    /// `addr_vreg -> iced_inst_index`, populated as the emitter
    /// renders each call-site `Mov64ri`. After assembly we use
    /// `CodeAssemblerResult::new_instruction_offsets` to look up each
    /// instruction's final byte offset.
    call_target_insts: HashMap<Reg, usize>,
    /// For each `PseudoInstruction::StackAlloc { dst, .. }`, the
    /// `rbp`-relative displacement at which the allocated region
    /// begins. Emitting the pseudo materializes `lea dst, [rbp+disp]`.
    alloca_offsets: HashMap<Reg, i32>,
}

/// One symbol-patch request: byte offset in the emitted buffer where
/// an 8-byte placeholder immediate lives, plus the symbol to resolve.
/// The symbol string is empty for indirect calls (no patching needed).
#[derive(Clone, Debug)]
pub struct EmittedCallReloc {
    pub imm_offset: usize,
    pub symbol: String,
}

/// Output of `emit_fn`: the raw code bytes plus every call-site
/// relocation that needs to be patched before the bytes are executed.
pub struct EmittedFunc {
    pub bytes: Vec<u8>,
    pub relocations: Vec<EmittedCallReloc>,
}

impl<'i> FnMCWriter<'i> {
    #[must_use]
    pub fn new(
        func: &'i Func<X64Inst>,
        ra_cfg: &'i RegAllocConfig,
        ra_res: &'i RegAllocResult,
    ) -> Self {
        let (alloca_extra, alloca_offsets) =
            Self::compute_alloca_layout(func, ra_res.frame_size);
        let raw_frame = ra_res.frame_size + alloca_extra;
        let frame_adjust = raw_frame.div_ceil(16) * 16;
        let saved_callee_regs = Self::compute_saved_callee_regs(ra_cfg, ra_res);
        let layout = BlockLayout::compute(func);
        let mut splits_by_point: HashMap<ProgramPoint, Vec<SplitMove>> = HashMap::new();
        for sm in &ra_res.split_moves {
            splits_by_point.entry(sm.at_point).or_default().push(*sm);
        }
        Self {
            asm: CodeAssembler::new(64).expect("iced-x86 supports 64-bit"),
            func,
            ra_cfg,
            ra_res,
            layout,
            frame_adjust,
            saved_callee_regs,
            splits_by_point,
            call_target_insts: HashMap::new(),
            alloca_offsets,
        }
    }

    /// Scan the func for `StackAlloc` pseudos, pack each below the
    /// spill region, and return `(extra_frame_bytes, vreg → rbp-disp)`.
    /// Each alloca's displacement is negative — `rbp + disp` is the
    /// lowest address of its allocated region, so the pointer is
    /// properly aligned to the requested `align`.
    fn compute_alloca_layout(
        func: &'i Func<X64Inst>,
        ra_frame_size: u32,
    ) -> (u32, HashMap<Reg, i32>) {
        let mut running: u32 = ra_frame_size;
        let mut offsets: HashMap<Reg, i32> = HashMap::new();
        for (_b, bd) in func.blocks_iter() {
            for inst in bd.iter() {
                if let Instruction::Pseudo(PseudoInstruction::StackAlloc {
                    dst,
                    size,
                    align,
                }) = inst
                {
                    // Grow running offset by `size`, then round up to
                    // `align` so the allocation's *low* address is
                    // aligned.
                    let new_running = (running + *size).next_multiple_of(*align);
                    offsets.insert(*dst, -(new_running as i32));
                    running = new_running;
                }
            }
        }
        (running - ra_frame_size, offsets)
    }

    /// Mark vreg `v` as the address operand of an upcoming call-site
    /// `Mov64ri`. The next time we emit a Mov64ri whose destination is
    /// `v` we'll record its iced instruction index so we can look up
    /// its byte offset after assembly.
    pub fn mark_call_target(&mut self, v: Reg) {
        // Use `usize::MAX` as "pending" sentinel; replaced with the
        // real iced index inside `emit_inst`.
        self.call_target_insts.insert(v, usize::MAX);
    }

    fn compute_saved_callee_regs(ra_cfg: &RegAllocConfig, ra_res: &RegAllocResult) -> Vec<Reg> {
        let mut used: BTreeSet<Reg> = BTreeSet::new();
        for (_v, asn) in ra_res.assignments.iter() {
            for slot in asn.slots() {
                if let AllocatedSlot::Reg(r) = slot {
                    used.insert(r);
                }
            }
        }
        for s in &ra_cfg.scratch_regs {
            used.insert(*s);
        }
        CALLEE_SAVED
            .iter()
            .filter(|r| used.contains(r))
            .filter(|&&r| r != RBP)
            .copied()
            .collect()
    }

    fn slot_offset(slot: StackSlot) -> i32 {
        -((slot as i32 + 1) * 8)
    }

    fn check_scratch_budget(&self) {
        let max_needed = self.max_scratch_demand();
        assert!(
            self.ra_cfg.scratch_regs.len() >= max_needed,
            "MC emitter needs {max_needed} scratch reg(s) for this function's worst-case \
             instruction, but RegAllocConfig only provides {}. Add more scratches to \
             the config (disjoint from `allocatable_regs`).",
            self.ra_cfg.scratch_regs.len()
        );
    }

    fn max_scratch_demand(&self) -> usize {
        let mut demand = 0usize;
        for (_b, bd) in self.func.blocks_iter() {
            for inst in bd.iter() {
                if let Instruction::Target(t) = inst {
                    let d = scratch_demand_of(t);
                    if d > demand {
                        demand = d;
                    }
                }
            }
        }
        demand
    }

    fn scratch(&self, idx: usize) -> AsmRegister64 {
        assert!(
            idx < self.ra_cfg.scratch_regs.len(),
            "MC emitter asked for scratch #{idx} but RegAllocConfig only provides {} scratch \
             registers. Memory addressing modes with both base and index spilled require up \
             to 3 scratches simultaneously.",
            self.ra_cfg.scratch_regs.len()
        );
        to_ice_reg(self.ra_cfg.scratch_regs[idx])
    }

    fn slot_of(&self, v: Reg, pt: ProgramPoint) -> AllocatedSlot {
        self.ra_res.at(v, pt).unwrap_or_else(|| {
            panic!("vreg {v} has no assignment at program point {pt}")
        })
    }

    /// Load an operand into a physical register suitable for reading at the
    /// given use point and return its `Reg` (preg) id. If the vreg lives in
    /// a register there, no MOV is emitted. If it's on the stack, load into
    /// `scratch_regs[scratch_idx]`.
    ///
    /// Width-specific emission paths use this and then convert through
    /// `to_ice_reg{8,16,32}` to avoid round-tripping via a 16-way linear
    /// search on an `AsmRegister64` handle.
    fn load_use_preg(&mut self, vreg: Reg, pt: ProgramPoint, scratch_idx: usize) -> Reg {
        match self.slot_of(vreg, pt) {
            AllocatedSlot::Reg(r) => r,
            AllocatedSlot::Stack(slot) => {
                let s_preg = self.ra_cfg.scratch_regs[scratch_idx];
                let s = to_ice_reg(s_preg);
                self.asm
                    .mov(s, rbp + i64::from(Self::slot_offset(slot)))
                    .expect("mov-load from slot");
                s_preg
            }
        }
    }

    fn load_use(&mut self, vreg: Reg, pt: ProgramPoint, scratch_idx: usize) -> AsmRegister64 {
        to_ice_reg(self.load_use_preg(vreg, pt, scratch_idx))
    }

    fn prepare_def_preg(&mut self, vreg: Reg, pt: ProgramPoint, scratch_idx: usize) -> Reg {
        match self.slot_of(vreg, pt) {
            AllocatedSlot::Reg(r) => r,
            AllocatedSlot::Stack(_) => self.ra_cfg.scratch_regs[scratch_idx],
        }
    }

    fn prepare_def(&mut self, vreg: Reg, pt: ProgramPoint, scratch_idx: usize) -> AsmRegister64 {
        to_ice_reg(self.prepare_def_preg(vreg, pt, scratch_idx))
    }

    fn store_def(&mut self, vreg: Reg, pt: ProgramPoint, scratch_idx: usize) {
        if let AllocatedSlot::Stack(slot) = self.slot_of(vreg, pt) {
            self.asm
                .mov(rbp + i64::from(Self::slot_offset(slot)), self.scratch(scratch_idx))
                .expect("mov-store to slot");
        }
    }

    /// `dst` is both read (at `use_pt`) and written (at `def_pt`).
    fn emit_rr_op<F>(&mut self, dst: Reg, src: Reg, use_pt: ProgramPoint, def_pt: ProgramPoint, op: F)
    where
        F: FnOnce(&mut CodeAssembler, AsmRegister64, AsmRegister64),
    {
        // If dst is Stack *and* its slot is the same at use_pt and def_pt,
        // we load from the same slot; standard case.
        let dst_r = self.load_use(dst, use_pt, 0);
        let src_r = self.load_use(src, use_pt, 1);
        op(&mut self.asm, dst_r, src_r);
        self.store_def(dst, def_pt, 0);
    }

    fn emit_prologue(&mut self) {
        self.asm.push(rbp).expect("push rbp");
        for &r in &self.saved_callee_regs {
            self.asm.push(to_ice_reg(r)).expect("push callee-saved");
        }
        let needs_pad_8 = self.saved_callee_regs.len() % 2 == 1;
        self.asm.mov(rbp, rsp).expect("mov rbp, rsp");
        let mut adj = self.frame_adjust;
        if needs_pad_8 {
            adj += 8;
        }
        if adj > 0 {
            self.asm.sub(rsp, adj as i32).expect("sub rsp, N");
        }
    }

    fn emit_epilogue(&mut self) {
        let needs_pad_8 = self.saved_callee_regs.len() % 2 == 1;
        let adj = self.frame_adjust + if needs_pad_8 { 8 } else { 0 };
        if adj > 0 {
            self.asm.add(rsp, adj as i32).expect("add rsp, N");
        }
        for &r in self.saved_callee_regs.iter().rev() {
            self.asm.pop(to_ice_reg(r)).expect("pop callee-saved");
        }
        self.asm.pop(rbp).expect("pop rbp");
        self.asm.ret().expect("ret");
    }

    /// Emit any split-store moves pending at this instruction's def-point.
    /// These preserve the evicted vreg's value before the new owner
    /// overwrites the preg.
    fn emit_pending_splits(&mut self, def_pt: ProgramPoint) {
        let Some(moves) = self.splits_by_point.get(&def_pt).cloned() else { return };
        for sm in moves {
            let reg = to_ice_reg(sm.from_preg);
            let off = i64::from(Self::slot_offset(sm.to_slot));
            self.asm.mov(rbp + off, reg).expect("split-store");
        }
    }

    fn emit_inst(
        &mut self,
        inst: &X64Inst,
        use_pt: ProgramPoint,
        def_pt: ProgramPoint,
        labels: &mut [CodeLabel],
    ) {
        match *inst {
            X64Inst::Mov64rr { dst, src } => {
                // Coalesced?
                if let (AllocatedSlot::Reg(a), AllocatedSlot::Reg(b)) =
                    (self.slot_of(dst, def_pt), self.slot_of(src, use_pt))
                    && a == b
                {
                    return;
                }
                let src_r = self.load_use(src, use_pt, 1);
                match self.slot_of(dst, def_pt) {
                    AllocatedSlot::Reg(r) => {
                        let dst_r = to_ice_reg(r);
                        self.asm.mov(dst_r, src_r).expect("mov rr");
                    }
                    AllocatedSlot::Stack(slot) => {
                        self.asm
                            .mov(rbp + i64::from(Self::slot_offset(slot)), src_r)
                            .expect("mov slot, rr");
                    }
                }
            }
            X64Inst::Mov64ri { dst, imm } => {
                // Before emitting, if `dst` is a call-target vreg, the
                // index of the iced instruction we're about to append
                // is the current length of `self.asm.instructions()`.
                if let Some(slot) = self.call_target_insts.get_mut(&dst) {
                    *slot = self.asm.instructions().len();
                }
                let dst_r = self.prepare_def(dst, def_pt, 0);
                self.asm.mov(dst_r, imm).expect("mov r, imm64");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov64rm { dst, src } => {
                let base_r = self.load_use(src.base, use_pt, 1);
                let dst_r = self.prepare_def(dst, def_pt, 0);
                if let Some(idx) = src.index {
                    let idx_r = self.load_use(idx, use_pt, 2);
                    self.asm
                        .mov(dst_r, base_r + idx_r * i32::from(src.scale) + src.disp)
                        .expect("mov r, [base+idx*s+disp]");
                } else {
                    self.asm
                        .mov(dst_r, base_r + i64::from(src.disp))
                        .expect("mov r, [base+disp]");
                }
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov64mr { dst, src } => {
                let base_r = self.load_use(dst.base, use_pt, 0);
                let src_r = self.load_use(src, use_pt, 1);
                if let Some(idx) = dst.index {
                    let idx_r = self.load_use(idx, use_pt, 2);
                    self.asm
                        .mov(base_r + idx_r * i32::from(dst.scale) + dst.disp, src_r)
                        .expect("mov [base+idx*s+disp], r");
                } else {
                    self.asm
                        .mov(base_r + i64::from(dst.disp), src_r)
                        .expect("mov [base+disp], r");
                }
            }
            // ----- 32-bit MOV variants. -----
            X64Inst::Mov32rr { dst, src } => {
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                self.asm
                    .mov(to_ice_reg32(dst_p), to_ice_reg32(src_p))
                    .expect("mov32 rr");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov32ri { dst, imm } => {
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                self.asm
                    .mov(to_ice_reg32(dst_p), imm)
                    .expect("mov32 r, imm");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov32rm { dst, src } => {
                let base_r = self.load_use(src.base, use_pt, 1);
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                if let Some(idx) = src.index {
                    let idx_r = self.load_use(idx, use_pt, 2);
                    self.asm
                        .mov(
                            to_ice_reg32(dst_p),
                            base_r + idx_r * i32::from(src.scale) + src.disp,
                        )
                        .expect("mov32 r, [mem]");
                } else {
                    self.asm
                        .mov(to_ice_reg32(dst_p), base_r + i64::from(src.disp))
                        .expect("mov32 r, [mem]");
                }
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov32mr { dst, src } => {
                let base_r = self.load_use(dst.base, use_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                if let Some(idx) = dst.index {
                    let idx_r = self.load_use(idx, use_pt, 2);
                    self.asm
                        .mov(
                            base_r + idx_r * i32::from(dst.scale) + dst.disp,
                            to_ice_reg32(src_p),
                        )
                        .expect("mov32 [mem], r");
                } else {
                    self.asm
                        .mov(base_r + i64::from(dst.disp), to_ice_reg32(src_p))
                        .expect("mov32 [mem], r");
                }
            }
            // ----- 16-bit MOV variants. -----
            X64Inst::Mov16rr { dst, src } => {
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                self.asm
                    .mov(to_ice_reg16(dst_p), to_ice_reg16(src_p))
                    .expect("mov16 rr");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov16ri { dst, imm } => {
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                self.asm
                    .mov(to_ice_reg16(dst_p), i32::from(imm))
                    .expect("mov16 r, imm");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov16rm { dst, src } => {
                let base_r = self.load_use(src.base, use_pt, 1);
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                if let Some(idx) = src.index {
                    let idx_r = self.load_use(idx, use_pt, 2);
                    self.asm
                        .mov(
                            to_ice_reg16(dst_p),
                            base_r + idx_r * i32::from(src.scale) + src.disp,
                        )
                        .expect("mov16 r, [mem]");
                } else {
                    self.asm
                        .mov(to_ice_reg16(dst_p), base_r + i64::from(src.disp))
                        .expect("mov16 r, [mem]");
                }
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov16mr { dst, src } => {
                let base_r = self.load_use(dst.base, use_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                if let Some(idx) = dst.index {
                    let idx_r = self.load_use(idx, use_pt, 2);
                    self.asm
                        .mov(
                            base_r + idx_r * i32::from(dst.scale) + dst.disp,
                            to_ice_reg16(src_p),
                        )
                        .expect("mov16 [mem], r");
                } else {
                    self.asm
                        .mov(base_r + i64::from(dst.disp), to_ice_reg16(src_p))
                        .expect("mov16 [mem], r");
                }
            }
            // ----- 8-bit MOV variants. -----
            X64Inst::Mov8rr { dst, src } => {
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                self.asm
                    .mov(to_ice_reg8(dst_p), to_ice_reg8(src_p))
                    .expect("mov8 rr");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov8ri { dst, imm } => {
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                self.asm
                    .mov(to_ice_reg8(dst_p), i32::from(imm))
                    .expect("mov8 r, imm");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov8rm { dst, src } => {
                let base_r = self.load_use(src.base, use_pt, 1);
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                if let Some(idx) = src.index {
                    let idx_r = self.load_use(idx, use_pt, 2);
                    self.asm
                        .mov(
                            to_ice_reg8(dst_p),
                            base_r + idx_r * i32::from(src.scale) + src.disp,
                        )
                        .expect("mov8 r, [mem]");
                } else {
                    self.asm
                        .mov(to_ice_reg8(dst_p), base_r + i64::from(src.disp))
                        .expect("mov8 r, [mem]");
                }
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Mov8mr { dst, src } => {
                let base_r = self.load_use(dst.base, use_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                if let Some(idx) = dst.index {
                    let idx_r = self.load_use(idx, use_pt, 2);
                    self.asm
                        .mov(
                            base_r + idx_r * i32::from(dst.scale) + dst.disp,
                            to_ice_reg8(src_p),
                        )
                        .expect("mov8 [mem], r");
                } else {
                    self.asm
                        .mov(base_r + i64::from(dst.disp), to_ice_reg8(src_p))
                        .expect("mov8 [mem], r");
                }
            }
            // ----- Sign-extending loads. -----
            X64Inst::Movsx64r8 { dst, src } => {
                let dst_r = self.prepare_def(dst, def_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                self.asm
                    .movsx(dst_r, to_ice_reg8(src_p))
                    .expect("movsx r64, r8");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Movsx64r16 { dst, src } => {
                let dst_r = self.prepare_def(dst, def_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                self.asm
                    .movsx(dst_r, to_ice_reg16(src_p))
                    .expect("movsx r64, r16");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Movsxd64r32 { dst, src } => {
                let dst_r = self.prepare_def(dst, def_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                self.asm
                    .movsxd(dst_r, to_ice_reg32(src_p))
                    .expect("movsxd r64, r32");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Movzx64r8 { dst, src } => {
                let dst_r = self.prepare_def(dst, def_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                self.asm
                    .movzx(dst_r, to_ice_reg8(src_p))
                    .expect("movzx r64, r8");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Movzx64r16 { dst, src } => {
                let dst_r = self.prepare_def(dst, def_pt, 0);
                let src_p = self.load_use_preg(src, use_pt, 1);
                self.asm
                    .movzx(dst_r, to_ice_reg16(src_p))
                    .expect("movzx r64, r16");
                self.store_def(dst, def_pt, 0);
            }
            // ----- LEA. -----
            X64Inst::Lea64rm { dst, src } => {
                let base_r = self.load_use(src.base, use_pt, 1);
                let dst_r = self.prepare_def(dst, def_pt, 0);
                if let Some(idx) = src.index {
                    let idx_r = self.load_use(idx, use_pt, 2);
                    self.asm
                        .lea(dst_r, base_r + idx_r * i32::from(src.scale) + src.disp)
                        .expect("lea r, [mem]");
                } else {
                    self.asm
                        .lea(dst_r, base_r + i64::from(src.disp))
                        .expect("lea r, [mem]");
                }
                self.store_def(dst, def_pt, 0);
            }
            // ----- Integer arithmetic. -----
            X64Inst::Add64rr { dst, src } => {
                self.emit_rr_op(dst, src, use_pt, def_pt, |a, d, s| {
                    a.add(d, s).expect("add rr");
                });
            }
            X64Inst::Sub64rr { dst, src } => {
                self.emit_rr_op(dst, src, use_pt, def_pt, |a, d, s| {
                    a.sub(d, s).expect("sub rr");
                });
            }
            X64Inst::Imul64rr { dst, src } => {
                self.emit_rr_op(dst, src, use_pt, def_pt, |a, d, s| {
                    a.imul_2(d, s).expect("imul rr");
                });
            }
            X64Inst::Add64ri32 { dst, imm } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.add(dst_r, imm).expect("add r, imm32");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Sub64ri32 { dst, imm } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.sub(dst_r, imm).expect("sub r, imm32");
                self.store_def(dst, def_pt, 0);
            }
            // ----- Bitwise. -----
            X64Inst::And64rr { dst, src } => {
                self.emit_rr_op(dst, src, use_pt, def_pt, |a, d, s| {
                    a.and(d, s).expect("and rr");
                });
            }
            X64Inst::Or64rr { dst, src } => {
                self.emit_rr_op(dst, src, use_pt, def_pt, |a, d, s| {
                    a.or(d, s).expect("or rr");
                });
            }
            X64Inst::Xor64rr { dst, src } => {
                self.emit_rr_op(dst, src, use_pt, def_pt, |a, d, s| {
                    a.xor(d, s).expect("xor rr");
                });
            }
            X64Inst::And64ri32 { dst, imm } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.and(dst_r, imm).expect("and r, imm32");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Or64ri32 { dst, imm } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.or(dst_r, imm).expect("or r, imm32");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Xor64ri32 { dst, imm } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.xor(dst_r, imm).expect("xor r, imm32");
                self.store_def(dst, def_pt, 0);
            }
            // ----- Unary. -----
            X64Inst::Not64r { dst } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.not(dst_r).expect("not r");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Neg64r { dst } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.neg(dst_r).expect("neg r");
                self.store_def(dst, def_pt, 0);
            }
            // ----- Shifts by imm8. -----
            X64Inst::Shl64ri8 { dst, imm } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.shl(dst_r, i32::from(imm)).expect("shl r, imm8");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Shr64ri8 { dst, imm } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.shr(dst_r, i32::from(imm)).expect("shr r, imm8");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Sar64ri8 { dst, imm } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.sar(dst_r, i32::from(imm)).expect("sar r, imm8");
                self.store_def(dst, def_pt, 0);
            }
            // ----- Shifts by CL. Frontend has pre-bound `count` to RCX. -----
            X64Inst::Shl64rcl { dst, count } => {
                assert_cl_pin(self.slot_of(count, use_pt), "shl");
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.shl(dst_r, cl).expect("shl r, cl");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Shr64rcl { dst, count } => {
                assert_cl_pin(self.slot_of(count, use_pt), "shr");
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.shr(dst_r, cl).expect("shr r, cl");
                self.store_def(dst, def_pt, 0);
            }
            X64Inst::Sar64rcl { dst, count } => {
                assert_cl_pin(self.slot_of(count, use_pt), "sar");
                let dst_r = self.load_use(dst, use_pt, 0);
                self.asm.sar(dst_r, cl).expect("sar r, cl");
                self.store_def(dst, def_pt, 0);
            }
            // ----- Division. Dividend/quotient/remainder all pre-bound. -----
            X64Inst::Idiv64r { divisor, hi_in, lo_in, .. } => {
                assert_preg_pin(self.slot_of(hi_in, use_pt), RDX, "idiv hi_in");
                assert_preg_pin(self.slot_of(lo_in, use_pt), RAX, "idiv lo_in");
                let div_r = self.load_use(divisor, use_pt, 0);
                self.asm.idiv(div_r).expect("idiv r");
            }
            X64Inst::Div64r { divisor, hi_in, lo_in, .. } => {
                assert_preg_pin(self.slot_of(hi_in, use_pt), RDX, "div hi_in");
                assert_preg_pin(self.slot_of(lo_in, use_pt), RAX, "div lo_in");
                let div_r = self.load_use(divisor, use_pt, 0);
                self.asm.div(div_r).expect("div r");
            }
            // ----- Compare / test. -----
            X64Inst::Cmp64rr { lhs, rhs } => {
                let lhs_r = self.load_use(lhs, use_pt, 0);
                let rhs_r = self.load_use(rhs, use_pt, 1);
                self.asm.cmp(lhs_r, rhs_r).expect("cmp rr");
            }
            X64Inst::Cmp64ri32 { lhs, imm } => {
                let lhs_r = self.load_use(lhs, use_pt, 0);
                self.asm.cmp(lhs_r, imm).expect("cmp r, imm32");
            }
            X64Inst::Test64rr { lhs, rhs } => {
                let lhs_r = self.load_use(lhs, use_pt, 0);
                let rhs_r = self.load_use(rhs, use_pt, 1);
                self.asm.test(lhs_r, rhs_r).expect("test rr");
            }
            X64Inst::Test64ri32 { lhs, imm } => {
                let lhs_r = self.load_use(lhs, use_pt, 0);
                self.asm.test(lhs_r, imm).expect("test r, imm32");
            }
            // ----- Conditional move. -----
            X64Inst::Cmov64rr { cond, dst, src } => {
                let dst_r = self.load_use(dst, use_pt, 0);
                let src_r = self.load_use(src, use_pt, 1);
                emit_cmov(&mut self.asm, cond, dst_r, src_r);
                self.store_def(dst, def_pt, 0);
            }
            // ----- Set-byte-on-condition. -----
            X64Inst::Setcc8r { cond, dst } => {
                let dst_p = self.prepare_def_preg(dst, def_pt, 0);
                emit_setcc(&mut self.asm, cond, to_ice_reg8(dst_p));
                self.store_def_byte(dst, def_pt, 0);
            }
            // ----- Call (indirect). -----
            X64Inst::Call64r { target } => {
                let tgt_r = self.load_use(target, use_pt, 0);
                self.asm.call(tgt_r).expect("call r");
            }
            // ----- Control flow. -----
            X64Inst::Jmp { dst } => {
                self.asm.jmp(labels[dst.index()]).expect("jmp label");
            }
            X64Inst::CondJmp { cond, taken, not_taken } => {
                let taken_lbl = labels[taken.index()];
                let not_taken_lbl = labels[not_taken.index()];
                match cond {
                    Cond::Z => self.asm.jz(taken_lbl).expect("jz"),
                    Cond::NZ => self.asm.jnz(taken_lbl).expect("jnz"),
                    Cond::L => self.asm.jl(taken_lbl).expect("jl"),
                    Cond::LE => self.asm.jle(taken_lbl).expect("jle"),
                    Cond::G => self.asm.jg(taken_lbl).expect("jg"),
                    Cond::GE => self.asm.jge(taken_lbl).expect("jge"),
                    Cond::B => self.asm.jb(taken_lbl).expect("jb"),
                    Cond::BE => self.asm.jbe(taken_lbl).expect("jbe"),
                    Cond::A => self.asm.ja(taken_lbl).expect("ja"),
                    Cond::AE => self.asm.jae(taken_lbl).expect("jae"),
                }
                self.asm.jmp(not_taken_lbl).expect("jmp fallthrough");
            }
            X64Inst::Jmp64r { target } => {
                let t_r = self.load_use(target, use_pt, 0);
                self.asm.jmp(t_r).expect("jmp r");
            }
            X64Inst::Ud2 => {
                self.asm.ud2().expect("ud2");
            }
            X64Inst::Mfence => {
                self.asm.mfence().expect("mfence");
            }
            X64Inst::RawRet => self.emit_epilogue(),
        }
    }

    /// Byte-sized variant of `store_def` — writes only the low byte of
    /// the scratch register back to the slot. Used by `Setcc8r`.
    ///
    /// Invariant: the upper 7 bytes of the spill slot are left with
    /// whatever was there previously. Every consumer of an `i1` / 8-bit
    /// vreg in this IR (`Mov8rr`, `Mov8mr`, `Movsx64r8`, `Movzx64r8`)
    /// reads exactly one byte, so the undefined upper bytes are harmless.
    /// If the IR ever grows a `Mov64rr` that reads an 8-bit vreg as 64
    /// bits, this helper must be upgraded to zero-extend into the slot.
    fn store_def_byte(&mut self, vreg: Reg, pt: ProgramPoint, scratch_idx: usize) {
        if let AllocatedSlot::Stack(slot) = self.slot_of(vreg, pt) {
            let s = to_ice_reg8(self.ra_cfg.scratch_regs[scratch_idx]);
            self.asm
                .mov(rbp + i64::from(Self::slot_offset(slot)), s)
                .expect("mov-store byte to slot");
        }
    }

    fn emit_pseudo(
        &mut self,
        p: &PseudoInstruction,
        use_pt: ProgramPoint,
        def_pt: ProgramPoint,
    ) {
        match *p {
            PseudoInstruction::Arg { .. } => {
                // Pinned-shim def. ABI already put the value in the preg.
            }
            PseudoInstruction::Copy { dst, src } => {
                let src_slot = self.slot_of(src, use_pt);
                let dst_slot = self.slot_of(dst, def_pt);
                // Coalesced on the same preg → no-op.
                if let (AllocatedSlot::Reg(a), AllocatedSlot::Reg(b)) = (dst_slot, src_slot)
                    && a == b
                {
                    return;
                }
                // Otherwise emit Mov64rr semantics using the live slots.
                let src_r = self.load_use(src, use_pt, 1);
                match dst_slot {
                    AllocatedSlot::Reg(r) => {
                        let dst_r = to_ice_reg(r);
                        self.asm.mov(dst_r, src_r).expect("copy: mov rr");
                    }
                    AllocatedSlot::Stack(slot) => {
                        self.asm
                            .mov(rbp + i64::from(Self::slot_offset(slot)), src_r)
                            .expect("copy: mov slot, r");
                    }
                }
            }
            PseudoInstruction::Return { .. } => {
                panic!("Return pseudo should have been lowered to RawRet before emission");
            }
            PseudoInstruction::Phi { .. } => {
                panic!("Phi pseudo should have been lowered to parallel Copies before emission");
            }
            PseudoInstruction::CallPseudo { .. } => {
                panic!("CallPseudo should have been lowered to a target CALL before emission");
            }
            PseudoInstruction::StackAlloc { dst, .. } => {
                let disp = *self.alloca_offsets.get(&dst).unwrap_or_else(|| {
                    panic!("StackAlloc for vreg {dst} has no computed frame offset")
                });
                let dst_preg = self.prepare_def_preg(dst, def_pt, 0);
                let dst_r = to_ice_reg(dst_preg);
                self.asm
                    .lea(dst_r, rbp + i64::from(disp))
                    .expect("lea rbp-rel for stack alloca");
                self.store_def(dst, def_pt, 0);
            }
            PseudoInstruction::FrameSetup | PseudoInstruction::FrameDestroy => {
                panic!("Frame markers should have been replaced by prologue/epilogue sequences");
            }
            PseudoInstruction::ImplicitDef { .. }
            | PseudoInstruction::Kill { .. }
            | PseudoInstruction::RegDef { .. } => {
                // No separate pseudo_cleanup pass runs today. These pseudos
                // carry regalloc-only information (pins, undef defs,
                // explicit live-range endpoints) and emit no machine code.
            }
        }
    }

    pub fn emit_fn(&mut self) -> Vec<u8> {
        self.emit_fn_with_relocs(&[]).bytes
    }

    /// Full emission path that surfaces call-site relocations so the
    /// loader can patch the placeholder immediate in each call-target
    /// `Mov64ri`. Pass `call_sites` from the ABI lowering pass —
    /// their `addr_vreg` fields mark which `Mov64ri` destinations we
    /// need to track by iced instruction index, and whose final byte
    /// offset we compute via `CodeAssemblerResult::new_instruction_offsets`.
    pub fn emit_fn_with_relocs(
        &mut self,
        call_sites: &[crate::codegen::passes::CallSite],
    ) -> EmittedFunc {
        self.check_scratch_budget();
        self.emit_prologue();

        // Register tracked addr vregs up front.
        for cs in call_sites {
            if !cs.symbol.is_empty() {
                self.mark_call_target(cs.addr_vreg);
            }
        }

        let mut labels: Vec<CodeLabel> = (0..self.func.blocks_count())
            .map(|_| self.asm.create_label())
            .collect();

        for (block, block_data) in self.func.blocks_iter() {
            self.asm
                .set_label(&mut labels[block.index()])
                .expect("set_label");
            for (idx, instr) in block_data.iter().enumerate() {
                let i = idx as u32;
                let use_pt = self.layout.use_pt(block, i);
                let def_pt = self.layout.def_pt(block, i);

                // If the allocator split a vreg's life at this def_pt, save
                // its preg to the stack slot BEFORE the inst executes. The
                // inst (or Copy) then freely overwrites the preg for the
                // new owner.
                self.emit_pending_splits(def_pt);

                match instr {
                    Instruction::Target(x64_inst) => {
                        self.emit_inst(x64_inst, use_pt, def_pt, &mut labels);
                    }
                    Instruction::Pseudo(p) => self.emit_pseudo(p, use_pt, def_pt),
                }
            }
        }

        use iced_x86::BlockEncoderOptions;
        let res = self
            .asm
            .assemble_options(0, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS)
            .expect("assemble_options");

        // Build relocations. For each call site whose addr_vreg was
        // tracked, find its iced instruction offset and add 2 (REX +
        // opcode byte) to reach the 8-byte immediate.
        let mut relocations = Vec::with_capacity(call_sites.len());
        for cs in call_sites {
            if cs.symbol.is_empty() {
                continue; // indirect call — no patch needed
            }
            let Some(&iced_idx) = self.call_target_insts.get(&cs.addr_vreg) else {
                continue;
            };
            assert!(
                iced_idx != usize::MAX,
                "call-target Mov64ri for vreg {} was never emitted",
                cs.addr_vreg
            );
            let inst_offset = res.inner.new_instruction_offsets[iced_idx] as usize;
            relocations.push(EmittedCallReloc {
                imm_offset: inst_offset + 2, // MOV r64,imm64 = REX(1)+opcode(1)+imm(8)
                symbol: cs.symbol.clone(),
            });
        }

        EmittedFunc {
            bytes: res.inner.code_buffer,
            relocations,
        }
    }
}

fn emit_cmov(
    asm: &mut CodeAssembler,
    cond: Cond,
    dst: AsmRegister64,
    src: AsmRegister64,
) {
    match cond {
        Cond::Z => asm.cmove(dst, src).expect("cmove"),
        Cond::NZ => asm.cmovne(dst, src).expect("cmovne"),
        Cond::L => asm.cmovl(dst, src).expect("cmovl"),
        Cond::LE => asm.cmovle(dst, src).expect("cmovle"),
        Cond::G => asm.cmovg(dst, src).expect("cmovg"),
        Cond::GE => asm.cmovge(dst, src).expect("cmovge"),
        Cond::B => asm.cmovb(dst, src).expect("cmovb"),
        Cond::BE => asm.cmovbe(dst, src).expect("cmovbe"),
        Cond::A => asm.cmova(dst, src).expect("cmova"),
        Cond::AE => asm.cmovae(dst, src).expect("cmovae"),
    }
}

fn emit_setcc(asm: &mut CodeAssembler, cond: Cond, dst: AsmRegister8) {
    match cond {
        Cond::Z => asm.sete(dst).expect("sete"),
        Cond::NZ => asm.setne(dst).expect("setne"),
        Cond::L => asm.setl(dst).expect("setl"),
        Cond::LE => asm.setle(dst).expect("setle"),
        Cond::G => asm.setg(dst).expect("setg"),
        Cond::GE => asm.setge(dst).expect("setge"),
        Cond::B => asm.setb(dst).expect("setb"),
        Cond::BE => asm.setbe(dst).expect("setbe"),
        Cond::A => asm.seta(dst).expect("seta"),
        Cond::AE => asm.setae(dst).expect("setae"),
    }
}

fn assert_cl_pin(slot: AllocatedSlot, op_name: &str) {
    assert_preg_pin(slot, RCX, op_name);
}

fn assert_preg_pin(slot: AllocatedSlot, expected: Reg, op_name: &str) {
    match slot {
        AllocatedSlot::Reg(r) if r == expected => {}
        other => panic!(
            "{op_name}: implicit-operand vreg must be pre-bound to preg {expected} via \
             `reg_bind`, but is at {other:?}. Frontend should emit `RegDef` / populate \
             `reg_bind` so the regalloc keeps the vreg in the required physical register."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::analysis::cfg::CFG;
    use crate::codegen::isa::x64::passes::abi_lower::SysVAmd64Lowering;
    use crate::codegen::passes::AbiLowering;
    use crate::codegen::regalloc::{LinearScan, RegAllocConfig, RegAllocator};
    use crate::codegen::tir::{Func, PseudoInstruction};
    use std::collections::HashMap;

    fn test_ra_config(reg_bind: HashMap<Reg, Reg>) -> RegAllocConfig {
        RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11],
            scratch_regs: vec![RBX, R12, R13],
            reg_bind,
        }
    }

    #[test]
    fn emit_identity_function_assembles_without_panicking() {
        let mut func = Func::<X64Inst>::new("identity".to_string());
        let b = func.add_empty_block();
        let a = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: a, idx: 0 });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: a });
        }
        let abi = SysVAmd64Lowering.lower(&mut func);
        let cfg = CFG::compute(&func).unwrap();
        let cfg_cfg = test_ra_config(abi.reg_bind);
        let res = LinearScan::allocate(&func, &cfg, &cfg_cfg);
        let mut w = FnMCWriter::new(&func, &cfg_cfg, &res);
        let bytes = w.emit_fn();
        assert!(bytes.len() >= 4);
        assert_eq!(bytes[0], 0x55); // push rbp
        assert_eq!(*bytes.last().unwrap(), 0xC3); // ret
    }

    #[test]
    fn scratch_index_out_of_range_panics_with_clear_message() {
        use crate::codegen::regalloc::RegAllocResult;
        use crate::support::slotmap::SecondaryMap;

        let func = Func::<X64Inst>::new("t".to_string());
        let empty_cfg = RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RAX],
            scratch_regs: vec![],
            reg_bind: HashMap::new(),
        };
        let empty_ra = RegAllocResult {
            assignments: SecondaryMap::new(0),
            frame_layout: Vec::new(),
            frame_size: 0,
            split_moves: Vec::new(),
        };
        let w = FnMCWriter::new(&func, &empty_cfg, &empty_ra);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.scratch(0)));
        assert!(caught.is_err());
    }

    // ----- MC-emission tests for the newly added instructions. -----
    //
    // Each test constructs a tiny function, lowers ABI, allocates registers,
    // and emits machine code — and asserts that it (a) doesn't panic, and
    // (b) produces a well-formed prologue/epilogue with at least one byte
    // of body in between. The goal is to exercise every new opcode's MC
    // path end-to-end: `iced-x86` will itself reject malformed operands, so
    // "assembles without error" is the real assertion.

    use crate::codegen::isa::x64::inst::{Cond, Mem};
    use crate::codegen::tir::Instruction;

    /// Run the whole pipeline and return the emitted bytes. Any additional
    /// pre-bindings are merged on top of the abi-lower output.
    fn emit_with_binds(
        mut func: Func<X64Inst>,
        extra_binds: &[(Reg, Reg)],
    ) -> Vec<u8> {
        let abi = SysVAmd64Lowering.lower(&mut func);
        let mut reg_bind = abi.reg_bind;
        for (v, p) in extra_binds {
            reg_bind.insert(*v, *p);
        }
        let cfg = CFG::compute(&func).unwrap();
        let cfg_cfg = test_ra_config(reg_bind);
        let res = LinearScan::allocate(&func, &cfg, &cfg_cfg);
        let mut w = FnMCWriter::new(&func, &cfg_cfg, &res);
        w.emit_fn()
    }

    fn assert_has_prologue_and_epilogue(bytes: &[u8]) {
        assert!(bytes.len() >= 4, "code too short: {bytes:?}");
        assert_eq!(bytes[0], 0x55, "expected push rbp as first byte"); // push rbp
        assert_eq!(*bytes.last().unwrap(), 0xC3, "expected ret as last byte");
    }

    /// Scaffolding: build a single-block func `args → body → return src`,
    /// hand the caller the pre-allocated vregs via `body`.
    fn with_unary_body<F>(body: F) -> Func<X64Inst>
    where
        F: FnOnce(&mut Func<X64Inst>, Reg /*arg0*/) -> Reg /*return value*/,
    {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let arg0 = func.new_vreg();
        func.get_block_data_mut(b)
            .push_pseudo_inst(PseudoInstruction::Arg { dst: arg0, idx: 0 });
        let ret = body(&mut func, arg0);
        func.get_block_data_mut(b)
            .push_pseudo_inst(PseudoInstruction::Return { src: ret });
        func
    }

    fn with_binary_body<F>(body: F) -> Func<X64Inst>
    where
        F: FnOnce(&mut Func<X64Inst>, Reg /*arg0*/, Reg /*arg1*/) -> Reg,
    {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let a0 = func.new_vreg();
        let a1 = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: a0, idx: 0 });
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: a1, idx: 1 });
        }
        let ret = body(&mut func, a0, a1);
        func.get_block_data_mut(b)
            .push_pseudo_inst(PseudoInstruction::Return { src: ret });
        func
    }

    fn entry(func: &Func<X64Inst>) -> crate::codegen::tir::Block {
        func.get_entry_block().unwrap()
    }

    #[test]
    fn emit_mov32rr_assembles() {
        let func = with_unary_body(|f, a0| {
            let b = entry(f);
            let dst = f.new_vreg();
            f.get_block_data_mut(b)
                .push_target_inst(X64Inst::Mov32rr { dst, src: a0 });
            dst
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
    }

    #[test]
    fn emit_mov16rr_and_mov8rr_assemble() {
        let func16 = with_unary_body(|f, a0| {
            let b = entry(f);
            let d = f.new_vreg();
            f.get_block_data_mut(b)
                .push_target_inst(X64Inst::Mov16rr { dst: d, src: a0 });
            d
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func16, &[]));

        let func8 = with_unary_body(|f, a0| {
            let b = entry(f);
            let d = f.new_vreg();
            f.get_block_data_mut(b)
                .push_target_inst(X64Inst::Mov8rr { dst: d, src: a0 });
            d
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func8, &[]));
    }

    #[test]
    fn emit_mov32rm_and_mov32mr_assemble() {
        // load dword from [arg0]
        let func_ld = with_unary_body(|f, a0| {
            let b = entry(f);
            let d = f.new_vreg();
            f.get_block_data_mut(b).push_target_inst(X64Inst::Mov32rm {
                dst: d,
                src: Mem::base(a0),
            });
            d
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func_ld, &[]));

        // store dword to [arg0] from arg1
        let func_st = with_binary_body(|f, a0, a1| {
            let b = entry(f);
            f.get_block_data_mut(b).push_target_inst(X64Inst::Mov32mr {
                dst: Mem::base(a0),
                src: a1,
            });
            a1
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func_st, &[]));
    }

    #[test]
    fn emit_mov8_and_mov16_mem_variants_assemble() {
        // 8-bit load
        let func_ld8 = with_unary_body(|f, a0| {
            let b = entry(f);
            let d = f.new_vreg();
            f.get_block_data_mut(b).push_target_inst(X64Inst::Mov8rm {
                dst: d,
                src: Mem::base(a0),
            });
            d
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func_ld8, &[]));

        // 16-bit load
        let func_ld16 = with_unary_body(|f, a0| {
            let b = entry(f);
            let d = f.new_vreg();
            f.get_block_data_mut(b).push_target_inst(X64Inst::Mov16rm {
                dst: d,
                src: Mem::base(a0),
            });
            d
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func_ld16, &[]));
    }

    #[test]
    fn emit_movsxd_and_movzx_assemble() {
        for inst_ctor in [
            |d, s| X64Inst::Movsxd64r32 { dst: d, src: s },
            |d, s| X64Inst::Movsx64r8 { dst: d, src: s },
            |d, s| X64Inst::Movsx64r16 { dst: d, src: s },
            |d, s| X64Inst::Movzx64r8 { dst: d, src: s },
            |d, s| X64Inst::Movzx64r16 { dst: d, src: s },
        ] {
            let func = with_unary_body(|f, a0| {
                let b = entry(f);
                let d = f.new_vreg();
                f.get_block_data_mut(b).push_target_inst(inst_ctor(d, a0));
                d
            });
            assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
        }
    }

    #[test]
    fn emit_lea64rm_assembles_with_indexed_addressing() {
        let func = with_binary_body(|f, a0, a1| {
            let b = entry(f);
            let d = f.new_vreg();
            f.get_block_data_mut(b).push_target_inst(X64Inst::Lea64rm {
                dst: d,
                src: Mem {
                    base: a0,
                    index: Some(a1),
                    scale: 4,
                    disp: 16,
                },
            });
            d
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
    }

    #[test]
    fn emit_bitwise_and_or_xor_rr_assemble() {
        for ctor in [
            |d, s| X64Inst::And64rr { dst: d, src: s },
            |d, s| X64Inst::Or64rr { dst: d, src: s },
            |d, s| X64Inst::Xor64rr { dst: d, src: s },
        ] {
            let func = with_binary_body(|f, a0, a1| {
                let b = entry(f);
                let d = f.new_vreg();
                {
                    let bd = f.get_block_data_mut(b);
                    bd.push_pseudo_inst(PseudoInstruction::Copy { dst: d, src: a0 });
                    bd.push_target_inst(ctor(d, a1));
                }
                d
            });
            assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
        }
    }

    #[test]
    fn emit_and_or_xor_ri32_assemble() {
        for ctor in [
            |d, i| X64Inst::And64ri32 { dst: d, imm: i },
            |d, i| X64Inst::Or64ri32 { dst: d, imm: i },
            |d, i| X64Inst::Xor64ri32 { dst: d, imm: i },
        ] {
            let func = with_unary_body(|f, a0| {
                let b = entry(f);
                let d = f.new_vreg();
                {
                    let bd = f.get_block_data_mut(b);
                    bd.push_pseudo_inst(PseudoInstruction::Copy { dst: d, src: a0 });
                    bd.push_target_inst(ctor(d, 0x7F));
                }
                d
            });
            assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
        }
    }

    #[test]
    fn emit_not_and_neg_assemble() {
        for ctor in [
            |d| X64Inst::Not64r { dst: d },
            |d| X64Inst::Neg64r { dst: d },
        ] {
            let func = with_unary_body(|f, a0| {
                let b = entry(f);
                let d = f.new_vreg();
                {
                    let bd = f.get_block_data_mut(b);
                    bd.push_pseudo_inst(PseudoInstruction::Copy { dst: d, src: a0 });
                    bd.push_target_inst(ctor(d));
                }
                d
            });
            assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
        }
    }

    #[test]
    fn emit_shift_by_imm_assembles() {
        for ctor in [
            |d, i| X64Inst::Shl64ri8 { dst: d, imm: i },
            |d, i| X64Inst::Shr64ri8 { dst: d, imm: i },
            |d, i| X64Inst::Sar64ri8 { dst: d, imm: i },
        ] {
            let func = with_unary_body(|f, a0| {
                let b = entry(f);
                let d = f.new_vreg();
                {
                    let bd = f.get_block_data_mut(b);
                    bd.push_pseudo_inst(PseudoInstruction::Copy { dst: d, src: a0 });
                    bd.push_target_inst(ctor(d, 3));
                }
                d
            });
            assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
        }
    }

    #[test]
    fn emit_shift_by_cl_assembles_when_count_is_pinned_to_rcx() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let a0 = func.new_vreg();
        let a1 = func.new_vreg();
        let d = func.new_vreg();
        let count = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: a0, idx: 0 });
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: a1, idx: 1 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: d, src: a0 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: count, src: a1 });
            bd.push_target_inst(X64Inst::Shl64rcl { dst: d, count });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: d });
        }
        assert_has_prologue_and_epilogue(&emit_with_binds(func, &[(count, RCX)]));
    }

    #[test]
    fn shift_by_cl_panics_if_count_not_pinned_to_rcx() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let a0 = func.new_vreg();
        let a1 = func.new_vreg();
        let d = func.new_vreg();
        let count = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: a0, idx: 0 });
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: a1, idx: 1 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: d, src: a0 });
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: count, src: a1 });
            bd.push_target_inst(X64Inst::Shl64rcl { dst: d, count });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: d });
        }
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Deliberately omit the `count → RCX` binding.
            emit_with_binds(func, &[]);
        }));
        assert!(caught.is_err(), "emitter must refuse an un-pinned shift count");
    }

    #[test]
    fn emit_idiv_assembles_with_dividend_and_result_pins() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let divisor = func.new_vreg();
        let hi_in = func.new_vreg();
        let lo_in = func.new_vreg();
        let quotient = func.new_vreg();
        let remainder = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: divisor, idx: 0 });
            // Pretend the dividend halves are produced elsewhere — zero them.
            bd.push_target_inst(X64Inst::Mov64ri { dst: hi_in, imm: 0 });
            bd.push_target_inst(X64Inst::Mov64ri { dst: lo_in, imm: 42 });
            bd.push_target_inst(X64Inst::Idiv64r {
                divisor,
                hi_in,
                lo_in,
                quotient,
                remainder,
            });
            // Return `divisor` so the RAX pin for `quotient` + the RAX pin
            // for the return-value shim don't fight each other — both get a
            // chance to land at RAX at disjoint program points.
            bd.push_pseudo_inst(PseudoInstruction::Return { src: divisor });
        }
        assert_has_prologue_and_epilogue(&emit_with_binds(
            func,
            &[(hi_in, RDX), (lo_in, RAX), (quotient, RAX), (remainder, RDX)],
        ));
    }

    #[test]
    fn emit_div_unsigned_assembles() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let divisor = func.new_vreg();
        let hi_in = func.new_vreg();
        let lo_in = func.new_vreg();
        let quotient = func.new_vreg();
        let remainder = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: divisor, idx: 0 });
            bd.push_target_inst(X64Inst::Mov64ri { dst: hi_in, imm: 0 });
            bd.push_target_inst(X64Inst::Mov64ri { dst: lo_in, imm: 42 });
            bd.push_target_inst(X64Inst::Div64r {
                divisor,
                hi_in,
                lo_in,
                quotient,
                remainder,
            });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: divisor });
        }
        assert_has_prologue_and_epilogue(&emit_with_binds(
            func,
            &[(hi_in, RDX), (lo_in, RAX), (quotient, RAX), (remainder, RDX)],
        ));
    }

    #[test]
    fn emit_test_rr_and_ri32_assemble() {
        let func_rr = with_binary_body(|f, a0, a1| {
            let b = entry(f);
            f.get_block_data_mut(b)
                .push_target_inst(X64Inst::Test64rr { lhs: a0, rhs: a1 });
            a0
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func_rr, &[]));

        let func_ri = with_unary_body(|f, a0| {
            let b = entry(f);
            f.get_block_data_mut(b).push_target_inst(X64Inst::Test64ri32 {
                lhs: a0,
                imm: 0x1,
            });
            a0
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func_ri, &[]));
    }

    #[test]
    fn emit_cmov_assembles_with_all_conditions() {
        for cond in [
            Cond::Z, Cond::NZ, Cond::L, Cond::LE, Cond::G, Cond::GE, Cond::B, Cond::BE,
            Cond::A, Cond::AE,
        ] {
            let func = with_binary_body(|f, a0, a1| {
                let b = entry(f);
                let d = f.new_vreg();
                {
                    let bd = f.get_block_data_mut(b);
                    bd.push_pseudo_inst(PseudoInstruction::Copy { dst: d, src: a0 });
                    bd.push_target_inst(X64Inst::Cmp64rr { lhs: a0, rhs: a1 });
                    bd.push_target_inst(X64Inst::Cmov64rr { cond, dst: d, src: a1 });
                }
                d
            });
            assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
        }
    }

    #[test]
    fn emit_setcc_assembles_with_all_conditions() {
        for cond in [
            Cond::Z, Cond::NZ, Cond::L, Cond::LE, Cond::G, Cond::GE, Cond::B, Cond::BE,
            Cond::A, Cond::AE,
        ] {
            let func = with_unary_body(|f, a0| {
                let b = entry(f);
                let d = f.new_vreg();
                {
                    let bd = f.get_block_data_mut(b);
                    bd.push_target_inst(X64Inst::Cmp64ri32 { lhs: a0, imm: 0 });
                    bd.push_target_inst(X64Inst::Setcc8r { cond, dst: d });
                }
                d
            });
            assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
        }
    }

    #[test]
    fn emit_call_indirect_assembles() {
        // arg0 holds a function pointer; call through it, then return a
        // freshly-minted value so no caller-saved vreg crosses the call.
        let func = with_unary_body(|f, a0| {
            let b = entry(f);
            let after = f.new_vreg();
            {
                let bd = f.get_block_data_mut(b);
                bd.push_target_inst(X64Inst::Call64r { target: a0 });
                bd.push_target_inst(X64Inst::Mov64ri { dst: after, imm: 0 });
            }
            after
        });
        assert_has_prologue_and_epilogue(&emit_with_binds(func, &[]));
    }

    #[test]
    fn emit_pseudo_panics_if_phi_reaches_mc() {
        // Smoke test that the emitter refuses to handle a Phi that wasn't
        // lowered. Wrapped in `catch_unwind` so the expected panic doesn't
        // poison the harness.
        use crate::codegen::regalloc::RegAllocResult;
        use crate::support::slotmap::SecondaryMap;

        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let v = func.new_vreg();
        let pid = func.new_phi(Vec::new());
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Phi { dst: v, id: pid });
            bd.push_target_inst(X64Inst::RawRet);
        }
        let ra_cfg = test_ra_config(HashMap::new());
        let res = RegAllocResult {
            assignments: SecondaryMap::new(0),
            frame_layout: Vec::new(),
            frame_size: 0,
            split_moves: Vec::new(),
        };
        let mut w = FnMCWriter::new(&func, &ra_cfg, &res);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            w.emit_fn();
        }));
        let payload = caught.expect_err("emit_pseudo must panic on Phi");
        let msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("");
        assert!(
            msg.contains("Phi"),
            "panic payload should name the Phi pseudo, got {msg:?}"
        );
    }

    /// Golden-byte regression: confirm each `Cond` picks the matching
    /// `setCC` opcode, not some other one. The x86 opcode for `set<cond> al`
    /// is `0F XX C0` — different `XX` per condition. If `emit_setcc`
    /// accidentally swaps arms (e.g. mapping `Cond::Z → setl`), a
    /// "doesn't panic" test still passes but this one catches it.
    #[test]
    fn setcc_opcode_bytes_match_condition() {
        // Second opcode byte per cond, per Intel SDM vol 2 table 4-8.
        let cases: &[(Cond, u8)] = &[
            (Cond::Z, 0x94),
            (Cond::NZ, 0x95),
            (Cond::L, 0x9C),
            (Cond::LE, 0x9E),
            (Cond::G, 0x9F),
            (Cond::GE, 0x9D),
            (Cond::B, 0x92),
            (Cond::BE, 0x96),
            (Cond::A, 0x97),
            (Cond::AE, 0x93),
        ];
        for (cond, opcode) in cases {
            let func = with_unary_body(|f, a0| {
                let b = entry(f);
                let d = f.new_vreg();
                {
                    let bd = f.get_block_data_mut(b);
                    bd.push_target_inst(X64Inst::Cmp64ri32 { lhs: a0, imm: 0 });
                    bd.push_target_inst(X64Inst::Setcc8r { cond: *cond, dst: d });
                }
                d
            });
            let bytes = emit_with_binds(func, &[]);
            // Look for the pair `0F XX` anywhere in the body. Every setCC
            // begins with 0x0F followed by the condition-specific opcode.
            let hit = bytes.windows(2).any(|w| w[0] == 0x0F && w[1] == *opcode);
            assert!(
                hit,
                "expected setCC opcode 0F {opcode:02X} for {cond:?} in {bytes:02X?}"
            );
        }
    }

    /// Mirror regression test for `cmovCC`. Each `Cond` must pick its
    /// own iced method — `0F 4X` where X depends on the condition.
    #[test]
    fn cmov_opcode_bytes_match_condition() {
        let cases: &[(Cond, u8)] = &[
            (Cond::Z, 0x44),
            (Cond::NZ, 0x45),
            (Cond::L, 0x4C),
            (Cond::LE, 0x4E),
            (Cond::G, 0x4F),
            (Cond::GE, 0x4D),
            (Cond::B, 0x42),
            (Cond::BE, 0x46),
            (Cond::A, 0x47),
            (Cond::AE, 0x43),
        ];
        for (cond, opcode) in cases {
            let func = with_binary_body(|f, a0, a1| {
                let b = entry(f);
                let d = f.new_vreg();
                {
                    let bd = f.get_block_data_mut(b);
                    bd.push_pseudo_inst(PseudoInstruction::Copy { dst: d, src: a0 });
                    bd.push_target_inst(X64Inst::Cmp64rr { lhs: a0, rhs: a1 });
                    bd.push_target_inst(X64Inst::Cmov64rr {
                        cond: *cond,
                        dst: d,
                        src: a1,
                    });
                }
                d
            });
            let bytes = emit_with_binds(func, &[]);
            let hit = bytes.windows(2).any(|w| w[0] == 0x0F && w[1] == *opcode);
            assert!(
                hit,
                "expected cmovCC opcode 0F {opcode:02X} for {cond:?} in {bytes:02X?}"
            );
        }
    }

    /// Round-trip sanity: a function containing one new instruction survives
    /// iteration and still has the expected instruction at its tail.
    #[test]
    fn inserted_instruction_is_preserved_in_block() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let a = func.new_vreg();
        let d = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: a, idx: 0 });
            bd.push_target_inst(X64Inst::Movsxd64r32 { dst: d, src: a });
            bd.push_pseudo_inst(PseudoInstruction::Return { src: d });
        }
        let bd = func.get_block_data(b);
        let mut saw_movsxd = false;
        for inst in bd.iter() {
            if matches!(inst, Instruction::Target(X64Inst::Movsxd64r32 { .. })) {
                saw_movsxd = true;
            }
        }
        assert!(saw_movsxd);
    }
}
