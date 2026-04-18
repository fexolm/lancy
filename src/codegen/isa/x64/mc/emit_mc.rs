//! Machine-code emission for the x86-64 target.
//!
//! **Requires:** Pseudo cleanup has run. Every surviving instruction is either
//! a `Target(X64Inst)` or an erasable no-op (none should remain). Every vreg
//! appears in `RegAllocResult::coloring` as either a preg or a stack slot.
//!
//! **Effect:** Emits a flat `Vec<u8>` of x86-64 machine code via iced-x86.
//! Inserts the prologue (`push rbp; mov rbp, rsp; sub rsp, N`) and epilogue
//! (`add rsp, N; pop rbp; ret`) around the user body.
//!
//! **Spill handling:** When an operand is stack-allocated, we load into /
//! store out of a scratch register around the instruction. Scratch registers
//! must not overlap the allocatable pool — the frontend's
//! `RegAllocConfig` is responsible for that.

use crate::codegen::isa::x64::inst::{Cond, X64Inst};
use crate::codegen::isa::x64::regs::{
    R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP,
};
use crate::codegen::isa::x64::sysv::CALLEE_SAVED;
use crate::codegen::regalloc::{AllocatedSlot, RegAllocConfig, RegAllocResult, StackSlot};
use crate::codegen::tir::{Func, Instruction, Reg};
use crate::support::slotmap::Key;
use iced_x86::code_asm::registers::{
    r10, r11, r12, r13, r14, r15, r8, r9, rax, rbp, rbx, rcx, rdi, rdx, rsi, rsp,
};
use iced_x86::code_asm::{AsmRegister64, CodeAssembler, CodeLabel};
use std::collections::BTreeSet;

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

pub struct FnMCWriter<'i> {
    asm: CodeAssembler,
    func: &'i Func<X64Inst>,
    ra_cfg: &'i RegAllocConfig,
    ra_res: &'i RegAllocResult,
    /// Total bytes reserved on the stack by the prologue's `sub rsp, N` for
    /// local storage (regalloc spill slots). Always a multiple of 16 so that
    /// stack alignment is preserved at any call site.
    frame_adjust: u32,
    /// Callee-saved registers that need to be saved/restored because they
    /// appear in the ra coloring *or* the scratch set. Sorted, deduplicated,
    /// and pushed in order during the prologue (popped in reverse in the
    /// epilogue).
    saved_callee_regs: Vec<Reg>,
}

impl<'i> FnMCWriter<'i> {
    #[must_use]
    pub fn new(
        func: &'i Func<X64Inst>,
        ra_cfg: &'i RegAllocConfig,
        ra_res: &'i RegAllocResult,
    ) -> Self {
        let raw_frame = ra_res.frame_size;
        let frame_adjust = raw_frame.div_ceil(16) * 16;
        let saved_callee_regs = Self::compute_saved_callee_regs(ra_cfg, ra_res);
        Self {
            asm: CodeAssembler::new(64).expect("iced-x86 supports 64-bit"),
            func,
            ra_cfg,
            ra_res,
            frame_adjust,
            saved_callee_regs,
        }
    }

    fn compute_saved_callee_regs(ra_cfg: &RegAllocConfig, ra_res: &RegAllocResult) -> Vec<Reg> {
        let mut used: BTreeSet<Reg> = BTreeSet::new();
        for (_v, slot) in ra_res.coloring.iter() {
            if let AllocatedSlot::Reg(r) = slot {
                used.insert(*r);
            }
        }
        for s in &ra_cfg.scratch_regs {
            used.insert(*s);
        }
        CALLEE_SAVED
            .iter()
            .filter(|r| used.contains(r))
            // Filter out RBP — we handle it explicitly with push/mov/pop.
            .filter(|&&r| r != RBP)
            .copied()
            .collect()
    }

    /// Offset for a stack slot, as an `[rbp - off]` displacement. Slot 0 sits
    /// at `[rbp - 8]`, slot 1 at `[rbp - 16]`, and so on. Since we set
    /// `rbp <- rsp` *before* `sub rsp, N`, slots live below rbp inside the
    /// frame regardless of `frame_adjust` padding.
    fn slot_offset(slot: StackSlot) -> i32 {
        -((slot as i32 + 1) * 8)
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

    /// Load an operand into a physical register suitable for reading. If the
    /// vreg lives in a register, no MOV is emitted and that register is
    /// returned. If it's spilled, we load from `[rbp - off]` into
    /// `scratch_regs[scratch_idx]`.
    fn load_use(&mut self, vreg: Reg, scratch_idx: usize) -> AsmRegister64 {
        match self.ra_res.coloring[vreg] {
            AllocatedSlot::Reg(r) => to_ice_reg(r),
            AllocatedSlot::Stack(slot) => {
                let s = self.scratch(scratch_idx);
                self.asm
                    .mov(s, rbp + i64::from(Self::slot_offset(slot)))
                    .expect("mov-load from slot");
                s
            }
        }
    }

    /// Prepare a physical register to receive a def. Returns the register we
    /// should emit into. If the vreg is spilled, we return a scratch — the
    /// caller must then call `store_def` to spill back.
    fn prepare_def(&mut self, vreg: Reg, scratch_idx: usize) -> AsmRegister64 {
        match self.ra_res.coloring[vreg] {
            AllocatedSlot::Reg(r) => to_ice_reg(r),
            AllocatedSlot::Stack(_) => self.scratch(scratch_idx),
        }
    }

    /// Store-back for a def. No-op if the vreg lives in a register.
    fn store_def(&mut self, vreg: Reg, scratch_idx: usize) {
        if let AllocatedSlot::Stack(slot) = self.ra_res.coloring[vreg] {
            self.asm
                .mov(rbp + i64::from(Self::slot_offset(slot)), self.scratch(scratch_idx))
                .expect("mov-store to slot");
        }
    }

    /// Emit a two-operand `op` where `dst` is both read and written and `src`
    /// is read.
    fn emit_rr_op<F>(&mut self, dst: Reg, src: Reg, op: F)
    where
        F: FnOnce(&mut CodeAssembler, AsmRegister64, AsmRegister64),
    {
        // Load dst first into scratch 0 (if spilled) — dst is both used and
        // defined, so we need its current value before the op.
        let dst_r = self.load_use(dst, 0);
        let src_r = self.load_use(src, 1);
        op(&mut self.asm, dst_r, src_r);
        // Now `dst_r` holds the updated value. If dst is spilled, `dst_r` IS
        // the scratch, and we spill back.
        self.store_def(dst, 0);
    }

    /// Emit: `push rbp; push <saved callee>*; mov rbp, rsp; sub rsp, N`
    /// Also pads with one extra 8-byte push if needed to keep `rsp % 16 == 0`
    /// at any call site below (at entry `rsp % 16 == 8`, `push rbp` aligns
    /// to 0, then every additional push toggles parity).
    fn emit_prologue(&mut self) {
        self.asm.push(rbp).expect("push rbp");
        for &r in &self.saved_callee_regs {
            self.asm.push(to_ice_reg(r)).expect("push callee-saved");
        }
        // After `push rbp` (1 push) + saved_callee_regs (N pushes), rsp has
        // moved by 8*(1+N). For 16-alignment at the next call we need that
        // total to be ≡ 8 mod 16 (entry was +8). So 8*(1+N) ≡ 8 mod 16,
        // i.e. N must be even. Pad with a dummy `sub rsp, 8` if N is odd.
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

    fn emit_inst(&mut self, inst: &X64Inst, labels: &mut [CodeLabel]) {
        match *inst {
            X64Inst::Mov64rr { dst, src } => {
                // If dst and src share a preg already, no-op.
                if let (AllocatedSlot::Reg(a), AllocatedSlot::Reg(b)) =
                    (self.ra_res.coloring[dst], self.ra_res.coloring[src])
                    && a == b
                {
                    return;
                }
                // If src is spilled -> load to scratch -> store at dst (or use
                // dst reg directly). Keep it simple: load src, then either
                // mov into dst reg or store to dst slot.
                let src_r = self.load_use(src, 1);
                match self.ra_res.coloring[dst] {
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
                let dst_r = self.prepare_def(dst, 0);
                self.asm.mov(dst_r, imm).expect("mov r, imm64");
                self.store_def(dst, 0);
            }
            X64Inst::Mov64rm { dst, src } => {
                let base_r = self.load_use(src.base, 1);
                let dst_r = self.prepare_def(dst, 0);
                if let Some(idx) = src.index {
                    let idx_r = self.load_use(idx, 2);
                    self.asm
                        .mov(dst_r, base_r + idx_r * i32::from(src.scale) + src.disp)
                        .expect("mov r, [base+idx*s+disp]");
                } else {
                    self.asm
                        .mov(dst_r, base_r + i64::from(src.disp))
                        .expect("mov r, [base+disp]");
                }
                self.store_def(dst, 0);
            }
            X64Inst::Mov64mr { dst, src } => {
                let base_r = self.load_use(dst.base, 0);
                let src_r = self.load_use(src, 1);
                if let Some(idx) = dst.index {
                    let idx_r = self.load_use(idx, 2);
                    self.asm
                        .mov(base_r + idx_r * i32::from(dst.scale) + dst.disp, src_r)
                        .expect("mov [base+idx*s+disp], r");
                } else {
                    self.asm
                        .mov(base_r + i64::from(dst.disp), src_r)
                        .expect("mov [base+disp], r");
                }
            }
            X64Inst::Add64rr { dst, src } => self.emit_rr_op(dst, src, |a, d, s| {
                a.add(d, s).expect("add rr");
            }),
            X64Inst::Sub64rr { dst, src } => self.emit_rr_op(dst, src, |a, d, s| {
                a.sub(d, s).expect("sub rr");
            }),
            X64Inst::Imul64rr { dst, src } => self.emit_rr_op(dst, src, |a, d, s| {
                a.imul_2(d, s).expect("imul rr");
            }),
            X64Inst::Add64ri32 { dst, imm } => {
                let dst_r = self.load_use(dst, 0);
                self.asm.add(dst_r, imm).expect("add r, imm32");
                self.store_def(dst, 0);
            }
            X64Inst::Sub64ri32 { dst, imm } => {
                let dst_r = self.load_use(dst, 0);
                self.asm.sub(dst_r, imm).expect("sub r, imm32");
                self.store_def(dst, 0);
            }
            X64Inst::Cmp64rr { lhs, rhs } => {
                let lhs_r = self.load_use(lhs, 0);
                let rhs_r = self.load_use(rhs, 1);
                self.asm.cmp(lhs_r, rhs_r).expect("cmp rr");
            }
            X64Inst::Cmp64ri32 { lhs, imm } => {
                let lhs_r = self.load_use(lhs, 0);
                self.asm.cmp(lhs_r, imm).expect("cmp r, imm32");
            }
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
            X64Inst::RawRet => self.emit_epilogue(),
        }
    }

    pub fn emit_fn(&mut self) -> Vec<u8> {
        self.emit_prologue();

        let mut labels: Vec<CodeLabel> = (0..self.func.blocks_count())
            .map(|_| self.asm.create_label())
            .collect();

        for (block, block_data) in self.func.blocks_iter() {
            self.asm
                .set_label(&mut labels[block.index()])
                .expect("set_label");
            for instr in block_data.iter() {
                match instr {
                    Instruction::Target(x64_inst) => self.emit_inst(x64_inst, &mut labels),
                    Instruction::Pseudo(p) => {
                        panic!(
                            "MC emission received un-lowered pseudo: {p:?}. \
                             Run the pseudo_cleanup pass before emission."
                        )
                    }
                }
            }
        }
        self.asm.assemble(0).expect("assemble")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::analysis::cfg::CFG;
    use crate::codegen::isa::x64::passes::{abi_lower, pseudo_cleanup};
    use crate::codegen::isa::x64::sysv::SysVAmd64;
    use crate::codegen::regalloc::{RegAlloc, RegAllocConfig};
    use crate::codegen::tir::{Func, PseudoInstruction};
    use std::collections::HashMap;

    fn test_ra_config(reg_bind: HashMap<Reg, Reg>) -> RegAllocConfig {
        // Mirrors the default pipeline config — 9 caller-saved allocatable,
        // 3 callee-saved scratches (so memory-operand paths that consume up
        // to 3 scratches at once can't panic).
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
        let abi = abi_lower::lower(&mut func, SysVAmd64);
        let cfg = CFG::compute(&func).unwrap();
        let cfg_cfg = test_ra_config(abi.reg_bind);
        let mut ra = RegAlloc::new(&func, &cfg, &cfg_cfg);
        let res = ra.run();
        pseudo_cleanup::run(&mut func, &res);
        let mut w = FnMCWriter::new(&func, &cfg_cfg, &res);
        let bytes = w.emit_fn();
        assert!(bytes.len() >= 4);
        assert_eq!(bytes[0], 0x55); // push rbp
        assert_eq!(*bytes.last().unwrap(), 0xC3); // ret
    }

    #[test]
    fn scratch_index_out_of_range_panics_with_clear_message() {
        // Construct a bare FnMCWriter by hand with a config that provides no
        // scratches. This would never happen via the pipeline but the guard
        // in `scratch()` is the project's single documented point of contact
        // for "emitter requested a scratch the allocator didn't reserve".
        use crate::codegen::regalloc::RegAllocResult;
        use crate::support::slotmap::SecondaryMap;

        let func = Func::<X64Inst>::new("t".to_string());
        let empty_cfg = RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RAX],
            scratch_regs: vec![], // intentionally empty
            reg_bind: HashMap::new(),
        };
        let empty_ra = RegAllocResult {
            coloring: SecondaryMap::new(0),
            frame_layout: SecondaryMap::new(0),
            frame_size: 0,
        };
        let w = FnMCWriter::new(&func, &empty_cfg, &empty_ra);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.scratch(0)));
        assert!(caught.is_err());
    }
}
