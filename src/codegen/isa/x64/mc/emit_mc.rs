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
    r10, r11, r12, r13, r14, r15, r8, r9, rax, rbp, rbx, rcx, rdi, rdx, rsi, rsp,
};
use iced_x86::code_asm::{AsmRegister64, CodeAssembler, CodeLabel};
use std::collections::BTreeSet;

/// Maximum simultaneous scratch registers this instruction can demand in the
/// worst case (all operand vregs spilled). Stays in sync with `emit_inst`.
fn scratch_demand_of(inst: &X64Inst) -> usize {
    match inst {
        X64Inst::Mov64rm { src, .. } | X64Inst::Mov64mr { dst: src, .. }
            if src.index.is_some() =>
        {
            3
        }
        X64Inst::Mov64rm { .. } | X64Inst::Mov64mr { .. } => 2,
        X64Inst::Add64rr { .. }
        | X64Inst::Sub64rr { .. }
        | X64Inst::Imul64rr { .. }
        | X64Inst::Cmp64rr { .. }
        | X64Inst::Mov64rr { .. } => 2,
        X64Inst::Mov64ri { .. }
        | X64Inst::Add64ri32 { .. }
        | X64Inst::Sub64ri32 { .. }
        | X64Inst::Cmp64ri32 { .. } => 1,
        X64Inst::Jmp { .. } | X64Inst::CondJmp { .. } | X64Inst::RawRet => 0,
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

pub struct FnMCWriter<'i> {
    asm: CodeAssembler,
    func: &'i Func<X64Inst>,
    ra_cfg: &'i RegAllocConfig,
    ra_res: &'i RegAllocResult,
    layout: BlockLayout,
    frame_adjust: u32,
    saved_callee_regs: Vec<Reg>,
    splits_by_point: HashMap<ProgramPoint, Vec<SplitMove>>,
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
        }
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
    /// given use point. If the vreg lives in a register there, no MOV is
    /// emitted and that register is returned. If it's on the stack at that
    /// point, load into `scratch_regs[scratch_idx]`.
    fn load_use(&mut self, vreg: Reg, pt: ProgramPoint, scratch_idx: usize) -> AsmRegister64 {
        match self.slot_of(vreg, pt) {
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

    fn prepare_def(&mut self, vreg: Reg, pt: ProgramPoint, scratch_idx: usize) -> AsmRegister64 {
        match self.slot_of(vreg, pt) {
            AllocatedSlot::Reg(r) => to_ice_reg(r),
            AllocatedSlot::Stack(_) => self.scratch(scratch_idx),
        }
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
            X64Inst::Cmp64rr { lhs, rhs } => {
                let lhs_r = self.load_use(lhs, use_pt, 0);
                let rhs_r = self.load_use(rhs, use_pt, 1);
                self.asm.cmp(lhs_r, rhs_r).expect("cmp rr");
            }
            X64Inst::Cmp64ri32 { lhs, imm } => {
                let lhs_r = self.load_use(lhs, use_pt, 0);
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
        }
    }

    pub fn emit_fn(&mut self) -> Vec<u8> {
        self.check_scratch_budget();
        self.emit_prologue();

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
        self.asm.assemble(0).expect("assemble")
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
}
