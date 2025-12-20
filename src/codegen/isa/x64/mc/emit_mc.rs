use crate::codegen::isa::x64::inst::X64Inst;
use crate::codegen::isa::x64::regs::*;
use crate::codegen::regalloc::{AllocatedSlot, RegAllocConfig, RegAllocResult};
use crate::codegen::tir::{Func, Inst, Instruction, Reg};
use crate::support::slotmap::Key;
use iced_x86::code_asm::registers::*;
use iced_x86::code_asm::{AsmRegister64, CodeAssembler};

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
        _ => todo!(),
    }
}
pub struct FnMCWriter<'i> {
    asm: CodeAssembler,
    func: &'i Func<X64Inst>,
    ra_cfg: &'i RegAllocConfig,
    ra_res: RegAllocResult,
}

impl<'i> FnMCWriter<'i> {
    pub fn new(func: &'i Func<X64Inst>, ra_cfg: &'i RegAllocConfig, ra_res: RegAllocResult) -> Self {
        Self {
            asm: CodeAssembler::new(64).unwrap(),
            func,
            ra_cfg,
            ra_res,
        }
    }

    fn emit_vreg_load(&mut self, vreg: Reg) -> AsmRegister64 {
        match self.ra_res.coloring[vreg] {
            AllocatedSlot::Reg(r) => to_ice_reg(r),
            AllocatedSlot::Stack(slot) => {
                self.asm
                    .mov(
                        to_ice_reg(self.ra_cfg.scratch_regs[0]),
                        rsp - self.ra_res.frame_layout[slot] as u64,
                    )
                    .unwrap();
                to_ice_reg(self.ra_cfg.scratch_regs[0])
            }
        }
    }

    fn choose_store_reg(&self, vreg: Reg) -> AsmRegister64 {
        match self.ra_res.coloring[vreg] {
            AllocatedSlot::Reg(r) => to_ice_reg(r),
            AllocatedSlot::Stack(slot) => {
                to_ice_reg(self.ra_cfg.scratch_regs[0])
            }
        }
    }

    fn emit_post_vreg_store(&mut self, vreg: Reg) {
        match self.ra_res.coloring[vreg] {
            AllocatedSlot::Reg(r) => {}
            AllocatedSlot::Stack(slot) => {
                self.asm
                    .mov(
                        rsp - self.ra_res.frame_layout[slot],
                        to_ice_reg(self.ra_cfg.scratch_regs[0]),
                    )
                    .unwrap();
            }
        }
    }

    fn emit_prologue(&mut self) {
        self.asm.add(rsp, self.ra_res.frame_size as i32);
    }

    fn emit_epilogue(&mut self) {
        self.asm.sub(rsp, self.ra_res.frame_size as i32);
    }

    pub fn emit_fn(&mut self) -> Vec<u8> {
        self.emit_prologue();

        let mut labels = Vec::new();

        for (block, block_data) in self.func.blocks_iter() {
            labels.push(self.asm.create_label());
        }

        for (block, block_data) in self.func.blocks_iter() {
            self.asm.set_label(&mut labels[block.index()]);
            for instr in block_data.iter() {
                match instr {
                    Instruction::Target(x64_inst) => match x64_inst {
                        X64Inst::Mov64rr { dst, src } => {
                            let src_preg = self.emit_vreg_load(*src);
                            let dst_preg = self.choose_store_reg(*dst);
                            self.asm.mov(dst_preg, src_preg);
                            self.emit_post_vreg_store(*dst);
                        }
                        X64Inst::Ret { src } => { /* will be emitted in the end */ }
                        X64Inst::Jmp { dst } => self.asm.jmp(labels[dst.index()]).unwrap(),
                        _ => todo!(),
                    },
                    Instruction::Pseudo(psedo) => { /* ignore */ }
                };
            }
        }
        self.emit_epilogue();
        self.asm.ret().unwrap();
        self.asm.assemble(0).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::codegen::analysis::cfg::CFG;
    use crate::codegen::isa::x64::inst::X64Inst;
    use crate::codegen::isa::x64::mc::emit_mc::FnMCWriter;
    use crate::codegen::isa::x64::regs::*;
    use crate::codegen::regalloc::{RegAlloc, RegAllocConfig};
    use crate::codegen::tir::{Func, PseudoInstruction};
    use std::collections::HashMap;

    #[test]
    fn simple_test() {
        // foo:
        // @0
        //     mov v1 v0
        //     jmp @1
        // @1
        //     mov v2 v1
        //     jmp @2
        // @2
        //     mov v3 v2
        //     ret v3
        let mut func = Func::<X64Inst>::new("foo".to_string());

        let b0 = func.add_empty_block();
        let b1 = func.add_empty_block();
        let b2 = func.add_empty_block();

        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut reg_bind = HashMap::new();
        reg_bind.insert(v0, RAX);
        reg_bind.insert(v3, RAX);

        {
            let block_data = func.get_block_data_mut(b0);
            block_data.push_pseudo_inst(PseudoInstruction::Arg { dst: v0 });
            block_data.push_target_inst(X64Inst::Mov64rr { dst: v1, src: v0 });
            block_data.push_target_inst(X64Inst::Jmp { dst: b1 });
        }

        {
            let block_data = func.get_block_data_mut(b1);
            block_data.push_target_inst(X64Inst::Mov64rr { dst: v2, src: v1 });
            block_data.push_target_inst(X64Inst::Jmp { dst: b2 });
        }

        {
            let block_data = func.get_block_data_mut(b2);
            block_data.push_target_inst(X64Inst::Mov64rr { dst: v3, src: v2 });
            block_data.push_target_inst(X64Inst::Ret { src: v3 });
        }

        println!("{func}");

        let cfg = CFG::compute(&func).unwrap();
        let mut allocatable_regs = vec![RAX, RBX, RCX, RDX];
        let mut scratch_regs = vec![R12, R13];
        let reg_alloc_config = RegAllocConfig {
            preg_count: 32,
            allocatable_regs,
            scratch_regs,
            reg_bind,
        };
        
        let mut regalloc = RegAlloc::new(&func, &cfg, &reg_alloc_config);
        let regalloc_result = regalloc.run();

        let mut writer = FnMCWriter::new(&func, &reg_alloc_config, regalloc_result);
        for b in &writer.emit_fn() {
            print!("{:02X} ", b);
        }
        assert_eq!(true, true);
    }
}
