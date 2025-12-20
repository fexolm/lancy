use crate::codegen;
use crate::codegen::isa::x64::inst::X64Inst;
use crate::codegen::isa::x64::regs::*;
use crate::codegen::regalloc::{AllocatedSlot, RegAlloc, RegAllocConfig, RegAllocResult};
use crate::codegen::tir::{Func, Inst, Instruction, PseudoInstruction, Reg};
use iced_x86::{Register};
use iced_x86::code_asm::{CodeAssembler, AsmRegister64};
use iced_x86::code_asm::registers::{*};
use crate::support::slotmap::Key;

fn reg(r: Reg) -> Register {
    match r {
        RAX => Register::RAX,
        RBX => Register::RBX,
        RCX => Register::RCX,
        RDX => Register::RDX,
        RSI => Register::RSI,
        RDI => Register::RDI,
        RSP => Register::RSP,
        RBP => Register::RBP,
        R8 => Register::R8,
        R9 => Register::R9,
        R10 => Register::R10,
        R11 => Register::R11,
        R12 => Register::R12,
        R13 => Register::R13,
        R14 => Register::R14,
        R15 => Register::R15,
        _ => todo!(),
    }
}

fn asm_reg(r: Reg) -> AsmRegister64 {
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

pub fn emit(func: &Func<X64Inst>, reg_alloc_config: RegAllocConfig, reg_alloc_result: RegAllocResult) -> Vec<u8> {
    let mut assembler = CodeAssembler::new(64).unwrap();
    let coloring = reg_alloc_result.coloring;
    let frame_layout = reg_alloc_result.frame_layout;
    let scratch_register = reg_alloc_config.scratch_regs;
    let mut labels = Vec::new();
    for (block, block_data) in func.blocks_iter() {
        labels.push(assembler.create_label());
    }
    for (block, block_data) in func.blocks_iter() {
        assembler.set_label(&mut labels[block.index()]);
        for instr in block_data.iter() {
            match instr {
                Instruction::Target(x64_inst) => match x64_inst {
                    X64Inst::Mov64rr { dst, src } => {
                        let lhs = match coloring[*dst] {
                            AllocatedSlot::Reg(r) => asm_reg(r),
                            AllocatedSlot::Stack(slot) => {
                                assembler.mov(asm_reg(scratch_register[0]), frame_layout[slot] as u64).unwrap();
                                asm_reg(scratch_register[0])
                            }
                        };
                        let rhs = match coloring[*src] {
                            AllocatedSlot::Reg(r) => asm_reg(r),
                            AllocatedSlot::Stack(slot) => {
                                assembler.mov(asm_reg(scratch_register[0]), frame_layout[slot] as u64).unwrap();
                                asm_reg(scratch_register[0])
                            }
                        };
                        assembler.mov(lhs, rhs);
                    }
                    X64Inst::Ret { src } => { assembler.ret().unwrap() }
                    X64Inst::Jmp { dst } => { assembler.jmp(labels[dst.index()]).unwrap() }
                    _ => todo!()
                },
                Instruction::Pseudo(psedo) => todo!()
            };
        }
    }
    assembler.assemble(0).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::codegen::analysis::cfg::CFG;
    use crate::codegen::isa::x64::inst::X64Inst;
    use crate::codegen::isa::x64::regs::*;
    use crate::codegen::regalloc::{RegAlloc, RegAllocConfig};
    use crate::codegen::tir::{BlockData, Func, PseudoInstruction};
    use std::collections::{HashMap, LinkedList};
    use crate::codegen::x64::lower::{emit, reg};

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
            // block_data.push_pseudo_inst(PseudoInstruction::Arg { dst: v0 });
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
        let config_copy = reg_alloc_config.clone();
        let mut regalloc = RegAlloc::new(&func, &cfg, reg_alloc_config);
        let regalloc_result = regalloc.run();

        let assembler = emit(&func, config_copy, regalloc_result);
        for b in &assembler {
            print!("{:02X} ", b);
        }
        assert_eq!(true, true);
    }
}
