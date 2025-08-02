use std::fmt::Display;

use crate::codegen::{
    isa::x64::regs::*,
    tir::{self, Block, Inst, Reg},
};

use smallvec::{SmallVec, smallvec};

#[derive(Clone, Copy)]
pub enum Cond {
    Z,
    NZ,
    B,
    NB,
    BE,
    NBE,
    L,
    LE,
    NL,
    NLE,
    O,
    NO,
    P,
    NP,
    S,
    NS,
}

#[derive(Copy, Clone)]
pub struct Mem {
    pub reg: Reg,
    index: Option<Reg>,
    scale: u8,
    disp: i32,
}

impl Mem {
    pub fn get_uses(&self) -> SmallVec<[Reg; 2]> {
        if let Some(idx) = self.index {
            smallvec![self.reg, idx]
        } else {
            smallvec![self.reg]
        }
    }
}

#[derive(Clone, Copy)]
pub enum X64Inst {
    Ret,
    Jmp {
        dst: Block,
    },
    CondJmp {
        cond: Cond,
        taken: Block,
        not_taken: Block,
    },
    Mov64rm {
        dst: Reg,
        src: Mem,
    },
    Mov64mr {
        dst: Mem,
        src: Reg,
    },
    Mov64rr {
        dst: Reg,
        src: Reg,
    },
    Mov64ri64 {
        dst: Reg,
        src: i64,
    },
    Mov64mi64 {
        dst: Mem,
        src: i64,
    },
    CMP64rr {
        lhs: Reg,
        rhs: Reg,
    },
}

impl Inst for X64Inst {
    fn is_ret(&self) -> bool {
        match self {
            X64Inst::Ret => true,
            _ => false,
        }
    }

    fn is_branch(&self) -> bool {
        match self {
            X64Inst::Jmp { .. } => true,
            X64Inst::CondJmp { .. } => true,
            _ => false,
        }
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]> {
        match self {
            X64Inst::Ret => smallvec![],
            X64Inst::Jmp { .. } => smallvec![],
            X64Inst::CondJmp { .. } => smallvec![],
            X64Inst::Mov64rm { dst, src } => src.get_uses(),
            X64Inst::Mov64mr { dst, src } => {
                let mut uses: SmallVec<[Reg; 2]> = dst.get_uses();
                uses.push(*src);
                uses
            }
            X64Inst::Mov64rr { dst, src } => smallvec![*src],
            X64Inst::Mov64ri64 { dst, src } => smallvec![],
            X64Inst::Mov64mi64 { dst, src } => dst.get_uses(),
            X64Inst::CMP64rr { lhs, rhs } => smallvec![*lhs, *rhs],
        }
    }
    fn get_defs(&self) -> SmallVec<[Reg; 1]> {
        match self {
            X64Inst::Ret => smallvec![],
            X64Inst::Jmp { .. } => smallvec![],
            X64Inst::CondJmp { .. } => smallvec![],
            X64Inst::Mov64rm { dst, src } => smallvec![*dst],
            X64Inst::Mov64mr { dst, src } => smallvec![],
            X64Inst::Mov64rr { dst, src } => smallvec![*dst],
            X64Inst::Mov64ri64 { dst, src } => smallvec![*dst],
            X64Inst::Mov64mi64 { dst, src } => todo!(),
            X64Inst::CMP64rr { lhs, rhs } => todo!(),
        }
    }

    fn get_branch_targets(&self) -> SmallVec<[Block; 2]> {
        match self {
            X64Inst::Jmp { dst } => smallvec![*dst],
            X64Inst::CondJmp {
                taken, not_taken, ..
            } => smallvec![*taken, *not_taken],
            _ => smallvec![],
        }
    }

    fn preg_name(reg: crate::codegen::tir::Reg) -> String {
        match reg {
            RAX => "rax".to_string(),
            RBX => "rbx".to_string(),
            RCX => "rcx".to_string(),
            RDX => "rdx".to_string(),
            RSI => "rsi".to_string(),
            RDI => "rdi".to_string(),
            RSP => "rsp".to_string(),
            RBP => "rbp".to_string(),
            R8 => "r8".to_string(),
            R9 => "r9".to_string(),
            R10 => "r10".to_string(),
            R11 => "r11".to_string(),
            R12 => "r12".to_string(),
            R13 => "r13".to_string(),
            R14 => "r14".to_string(),
            R15 => "r15".to_string(),
            _ => unreachable!(),
        }
    }

    fn replace(&self, old: Reg, new: Reg) -> Self {
        fn replace_reg(cur: Reg, old: Reg, new: Reg) -> Reg {
            if old == cur { new } else { cur }
        }

        fn replace_mem(mem: Mem, old: Reg, new: Reg) -> Mem {
            Mem {
                reg: replace_reg(mem.reg, old, new),
                index: if let Some(idx) = mem.index {
                    Some(replace_reg(idx, old, new))
                } else {
                    None
                },
                scale: mem.scale,
                disp: mem.disp,
            }
        }

        match *self {
            X64Inst::Ret => *self,
            X64Inst::Jmp { .. } => *self,
            X64Inst::CondJmp { .. } => *self,
            X64Inst::Mov64rm { dst, src } => X64Inst::Mov64rm {
                dst: replace_reg(dst, old, new),
                src: replace_mem(src, old, new),
            },
            X64Inst::Mov64mr { dst, src } => X64Inst::Mov64mr {
                dst: replace_mem(dst, old, new),
                src: replace_reg(src, old, new),
            },
            X64Inst::Mov64rr { dst, src } => X64Inst::Mov64rr {
                dst: replace_reg(dst, old, new),
                src: replace_reg(src, old, new),
            },
            X64Inst::Mov64ri64 { dst, src } => X64Inst::Mov64ri64 {
                dst: replace_reg(dst, old, new),
                src,
            },
            X64Inst::Mov64mi64 { dst, src } => X64Inst::Mov64mi64 {
                dst: replace_mem(dst, old, new),
                src,
            },
            X64Inst::CMP64rr { lhs, rhs } => X64Inst::CMP64rr {
                lhs: replace_reg(lhs, old, new),
                rhs: replace_reg(rhs, old, new),
            },
        }
    }

    fn preg_count() -> u32 {
        REGISTERS_COUNT
    }
}

fn reg_name(reg: Reg) -> String {
    tir::reg_name::<X64Inst>(reg)
}

impl Display for X64Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            X64Inst::Ret => write!(f, "ret"),
            X64Inst::Jmp { dst } => write!(f, "jmp {dst}"),
            X64Inst::CondJmp {
                cond,
                taken,
                not_taken,
            } => todo!(),
            X64Inst::Mov64rm { dst, src } => todo!(),
            X64Inst::Mov64mr { dst, src } => todo!(),
            X64Inst::Mov64rr { dst, src } => write!(f, "mov {} {}", reg_name(*dst), reg_name(*src)),
            X64Inst::Mov64ri64 { dst, src } => todo!(),
            X64Inst::Mov64mi64 { dst, src } => todo!(),
            X64Inst::CMP64rr { lhs, rhs } => todo!(),
        }
    }
}
