use std::fmt::Display;

use crate::codegen::tir::{Block, Inst, Reg};

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

#[derive(Clone, Copy)]
pub enum X64Inst {
    Ret,
    Jmp{dst: Block},
    CondJmp{cond: Cond, taken: Block, not_taken: Block},
    Mov64rm{dst: Reg, src: Mem},
    Mov64mr{dst: Mem, src: Reg},
    Mov64rr{dst: Reg, src: Reg},
    Mov64ri64{dst: Reg, src: i64},
    Mov64mi64{dst: Mem, src: i64},
    CMP64rr{lhs: Reg, rhs: Reg},
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
            X64Inst::Jmp {..} => true,
            X64Inst::CondJmp {..} => true,
            _=> false
        }
    }
}

fn reg_to_string(reg: &Reg) -> String {
    if reg.is_virtual() {
        format!("v{}", reg.id())
    } else {
        todo!()
    }
}

impl Display for X64Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            X64Inst::Ret => write!(f, "ret"),
            X64Inst::Jmp { dst } => write!(f, "jmp {dst}"),
            X64Inst::CondJmp { cond, taken, not_taken } => todo!(),
            X64Inst::Mov64rm { dst, src } => todo!(),
            X64Inst::Mov64mr { dst, src } => todo!(),
            X64Inst::Mov64rr { dst, src } => write!(f, "mov {} {}", reg_to_string(dst), reg_to_string(src)),
            X64Inst::Mov64ri64 { dst, src } => todo!(),
            X64Inst::Mov64mi64 { dst, src } => todo!(),
            X64Inst::CMP64rr { lhs, rhs } => todo!(),
        }
    }
}