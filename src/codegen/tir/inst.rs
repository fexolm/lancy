use smallvec::{smallvec, SmallVec};
use std::fmt::{Display, Formatter};

use super::Reg;
use crate::codegen::tir::Block;

pub trait Inst: Sized + Copy + Display {
    fn is_branch(&self) -> bool;
    fn is_ret(&self) -> bool;

    fn is_term(&self) -> bool {
        self.is_branch() || self.is_ret()
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]>;
    fn get_defs(&self) -> SmallVec<[Reg; 1]>;

    fn get_branch_targets(&self) -> SmallVec<[Block; 2]>;
}

#[derive(Copy, Clone, Debug)]
pub enum PseudoInstruction {
    Arg { dst: Reg }
}

impl Display for PseudoInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PseudoInstruction::Arg { dst } => write!(f, "arg {}", reg_name(*dst)),
        }
    }
}

impl Inst for PseudoInstruction {
    fn is_branch(&self) -> bool {
        false
    }

    fn is_ret(&self) -> bool {
        false
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]> {
        match self {
            PseudoInstruction::Arg { .. } => smallvec![],
        }
    }

    fn get_defs(&self) -> SmallVec<[Reg; 1]> {
        match self {
            PseudoInstruction::Arg { dst } => smallvec![*dst],
        }
    }

    fn get_branch_targets(&self) -> SmallVec<[Block; 2]> {
        smallvec![]
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Instruction<I: Inst> {
    Target(I),
    Pseudo(PseudoInstruction),
}

impl<I: Inst> Display for Instruction<I> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Pseudo(inst) => write!(f, "{}", inst),
            Instruction::Target(inst) => write!(f, "{}", inst),
        }
    }
}

impl<I: Inst> Inst for Instruction<I> {
    fn is_branch(&self) -> bool {
        match self {
            Instruction::Target(inst) => inst.is_branch(),
            Instruction::Pseudo(inst) => inst.is_branch(),
        }
    }

    fn is_ret(&self) -> bool {
        match self {
            Instruction::Target(inst) => inst.is_ret(),
            Instruction::Pseudo(inst) => inst.is_ret(),
        }
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]> {
        match self {
            Instruction::Target(inst) => inst.get_uses(),
            Instruction::Pseudo(inst) => inst.get_uses(),
        }
    }

    fn get_defs(&self) -> SmallVec<[Reg; 1]> {
        match self {
            Instruction::Target(inst) => inst.get_defs(),
            Instruction::Pseudo(inst) => inst.get_defs(),
        }
    }

    fn get_branch_targets(&self) -> SmallVec<[Block; 2]> {
        match self {
            Instruction::Target(inst) => inst.get_branch_targets(),
            Instruction::Pseudo(inst) => inst.get_branch_targets(),
        }
    }
}

pub fn reg_name(reg: Reg) -> String {
    format!("v{}", reg)
}
