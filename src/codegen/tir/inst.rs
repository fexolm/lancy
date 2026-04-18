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

/// Target-neutral pseudo instructions. Closed set.
///
/// * `Arg` / `Return` — ABI-neutral argument/return. Lowered by the ABI pass.
/// * `Copy` — typed value move. Coalescing candidate; may survive regalloc
///   and be turned into a target MOV (or elided) by the cleanup pass.
#[derive(Copy, Clone, Debug)]
pub enum PseudoInstruction {
    Arg { dst: Reg, idx: u32 },
    Copy { dst: Reg, src: Reg },
    Return { src: Reg },
}

impl Display for PseudoInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PseudoInstruction::Arg { dst, idx } => {
                write!(f, "{} = arg {idx}", reg_name(*dst))
            }
            PseudoInstruction::Copy { dst, src } => {
                write!(f, "{} = copy {}", reg_name(*dst), reg_name(*src))
            }
            PseudoInstruction::Return { src } => write!(f, "return {}", reg_name(*src)),
        }
    }
}

impl Inst for PseudoInstruction {
    fn is_branch(&self) -> bool {
        false
    }

    fn is_ret(&self) -> bool {
        matches!(self, PseudoInstruction::Return { .. })
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]> {
        match self {
            PseudoInstruction::Arg { .. } => smallvec![],
            PseudoInstruction::Copy { src, .. } => smallvec![*src],
            PseudoInstruction::Return { src } => smallvec![*src],
        }
    }

    fn get_defs(&self) -> SmallVec<[Reg; 1]> {
        match self {
            PseudoInstruction::Arg { dst, .. } | PseudoInstruction::Copy { dst, .. } => {
                smallvec![*dst]
            }
            PseudoInstruction::Return { .. } => smallvec![],
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
            Instruction::Pseudo(inst) => write!(f, "{inst}"),
            Instruction::Target(inst) => write!(f, "{inst}"),
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

#[must_use]
pub fn reg_name(reg: Reg) -> String {
    format!("v{reg}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_arg_defs_dst_uses_nothing() {
        let p = PseudoInstruction::Arg { dst: 7, idx: 0 };
        assert_eq!(p.get_defs().as_slice(), &[7]);
        assert!(p.get_uses().is_empty());
        assert!(!p.is_term());
    }

    #[test]
    fn pseudo_copy_has_src_use_and_dst_def() {
        let p = PseudoInstruction::Copy { dst: 3, src: 2 };
        assert_eq!(p.get_defs().as_slice(), &[3]);
        assert_eq!(p.get_uses().as_slice(), &[2]);
    }

    #[test]
    fn pseudo_return_is_terminator_and_uses_src() {
        let p = PseudoInstruction::Return { src: 9 };
        assert!(p.is_term());
        assert!(p.is_ret());
        assert_eq!(p.get_uses().as_slice(), &[9]);
        assert!(p.get_defs().is_empty());
    }
}
