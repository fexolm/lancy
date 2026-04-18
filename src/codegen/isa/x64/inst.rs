use std::fmt::Display;

use crate::codegen::tir::{self, Block, Inst, Reg};

use smallvec::{smallvec, SmallVec};

/// x86-64 condition codes. Name matches the suffix used with J/SET/CMOV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    Z,
    NZ,
    L,
    LE,
    G,
    GE,
    B,
    BE,
    A,
    AE,
}

impl Display for Cond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Cond::Z => "z",
            Cond::NZ => "nz",
            Cond::L => "l",
            Cond::LE => "le",
            Cond::G => "g",
            Cond::GE => "ge",
            Cond::B => "b",
            Cond::BE => "be",
            Cond::A => "a",
            Cond::AE => "ae",
        };
        f.write_str(s)
    }
}

/// `base + (index * scale) + disp`. Used by Mov64rm / Mov64mr / Mov64mi32.
#[derive(Copy, Clone, Debug)]
pub struct Mem {
    pub base: Reg,
    pub index: Option<Reg>,
    pub scale: u8,
    pub disp: i32,
}

impl Mem {
    #[must_use]
    pub fn base(base: Reg) -> Self {
        Self { base, index: None, scale: 1, disp: 0 }
    }

    #[must_use]
    pub fn base_disp(base: Reg, disp: i32) -> Self {
        Self { base, index: None, scale: 1, disp }
    }

    #[must_use]
    pub fn get_uses(&self) -> SmallVec<[Reg; 2]> {
        if let Some(idx) = self.index {
            smallvec![self.base, idx]
        } else {
            smallvec![self.base]
        }
    }
}

impl Display for Mem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}", tir::reg_name(self.base))?;
        if let Some(idx) = self.index {
            write!(f, "+{}*{}", tir::reg_name(idx), self.scale)?;
        }
        match self.disp.cmp(&0) {
            std::cmp::Ordering::Greater => write!(f, "+{}", self.disp)?,
            std::cmp::Ordering::Less => write!(f, "{}", self.disp)?,
            std::cmp::Ordering::Equal => {}
        }
        f.write_str("]")
    }
}

/// x86-64 target instructions. Two-operand form for arithmetic: the first
/// register operand is both read and written. The frontend / builder is
/// responsible for emitting a `Copy` before each arithmetic op that needs
/// three-operand semantics.
#[derive(Clone, Copy, Debug)]
pub enum X64Inst {
    // Moves
    Mov64rr { dst: Reg, src: Reg },
    Mov64ri { dst: Reg, imm: i64 },
    Mov64rm { dst: Reg, src: Mem },
    Mov64mr { dst: Mem, src: Reg },

    // Arithmetic — `dst = dst OP src`.
    Add64rr { dst: Reg, src: Reg },
    Sub64rr { dst: Reg, src: Reg },
    Imul64rr { dst: Reg, src: Reg },
    Add64ri32 { dst: Reg, imm: i32 },
    Sub64ri32 { dst: Reg, imm: i32 },

    // Compare
    Cmp64rr { lhs: Reg, rhs: Reg },
    Cmp64ri32 { lhs: Reg, imm: i32 },

    // Control flow
    Jmp { dst: Block },
    CondJmp { cond: Cond, taken: Block, not_taken: Block },
    // Raw RET — assumes ABI return register is already set and the frame has
    // been torn down. Emitted by the prologue/epilogue pass.
    RawRet,
}

impl Inst for X64Inst {
    fn is_branch(&self) -> bool {
        matches!(self, X64Inst::Jmp { .. } | X64Inst::CondJmp { .. })
    }

    fn is_ret(&self) -> bool {
        matches!(self, X64Inst::RawRet)
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]> {
        match self {
            X64Inst::Mov64rr { src, .. } => smallvec![*src],
            X64Inst::Mov64ri { .. } | X64Inst::RawRet => smallvec![],
            X64Inst::Mov64rm { src, .. } => src.get_uses(),
            X64Inst::Mov64mr { dst, src } => {
                let mut uses: SmallVec<[Reg; 2]> = dst.get_uses();
                uses.push(*src);
                uses
            }
            X64Inst::Add64rr { dst, src }
            | X64Inst::Sub64rr { dst, src }
            | X64Inst::Imul64rr { dst, src } => smallvec![*dst, *src],
            X64Inst::Add64ri32 { dst, .. } | X64Inst::Sub64ri32 { dst, .. } => smallvec![*dst],
            X64Inst::Cmp64rr { lhs, rhs } => smallvec![*lhs, *rhs],
            X64Inst::Cmp64ri32 { lhs, .. } => smallvec![*lhs],
            X64Inst::Jmp { .. } | X64Inst::CondJmp { .. } => smallvec![],
        }
    }

    fn get_defs(&self) -> SmallVec<[Reg; 1]> {
        match self {
            X64Inst::Mov64rr { dst, .. }
            | X64Inst::Mov64ri { dst, .. }
            | X64Inst::Mov64rm { dst, .. }
            | X64Inst::Add64rr { dst, .. }
            | X64Inst::Sub64rr { dst, .. }
            | X64Inst::Imul64rr { dst, .. }
            | X64Inst::Add64ri32 { dst, .. }
            | X64Inst::Sub64ri32 { dst, .. } => smallvec![*dst],
            X64Inst::Mov64mr { .. }
            | X64Inst::Cmp64rr { .. }
            | X64Inst::Cmp64ri32 { .. }
            | X64Inst::Jmp { .. }
            | X64Inst::CondJmp { .. }
            | X64Inst::RawRet => smallvec![],
        }
    }

    fn get_branch_targets(&self) -> SmallVec<[Block; 2]> {
        match self {
            X64Inst::Jmp { dst } => smallvec![*dst],
            X64Inst::CondJmp { taken, not_taken, .. } => smallvec![*taken, *not_taken],
            _ => smallvec![],
        }
    }
}

fn reg_name(reg: Reg) -> String {
    tir::reg_name(reg)
}

impl Display for X64Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            X64Inst::Mov64rr { dst, src } => {
                write!(f, "mov {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Mov64ri { dst, imm } => write!(f, "mov {}, {imm}", reg_name(*dst)),
            X64Inst::Mov64rm { dst, src } => write!(f, "mov {}, {src}", reg_name(*dst)),
            X64Inst::Mov64mr { dst, src } => write!(f, "mov {dst}, {}", reg_name(*src)),
            X64Inst::Add64rr { dst, src } => {
                write!(f, "add {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Sub64rr { dst, src } => {
                write!(f, "sub {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Imul64rr { dst, src } => {
                write!(f, "imul {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Add64ri32 { dst, imm } => write!(f, "add {}, {imm}", reg_name(*dst)),
            X64Inst::Sub64ri32 { dst, imm } => write!(f, "sub {}, {imm}", reg_name(*dst)),
            X64Inst::Cmp64rr { lhs, rhs } => {
                write!(f, "cmp {}, {}", reg_name(*lhs), reg_name(*rhs))
            }
            X64Inst::Cmp64ri32 { lhs, imm } => write!(f, "cmp {}, {imm}", reg_name(*lhs)),
            X64Inst::Jmp { dst } => write!(f, "jmp {dst}"),
            X64Inst::CondJmp { cond, taken, not_taken } => {
                write!(f, "j{cond} {taken} else {not_taken}")
            }
            X64Inst::RawRet => f.write_str("ret"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add64rr_uses_both_operands_and_defs_dst() {
        let inst = X64Inst::Add64rr { dst: 1, src: 2 };
        assert_eq!(inst.get_uses().as_slice(), &[1, 2]);
        assert_eq!(inst.get_defs().as_slice(), &[1]);
    }

    #[test]
    fn mov64ri_defs_dst_and_uses_nothing() {
        let inst = X64Inst::Mov64ri { dst: 5, imm: 42 };
        assert_eq!(inst.get_defs().as_slice(), &[5]);
        assert!(inst.get_uses().is_empty());
    }

    #[test]
    fn mov64mr_uses_base_and_source() {
        let inst = X64Inst::Mov64mr {
            dst: Mem::base_disp(3, -8),
            src: 4,
        };
        assert_eq!(inst.get_uses().as_slice(), &[3, 4]);
        assert!(inst.get_defs().is_empty());
    }

    #[test]
    fn mov64rm_uses_base_and_index() {
        let inst = X64Inst::Mov64rm {
            dst: 7,
            src: Mem {
                base: 3,
                index: Some(4),
                scale: 8,
                disp: 16,
            },
        };
        assert_eq!(inst.get_uses().as_slice(), &[3, 4]);
        assert_eq!(inst.get_defs().as_slice(), &[7]);
    }

    #[test]
    fn cmp64rr_uses_both_and_defs_nothing() {
        let inst = X64Inst::Cmp64rr { lhs: 2, rhs: 3 };
        assert_eq!(inst.get_uses().as_slice(), &[2, 3]);
        assert!(inst.get_defs().is_empty());
    }

    #[test]
    fn jmp_and_condjmp_have_branch_targets() {
        use crate::support::slotmap::Key;
        let b1 = Block::new(1);
        let b2 = Block::new(2);
        assert_eq!(X64Inst::Jmp { dst: b1 }.get_branch_targets().as_slice(), &[b1]);
        let cj = X64Inst::CondJmp {
            cond: Cond::L,
            taken: b1,
            not_taken: b2,
        };
        assert_eq!(cj.get_branch_targets().as_slice(), &[b1, b2]);
        assert!(cj.is_branch());
        assert!(!cj.is_ret());
    }

    #[test]
    fn rawret_is_terminator() {
        let r = X64Inst::RawRet;
        assert!(r.is_ret());
        assert!(r.is_term());
        assert!(!r.is_branch());
        assert!(r.get_uses().is_empty());
        assert!(r.get_defs().is_empty());
    }

    #[test]
    fn display_add_mov() {
        let add = X64Inst::Add64rr { dst: 1, src: 2 };
        assert_eq!(format!("{add}"), "add v1, v2");
        let mov = X64Inst::Mov64ri { dst: 3, imm: -5 };
        assert_eq!(format!("{mov}"), "mov v3, -5");
    }
}
