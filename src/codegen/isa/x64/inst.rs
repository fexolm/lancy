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

/// `base + (index * scale) + disp`. Shared across every memory-accessing
/// instruction (MOV of all widths, LEA).
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
///
/// Width suffix conventions: `64/32/16/8` is the operand width in bits.
/// `rr` — register-register. `ri` — register-immediate (`ri32` means the
/// immediate is sign-extended from 32 bits — the widest form x86-64
/// accepts for an `ADD r64, imm`). `rm` — register-memory load.
/// `mr` — memory-register store. `r` (no second letter) — a single
/// register operand for a unary op. `rcl` — shift by CL (implicit RCX
/// use). `ri8` — shift by 8-bit immediate.
///
/// `Idiv64r` / `Div64r` carry the full ABI-visible operand set:
/// `hi_in`/`lo_in` are the pre-bound dividend halves (RDX/RAX),
/// `quotient`/`remainder` are the pre-bound results (RAX/RDX). Modeling
/// it this way lets liveness see the implicit reads/writes without
/// special-casing.
#[derive(Clone, Copy, Debug)]
pub enum X64Inst {
    // Moves — 64-bit.
    Mov64rr { dst: Reg, src: Reg },
    Mov64ri { dst: Reg, imm: i64 },
    Mov64rm { dst: Reg, src: Mem },
    Mov64mr { dst: Mem, src: Reg },

    // Moves — 32-bit. A 32-bit write to a GPR zero-extends to 64.
    Mov32rr { dst: Reg, src: Reg },
    Mov32ri { dst: Reg, imm: i32 },
    Mov32rm { dst: Reg, src: Mem },
    Mov32mr { dst: Mem, src: Reg },

    // Moves — 16-bit.
    Mov16rr { dst: Reg, src: Reg },
    Mov16ri { dst: Reg, imm: i16 },
    Mov16rm { dst: Reg, src: Mem },
    Mov16mr { dst: Mem, src: Reg },

    // Moves — 8-bit.
    Mov8rr { dst: Reg, src: Reg },
    Mov8ri { dst: Reg, imm: i8 },
    Mov8rm { dst: Reg, src: Mem },
    Mov8mr { dst: Mem, src: Reg },

    // Sign-extending loads (LLVM IR `sext`). 32→64 is the dedicated
    // `MOVSXD`; 8/16→64 use `MOVSX`.
    Movsx64r8 { dst: Reg, src: Reg },
    Movsx64r16 { dst: Reg, src: Reg },
    Movsxd64r32 { dst: Reg, src: Reg },

    // Zero-extending loads (LLVM IR `zext`). 32→64 is a plain MOV r32,
    // which implicitly zero-extends, so no separate variant.
    Movzx64r8 { dst: Reg, src: Reg },
    Movzx64r16 { dst: Reg, src: Reg },

    // Load effective address — used for GEP lowering.
    Lea64rm { dst: Reg, src: Mem },

    // Integer arithmetic — `dst = dst OP src`.
    Add64rr { dst: Reg, src: Reg },
    Sub64rr { dst: Reg, src: Reg },
    Imul64rr { dst: Reg, src: Reg },
    Add64ri32 { dst: Reg, imm: i32 },
    Sub64ri32 { dst: Reg, imm: i32 },

    // Signed / unsigned 128-into-64 division. `quotient` ends up in
    // RAX, `remainder` in RDX; `hi_in`/`lo_in` must be pre-bound to
    // RDX/RAX respectively. A frontend that only consumes one of
    // `quotient` / `remainder` must emit a `Kill` pseudo on the other
    // immediately after the div — otherwise regalloc will hold RAX or
    // RDX reserved for the unread value all the way to the end of the
    // function.
    Idiv64r {
        divisor: Reg,
        hi_in: Reg,
        lo_in: Reg,
        quotient: Reg,
        remainder: Reg,
    },
    Div64r {
        divisor: Reg,
        hi_in: Reg,
        lo_in: Reg,
        quotient: Reg,
        remainder: Reg,
    },

    // Bitwise — `dst = dst OP src`.
    And64rr { dst: Reg, src: Reg },
    Or64rr { dst: Reg, src: Reg },
    Xor64rr { dst: Reg, src: Reg },
    And64ri32 { dst: Reg, imm: i32 },
    Or64ri32 { dst: Reg, imm: i32 },
    Xor64ri32 { dst: Reg, imm: i32 },

    // Unary — `dst = OP dst`.
    Not64r { dst: Reg },
    Neg64r { dst: Reg },

    // Shifts — by 8-bit immediate or by CL. The `rcl` forms take a
    // `count` vreg that the frontend is responsible for pinning to RCX
    // via `reg_bind`.
    Shl64ri8 { dst: Reg, imm: u8 },
    Shr64ri8 { dst: Reg, imm: u8 },
    Sar64ri8 { dst: Reg, imm: u8 },
    Shl64rcl { dst: Reg, count: Reg },
    Shr64rcl { dst: Reg, count: Reg },
    Sar64rcl { dst: Reg, count: Reg },

    // Compare — sets EFLAGS, writes no GPR.
    Cmp64rr { lhs: Reg, rhs: Reg },
    Cmp64ri32 { lhs: Reg, imm: i32 },

    // Test — bit-AND without storing the result. Used for zero-check
    // idioms that LLVM IR O0 emits from `icmp eq %x, 0`.
    Test64rr { lhs: Reg, rhs: Reg },
    Test64ri32 { lhs: Reg, imm: i32 },

    // Conditional move — `if cond { dst = src }`. Used by frontends
    // that want `select` without a branch.
    Cmov64rr { cond: Cond, dst: Reg, src: Reg },

    // Set-byte-on-condition — materializes an `icmp` result into a
    // scalar `i1` (as a byte). The write is 8-bit, so callers usually
    // follow with `Movzx64r8` to widen.
    Setcc8r { cond: Cond, dst: Reg },

    // Call — indirect through a register. Direct calls are modeled at
    // the target-neutral `CallPseudo` layer and resolved by the JIT at
    // load time; that pseudo lowers to `Call64r` plus the ABI's
    // argument-placement / return-extraction moves. Callers that emit
    // `Call64r` on their own (bypassing `CallPseudo`) must wrap it in
    // `Kill` pseudos for every caller-saved vreg that crosses the call
    // — the regalloc otherwise has no clobber information and will
    // keep those values alive across the call.
    Call64r { target: Reg },

    // Control flow.
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
            X64Inst::Mov64rr { src, .. }
            | X64Inst::Mov32rr { src, .. }
            | X64Inst::Mov16rr { src, .. }
            | X64Inst::Mov8rr { src, .. }
            | X64Inst::Movsx64r8 { src, .. }
            | X64Inst::Movsx64r16 { src, .. }
            | X64Inst::Movsxd64r32 { src, .. }
            | X64Inst::Movzx64r8 { src, .. }
            | X64Inst::Movzx64r16 { src, .. } => smallvec![*src],
            X64Inst::Mov64ri { .. }
            | X64Inst::Mov32ri { .. }
            | X64Inst::Mov16ri { .. }
            | X64Inst::Mov8ri { .. }
            | X64Inst::RawRet => smallvec![],
            X64Inst::Mov64rm { src, .. }
            | X64Inst::Mov32rm { src, .. }
            | X64Inst::Mov16rm { src, .. }
            | X64Inst::Mov8rm { src, .. }
            | X64Inst::Lea64rm { src, .. } => src.get_uses(),
            X64Inst::Mov64mr { dst, src }
            | X64Inst::Mov32mr { dst, src }
            | X64Inst::Mov16mr { dst, src }
            | X64Inst::Mov8mr { dst, src } => {
                let mut uses: SmallVec<[Reg; 2]> = dst.get_uses();
                uses.push(*src);
                uses
            }
            X64Inst::Add64rr { dst, src }
            | X64Inst::Sub64rr { dst, src }
            | X64Inst::Imul64rr { dst, src }
            | X64Inst::And64rr { dst, src }
            | X64Inst::Or64rr { dst, src }
            | X64Inst::Xor64rr { dst, src } => smallvec![*dst, *src],
            X64Inst::Add64ri32 { dst, .. }
            | X64Inst::Sub64ri32 { dst, .. }
            | X64Inst::And64ri32 { dst, .. }
            | X64Inst::Or64ri32 { dst, .. }
            | X64Inst::Xor64ri32 { dst, .. }
            | X64Inst::Shl64ri8 { dst, .. }
            | X64Inst::Shr64ri8 { dst, .. }
            | X64Inst::Sar64ri8 { dst, .. } => smallvec![*dst],
            X64Inst::Not64r { dst } | X64Inst::Neg64r { dst } => smallvec![*dst],
            X64Inst::Shl64rcl { dst, count }
            | X64Inst::Shr64rcl { dst, count }
            | X64Inst::Sar64rcl { dst, count } => smallvec![*dst, *count],
            X64Inst::Idiv64r { divisor, hi_in, lo_in, .. }
            | X64Inst::Div64r { divisor, hi_in, lo_in, .. } => {
                smallvec![*divisor, *hi_in, *lo_in]
            }
            X64Inst::Cmp64rr { lhs, rhs } | X64Inst::Test64rr { lhs, rhs } => {
                smallvec![*lhs, *rhs]
            }
            X64Inst::Cmp64ri32 { lhs, .. } | X64Inst::Test64ri32 { lhs, .. } => {
                smallvec![*lhs]
            }
            X64Inst::Cmov64rr { dst, src, .. } => smallvec![*dst, *src],
            X64Inst::Setcc8r { .. } => smallvec![],
            X64Inst::Call64r { target } => smallvec![*target],
            X64Inst::Jmp { .. } | X64Inst::CondJmp { .. } => smallvec![],
        }
    }

    fn get_defs(&self) -> SmallVec<[Reg; 1]> {
        match self {
            X64Inst::Mov64rr { dst, .. }
            | X64Inst::Mov64ri { dst, .. }
            | X64Inst::Mov64rm { dst, .. }
            | X64Inst::Mov32rr { dst, .. }
            | X64Inst::Mov32ri { dst, .. }
            | X64Inst::Mov32rm { dst, .. }
            | X64Inst::Mov16rr { dst, .. }
            | X64Inst::Mov16ri { dst, .. }
            | X64Inst::Mov16rm { dst, .. }
            | X64Inst::Mov8rr { dst, .. }
            | X64Inst::Mov8ri { dst, .. }
            | X64Inst::Mov8rm { dst, .. }
            | X64Inst::Movsx64r8 { dst, .. }
            | X64Inst::Movsx64r16 { dst, .. }
            | X64Inst::Movsxd64r32 { dst, .. }
            | X64Inst::Movzx64r8 { dst, .. }
            | X64Inst::Movzx64r16 { dst, .. }
            | X64Inst::Lea64rm { dst, .. }
            | X64Inst::Add64rr { dst, .. }
            | X64Inst::Sub64rr { dst, .. }
            | X64Inst::Imul64rr { dst, .. }
            | X64Inst::Add64ri32 { dst, .. }
            | X64Inst::Sub64ri32 { dst, .. }
            | X64Inst::And64rr { dst, .. }
            | X64Inst::Or64rr { dst, .. }
            | X64Inst::Xor64rr { dst, .. }
            | X64Inst::And64ri32 { dst, .. }
            | X64Inst::Or64ri32 { dst, .. }
            | X64Inst::Xor64ri32 { dst, .. }
            | X64Inst::Not64r { dst }
            | X64Inst::Neg64r { dst }
            | X64Inst::Shl64ri8 { dst, .. }
            | X64Inst::Shr64ri8 { dst, .. }
            | X64Inst::Sar64ri8 { dst, .. }
            | X64Inst::Shl64rcl { dst, .. }
            | X64Inst::Shr64rcl { dst, .. }
            | X64Inst::Sar64rcl { dst, .. }
            | X64Inst::Cmov64rr { dst, .. }
            | X64Inst::Setcc8r { dst, .. } => smallvec![*dst],
            X64Inst::Idiv64r { quotient, remainder, .. }
            | X64Inst::Div64r { quotient, remainder, .. } => {
                let mut defs: SmallVec<[Reg; 1]> = smallvec![*quotient];
                defs.push(*remainder);
                defs
            }
            X64Inst::Mov64mr { .. }
            | X64Inst::Mov32mr { .. }
            | X64Inst::Mov16mr { .. }
            | X64Inst::Mov8mr { .. }
            | X64Inst::Cmp64rr { .. }
            | X64Inst::Cmp64ri32 { .. }
            | X64Inst::Test64rr { .. }
            | X64Inst::Test64ri32 { .. }
            | X64Inst::Call64r { .. }
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
            X64Inst::Mov32rr { dst, src } => {
                write!(f, "mov32 {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Mov32ri { dst, imm } => write!(f, "mov32 {}, {imm}", reg_name(*dst)),
            X64Inst::Mov32rm { dst, src } => write!(f, "mov32 {}, {src}", reg_name(*dst)),
            X64Inst::Mov32mr { dst, src } => write!(f, "mov32 {dst}, {}", reg_name(*src)),
            X64Inst::Mov16rr { dst, src } => {
                write!(f, "mov16 {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Mov16ri { dst, imm } => write!(f, "mov16 {}, {imm}", reg_name(*dst)),
            X64Inst::Mov16rm { dst, src } => write!(f, "mov16 {}, {src}", reg_name(*dst)),
            X64Inst::Mov16mr { dst, src } => write!(f, "mov16 {dst}, {}", reg_name(*src)),
            X64Inst::Mov8rr { dst, src } => {
                write!(f, "mov8 {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Mov8ri { dst, imm } => write!(f, "mov8 {}, {imm}", reg_name(*dst)),
            X64Inst::Mov8rm { dst, src } => write!(f, "mov8 {}, {src}", reg_name(*dst)),
            X64Inst::Mov8mr { dst, src } => write!(f, "mov8 {dst}, {}", reg_name(*src)),
            X64Inst::Movsx64r8 { dst, src } => {
                write!(f, "movsx {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Movsx64r16 { dst, src } => {
                write!(f, "movsx {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Movsxd64r32 { dst, src } => {
                write!(f, "movsxd {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Movzx64r8 { dst, src } => {
                write!(f, "movzx {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Movzx64r16 { dst, src } => {
                write!(f, "movzx {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Lea64rm { dst, src } => write!(f, "lea {}, {src}", reg_name(*dst)),
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
            X64Inst::And64rr { dst, src } => {
                write!(f, "and {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Or64rr { dst, src } => {
                write!(f, "or {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Xor64rr { dst, src } => {
                write!(f, "xor {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::And64ri32 { dst, imm } => write!(f, "and {}, {imm}", reg_name(*dst)),
            X64Inst::Or64ri32 { dst, imm } => write!(f, "or {}, {imm}", reg_name(*dst)),
            X64Inst::Xor64ri32 { dst, imm } => write!(f, "xor {}, {imm}", reg_name(*dst)),
            X64Inst::Not64r { dst } => write!(f, "not {}", reg_name(*dst)),
            X64Inst::Neg64r { dst } => write!(f, "neg {}", reg_name(*dst)),
            X64Inst::Shl64ri8 { dst, imm } => write!(f, "shl {}, {imm}", reg_name(*dst)),
            X64Inst::Shr64ri8 { dst, imm } => write!(f, "shr {}, {imm}", reg_name(*dst)),
            X64Inst::Sar64ri8 { dst, imm } => write!(f, "sar {}, {imm}", reg_name(*dst)),
            X64Inst::Shl64rcl { dst, count } => {
                write!(f, "shl {}, cl ; count={}", reg_name(*dst), reg_name(*count))
            }
            X64Inst::Shr64rcl { dst, count } => {
                write!(f, "shr {}, cl ; count={}", reg_name(*dst), reg_name(*count))
            }
            X64Inst::Sar64rcl { dst, count } => {
                write!(f, "sar {}, cl ; count={}", reg_name(*dst), reg_name(*count))
            }
            X64Inst::Idiv64r { divisor, quotient, remainder, hi_in, lo_in } => {
                write!(
                    f,
                    "{}, {} = idiv {} ; hi={}, lo={}",
                    reg_name(*quotient),
                    reg_name(*remainder),
                    reg_name(*divisor),
                    reg_name(*hi_in),
                    reg_name(*lo_in)
                )
            }
            X64Inst::Div64r { divisor, quotient, remainder, hi_in, lo_in } => {
                write!(
                    f,
                    "{}, {} = div {} ; hi={}, lo={}",
                    reg_name(*quotient),
                    reg_name(*remainder),
                    reg_name(*divisor),
                    reg_name(*hi_in),
                    reg_name(*lo_in)
                )
            }
            X64Inst::Cmp64rr { lhs, rhs } => {
                write!(f, "cmp {}, {}", reg_name(*lhs), reg_name(*rhs))
            }
            X64Inst::Cmp64ri32 { lhs, imm } => write!(f, "cmp {}, {imm}", reg_name(*lhs)),
            X64Inst::Test64rr { lhs, rhs } => {
                write!(f, "test {}, {}", reg_name(*lhs), reg_name(*rhs))
            }
            X64Inst::Test64ri32 { lhs, imm } => write!(f, "test {}, {imm}", reg_name(*lhs)),
            X64Inst::Cmov64rr { cond, dst, src } => {
                write!(f, "cmov{cond} {}, {}", reg_name(*dst), reg_name(*src))
            }
            X64Inst::Setcc8r { cond, dst } => write!(f, "set{cond} {}", reg_name(*dst)),
            X64Inst::Call64r { target } => write!(f, "call {}", reg_name(*target)),
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
    fn test64rr_uses_both_and_defs_nothing() {
        let inst = X64Inst::Test64rr { lhs: 2, rhs: 3 };
        assert_eq!(inst.get_uses().as_slice(), &[2, 3]);
        assert!(inst.get_defs().is_empty());
    }

    #[test]
    fn bitwise_rr_uses_both_defs_dst() {
        for inst in [
            X64Inst::And64rr { dst: 1, src: 2 },
            X64Inst::Or64rr { dst: 1, src: 2 },
            X64Inst::Xor64rr { dst: 1, src: 2 },
        ] {
            assert_eq!(inst.get_uses().as_slice(), &[1, 2]);
            assert_eq!(inst.get_defs().as_slice(), &[1]);
        }
    }

    #[test]
    fn unary_r_uses_and_defs_dst() {
        let n = X64Inst::Not64r { dst: 4 };
        let g = X64Inst::Neg64r { dst: 4 };
        for inst in [n, g] {
            assert_eq!(inst.get_uses().as_slice(), &[4]);
            assert_eq!(inst.get_defs().as_slice(), &[4]);
        }
    }

    #[test]
    fn shift_by_cl_uses_dst_and_count() {
        let inst = X64Inst::Shl64rcl { dst: 5, count: 6 };
        assert_eq!(inst.get_uses().as_slice(), &[5, 6]);
        assert_eq!(inst.get_defs().as_slice(), &[5]);
    }

    #[test]
    fn shift_by_imm_uses_dst_defs_dst() {
        let inst = X64Inst::Shr64ri8 { dst: 3, imm: 2 };
        assert_eq!(inst.get_uses().as_slice(), &[3]);
        assert_eq!(inst.get_defs().as_slice(), &[3]);
    }

    #[test]
    fn idiv_uses_divisor_and_dividend_halves_defs_both_outputs() {
        let inst = X64Inst::Idiv64r {
            divisor: 10,
            hi_in: 11,
            lo_in: 12,
            quotient: 13,
            remainder: 14,
        };
        assert_eq!(inst.get_uses().as_slice(), &[10, 11, 12]);
        assert_eq!(inst.get_defs().as_slice(), &[13, 14]);
    }

    #[test]
    fn cmov_uses_dst_and_src_defs_dst() {
        let inst = X64Inst::Cmov64rr { cond: Cond::L, dst: 2, src: 3 };
        assert_eq!(inst.get_uses().as_slice(), &[2, 3]);
        assert_eq!(inst.get_defs().as_slice(), &[2]);
    }

    #[test]
    fn setcc_defs_dst_uses_nothing() {
        let inst = X64Inst::Setcc8r { cond: Cond::G, dst: 7 };
        assert!(inst.get_uses().is_empty());
        assert_eq!(inst.get_defs().as_slice(), &[7]);
    }

    #[test]
    fn call_indirect_uses_target_defs_nothing_at_ir_level() {
        let inst = X64Inst::Call64r { target: 9 };
        assert_eq!(inst.get_uses().as_slice(), &[9]);
        // Return-value pins are modeled as separate Copy + RegDef pseudos.
        assert!(inst.get_defs().is_empty());
    }

    #[test]
    fn movsx_and_movzx_use_src_def_dst() {
        for inst in [
            X64Inst::Movsx64r8 { dst: 1, src: 2 },
            X64Inst::Movsx64r16 { dst: 1, src: 2 },
            X64Inst::Movsxd64r32 { dst: 1, src: 2 },
            X64Inst::Movzx64r8 { dst: 1, src: 2 },
            X64Inst::Movzx64r16 { dst: 1, src: 2 },
        ] {
            assert_eq!(inst.get_uses().as_slice(), &[2]);
            assert_eq!(inst.get_defs().as_slice(), &[1]);
        }
    }

    #[test]
    fn lea64rm_uses_addressing_operands_defs_dst() {
        let inst = X64Inst::Lea64rm {
            dst: 1,
            src: Mem { base: 2, index: Some(3), scale: 4, disp: 0 },
        };
        assert_eq!(inst.get_uses().as_slice(), &[2, 3]);
        assert_eq!(inst.get_defs().as_slice(), &[1]);
    }

    #[test]
    fn width_mov_uses_and_defs_like_64bit() {
        for inst in [
            X64Inst::Mov32rr { dst: 1, src: 2 },
            X64Inst::Mov16rr { dst: 1, src: 2 },
            X64Inst::Mov8rr { dst: 1, src: 2 },
        ] {
            assert_eq!(inst.get_uses().as_slice(), &[2]);
            assert_eq!(inst.get_defs().as_slice(), &[1]);
        }
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

    #[test]
    fn display_new_integer_variants() {
        assert_eq!(
            format!("{}", X64Inst::And64rr { dst: 1, src: 2 }),
            "and v1, v2"
        );
        assert_eq!(
            format!("{}", X64Inst::Or64ri32 { dst: 1, imm: 0xFF }),
            "or v1, 255"
        );
        assert_eq!(format!("{}", X64Inst::Not64r { dst: 3 }), "not v3");
        assert_eq!(format!("{}", X64Inst::Neg64r { dst: 3 }), "neg v3");
        assert_eq!(
            format!("{}", X64Inst::Shl64ri8 { dst: 4, imm: 3 }),
            "shl v4, 3"
        );
        assert_eq!(
            format!("{}", X64Inst::Shr64rcl { dst: 4, count: 5 }),
            "shr v4, cl ; count=v5"
        );
        assert_eq!(
            format!(
                "{}",
                X64Inst::Idiv64r {
                    divisor: 6,
                    hi_in: 7,
                    lo_in: 8,
                    quotient: 9,
                    remainder: 10
                }
            ),
            "v9, v10 = idiv v6 ; hi=v7, lo=v8"
        );
        assert_eq!(
            format!(
                "{}",
                X64Inst::Cmov64rr {
                    cond: Cond::L,
                    dst: 1,
                    src: 2
                }
            ),
            "cmovl v1, v2"
        );
        assert_eq!(
            format!(
                "{}",
                X64Inst::Setcc8r {
                    cond: Cond::Z,
                    dst: 1
                }
            ),
            "setz v1"
        );
        assert_eq!(
            format!("{}", X64Inst::Call64r { target: 11 }),
            "call v11"
        );
        assert_eq!(
            format!(
                "{}",
                X64Inst::Movsxd64r32 { dst: 1, src: 2 }
            ),
            "movsxd v1, v2"
        );
        assert_eq!(
            format!(
                "{}",
                X64Inst::Lea64rm {
                    dst: 1,
                    src: Mem { base: 2, index: Some(3), scale: 4, disp: 8 }
                }
            ),
            "lea v1, [v2+v3*4+8]"
        );
    }
}
