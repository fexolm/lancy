//! System V AMD64 ABI — the default calling convention for the x64 target on
//! Linux/macOS/BSD.
//!
//! * Integer/pointer args: `RDI, RSI, RDX, RCX, R8, R9`.
//! * Integer return: `RAX`.
//! * Callee-saved: `RBX, RBP, R12..R15, RSP`.
//! * Caller-saved (volatile): `RAX, RCX, RDX, RSI, RDI, R8..R11`.
//! * 16-byte stack alignment *at the call instruction*; inside a callee,
//!   after `push rbp`, `rsp % 16 == 0`.
//!
//! Lives under `isa/x64/` because the register identifiers are x64-specific
//! and any practical consumer of this module also consumes the x64 ISA.

use crate::codegen::isa::x64::regs::{
    R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX, RSI, XMM0, XMM1, XMM2,
    XMM3, XMM4, XMM5, XMM6, XMM7,
};
use crate::codegen::tir::Reg;

pub const INT_ARG_REGS: &[Reg] = &[RDI, RSI, RDX, RCX, R8, R9];
pub const INT_RET_REG: Reg = RAX;

/// XMM registers used for floating-point/vector arguments under SysV.
pub const FP_ARG_REGS: &[Reg] = &[XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];
/// Floating-point return register: the first XMM.
pub const FP_RET_REG: Reg = XMM0;

pub const CALLEE_SAVED: &[Reg] = &[RBX, RBP, R12, R13, R14, R15];
pub const CALLER_SAVED: &[Reg] = &[RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11];

pub const STACK_ALIGN: u32 = 16;

/// Handle to the SysV AMD64 calling convention. Zero-sized — exists so the
/// compilation pipeline can talk about "which CC to use" without the ABI
/// lowering pass hardcoding global state.
#[derive(Clone, Copy, Debug, Default)]
pub struct SysVAmd64;

impl SysVAmd64 {
    /// Physical register used for incoming argument `idx` in the integer
    /// class, or `None` if `idx` exceeds the register-passed arg count
    /// (stack-passed args aren't supported yet).
    #[must_use]
    pub fn int_arg_reg(self, idx: u32) -> Option<Reg> {
        INT_ARG_REGS.get(idx as usize).copied()
    }

    /// XMM register used for float/vector argument `idx` in the SSE
    /// class. Independent of the integer-class counter — SysV gives the
    /// two classes separate counting pools.
    #[must_use]
    pub fn fp_arg_reg(self, idx: u32) -> Option<Reg> {
        FP_ARG_REGS.get(idx as usize).copied()
    }

    #[must_use]
    pub fn int_ret_reg(self) -> Reg {
        INT_RET_REG
    }

    #[must_use]
    pub fn fp_ret_reg(self) -> Reg {
        FP_RET_REG
    }

    #[must_use]
    pub fn max_int_args_in_regs(self) -> u32 {
        INT_ARG_REGS.len() as u32
    }

    #[must_use]
    pub fn max_fp_args_in_regs(self) -> u32 {
        FP_ARG_REGS.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arg_regs_in_expected_order() {
        assert_eq!(INT_ARG_REGS, &[RDI, RSI, RDX, RCX, R8, R9]);
    }

    #[test]
    fn callee_and_caller_saved_are_disjoint() {
        for c in CALLEE_SAVED {
            assert!(!CALLER_SAVED.contains(c));
        }
    }

    #[test]
    fn int_ret_is_rax() {
        assert_eq!(INT_RET_REG, RAX);
    }

    #[test]
    fn sysv_exposes_arg_regs_through_the_handle() {
        let cc = SysVAmd64;
        assert_eq!(cc.int_arg_reg(0), Some(RDI));
        assert_eq!(cc.int_arg_reg(5), Some(R9));
        assert_eq!(cc.int_arg_reg(6), None);
        assert_eq!(cc.max_int_args_in_regs(), 6);
        assert_eq!(cc.int_ret_reg(), RAX);
    }
}
