use crate::codegen::tir::Reg;
pub const RAX: Reg = 0;
pub const RBX: Reg = 1;
pub const RCX: Reg = 2;
pub const RDX: Reg = 3;
pub const RSI: Reg = 4;
pub const RDI: Reg = 5;
pub const RSP: Reg = 6;
pub const RBP: Reg = 7;
pub const R8: Reg = 8;
pub const R9: Reg = 9;
pub const R10: Reg = 10;
pub const R11: Reg = 11;
pub const R12: Reg = 12;
pub const R13: Reg = 13;
pub const R14: Reg = 14;
pub const R15: Reg = 15;

// XMM physical registers occupy the `16..32` half of the preg id space,
// keeping the GPR and XMM classes disjoint so the allocator can route
// each vreg type into its own pool without extra bookkeeping.
pub const XMM0: Reg = 16;
pub const XMM1: Reg = 17;
pub const XMM2: Reg = 18;
pub const XMM3: Reg = 19;
pub const XMM4: Reg = 20;
pub const XMM5: Reg = 21;
pub const XMM6: Reg = 22;
pub const XMM7: Reg = 23;
pub const XMM8: Reg = 24;
pub const XMM9: Reg = 25;
pub const XMM10: Reg = 26;
pub const XMM11: Reg = 27;
pub const XMM12: Reg = 28;
pub const XMM13: Reg = 29;
pub const XMM14: Reg = 30;
pub const XMM15: Reg = 31;

/// First XMM preg id. Everything `>= XMM_BASE` is an XMM class register.
pub const XMM_BASE: Reg = 16;

/// `true` iff `r` is an XMM (floating-point / vector) physical register.
#[must_use]
pub fn is_xmm(r: Reg) -> bool {
    r >= XMM_BASE
}

