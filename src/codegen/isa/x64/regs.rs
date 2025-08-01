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

pub(super) const REGISTERS_COUNT: u32 = 16;
