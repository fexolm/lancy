use crate::codegen::tir::Reg;
use crate::codegen::tir::RegClass::Int;
use crate::codegen::tir::RegType::Physical;

pub const RAX: Reg = Reg::new(Physical, Int(8), 0);
pub const RBX: Reg = Reg::new(Physical, Int(8), 1);
pub const RCX: Reg = Reg::new(Physical, Int(8), 2);
pub const RDX: Reg = Reg::new(Physical, Int(8), 3);
pub const RSI: Reg = Reg::new(Physical, Int(8), 4);
pub const RDI: Reg = Reg::new(Physical, Int(8), 5);
pub const RSP: Reg = Reg::new(Physical, Int(8), 6);
pub const RBP: Reg = Reg::new(Physical, Int(8), 7);
pub const R8: Reg = Reg::new(Physical, Int(8), 8);
pub const R9: Reg = Reg::new(Physical, Int(8), 9);
pub const R10: Reg = Reg::new(Physical, Int(8), 10);
pub const R11: Reg = Reg::new(Physical, Int(8), 11);
pub const R12: Reg = Reg::new(Physical, Int(8), 12);
pub const R13: Reg = Reg::new(Physical, Int(8), 13);
pub const R14: Reg = Reg::new(Physical, Int(8), 14);
pub const R15: Reg = Reg::new(Physical, Int(8), 15);

pub(super) const REGISTERS_COUNT: usize = 16;
