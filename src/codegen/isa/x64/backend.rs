use crate::codegen::{isa::x64::inst::X64Inst, tir::Backend};

pub struct X64Backend {}

impl Backend for X64Backend {
    type Inst = X64Inst;

    fn preg_name(reg: crate::codegen::tir::Reg) -> String {
        todo!()
    }
}
