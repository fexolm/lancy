use crate::codegen::tir::{Inst, Reg};

pub trait Backend {
    type Inst: Inst;

    fn preg_name(reg: Reg) -> String;
}

pub fn reg_name<B: Backend>(reg: Reg) -> String {
    match reg.typ() {
        super::RegType::Virtual => format!("v{}", reg.id()),
        super::RegType::Physical => B::preg_name(reg),
        super::RegType::Spill => format!("s{}", reg.id()),
    }
}
