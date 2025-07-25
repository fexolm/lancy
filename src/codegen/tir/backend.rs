use crate::codegen::tir::{Inst, Reg};

pub trait Backend {
    type Inst: Inst;

    fn preg_name(reg: Reg) -> String;
}

pub fn reg_name<B: Backend>(reg: Reg) -> String {
    if reg.is_virtual() {
        format!("v{}", reg.id())
    } else {
        B::preg_name(reg)
    }
}
