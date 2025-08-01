use smallvec::SmallVec;
use std::fmt::Display;

use crate::codegen::tir::Block;

use super::Reg;

pub trait Inst: Sized + Copy + Display {
    fn is_branch(&self) -> bool;
    fn is_ret(&self) -> bool;

    fn is_term(&self) -> bool {
        self.is_branch() || self.is_ret()
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]>;
    fn get_defs(&self) -> SmallVec<[Reg; 1]>;

    fn get_branch_targets(&self) -> SmallVec<[Block; 2]>;

    fn preg_name(reg: Reg) -> String;

    fn preg_count() -> u32;
}

pub fn reg_name<I: Inst>(reg: Reg) -> String {
    if reg < I::preg_count() {
        I::preg_name(reg)
    } else {
        format!("v{}", reg - I::preg_count())
    }
}
