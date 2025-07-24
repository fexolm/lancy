use std::fmt::Display;

pub trait Inst: Sized + Copy + Display {
    fn is_branch(&self) -> bool;
    fn is_ret(&self) -> bool;

    fn is_term(&self) -> bool {
        self.is_branch() || self.is_ret()
    }
}
