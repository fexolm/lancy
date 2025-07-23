pub trait Inst: Sized + Copy {
    fn is_term(&self) -> bool;
}
