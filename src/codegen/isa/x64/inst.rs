use crate::codegen::tir::Inst;

#[derive(Clone, Copy)]
pub enum X64Inst {
    Ret,
}

impl Inst for X64Inst {
    fn is_term(&self) -> bool {
        match self {
            X64Inst::Ret => true,
            _ => false,
        }
    }
}