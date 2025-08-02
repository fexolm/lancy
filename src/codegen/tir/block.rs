use crate::{codegen::tir::TirError, slotmap_key};
use std::fmt::{Debug, Display};

use super::Inst;

slotmap_key!(Block(u16));

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@{}", self.0)
    }
}

impl Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@{}", self.0)
    }
}

#[derive(Clone)]
pub struct BlockData<I: Inst> {
    insts: Vec<I>,
}

impl<I: Inst> Default for BlockData<I> {
    fn default() -> Self {
        Self {
            insts: Default::default(),
        }
    }
}

impl<I: Inst> BlockData<I> {
    pub fn new() -> Self {
        BlockData { insts: Vec::new() }
    }

    pub fn push(&mut self, inst: I) {
        self.insts.push(inst);
    }

    pub fn get_terminator(&self) -> Option<I> {
        if let Some(inst) = self.insts.last()
            && inst.is_term()
        {
            Some(*inst)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &I> {
        self.insts.iter()
    }

    pub fn len(&self) -> usize {
        self.insts.len()
    }
}

impl<I: Inst> Display for BlockData<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for inst in &self.insts {
            write!(f, "    {inst}\n")?;
        }
        Ok(())
    }
}
