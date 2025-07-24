use crate::slotmap_key;
use std::fmt::Display;

use super::Inst;
slotmap_key!(Block(u16));

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "^{}", self.0)
    }
}

pub struct BlockData<I: Inst> {
    insts: Vec<I>,
}

impl<I: Inst> BlockData<I> {
    pub fn new() -> Self {
        BlockData { insts: Vec::new() }
    }

    pub fn push(&mut self, inst: I) {
        self.insts.push(inst);
    }
}

impl <I: Inst> Display for BlockData<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for inst in &self.insts {
            write!(f, "    {inst}\n")?;
        }
        Ok(())
    }
}