use std::fmt::Display;

use crate::codegen::tir::{Reg, RegClass};
use crate::support::slotmap::PrimaryMap;

use super::{Inst, Block, BlockData};

pub struct Func<I: Inst> {
    blocks: PrimaryMap<Block, BlockData<I>>,
    max_vreg: u32
}

impl <I: Inst> Func<I> {
    pub fn new() -> Self {
        Func { max_vreg: 0, blocks: PrimaryMap::new() }
    }

    pub fn add_block(&mut self, data: BlockData<I>) -> Block {
        self.blocks.insert(data)
    }

    pub fn new_vreg(&mut self, cls: RegClass) -> Reg {
        let res = Reg::virt(cls, self.max_vreg);
        self.max_vreg += 1;
        res
    }
}

impl <I: Inst> Display for Func<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (id, data) in &self.blocks {
            write!(f, "{id}\n");
        }

        Ok(())
    }
}