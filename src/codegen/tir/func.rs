use std::fmt::Display;

use crate::support::slotmap::{Key, PrimaryMap};

use super::{Block, BlockData, Inst};

pub type Reg = u32;

pub struct Func<I: Inst> {
    name: String,
    blocks: PrimaryMap<Block, BlockData<I>>,
    regs_count: u32,
}

impl<I: Inst> Func<I> {
    #[must_use]
    pub fn new(name: String) -> Self {
        Func {
            name,
            regs_count: 0,
            blocks: PrimaryMap::new(),
        }
    }

    pub fn add_block(&mut self, data: BlockData<I>) -> Block {
        self.blocks.insert(data)
    }

    pub fn add_empty_block(&mut self) -> Block {
        self.blocks.insert(BlockData::default())
    }

    pub fn get_block_data_mut(&mut self, block: Block) -> &mut BlockData<I> {
        &mut self.blocks[block]
    }

    #[must_use]
    pub fn get_block_data(&self, block: Block) -> &BlockData<I> {
        &self.blocks[block]
    }

    pub fn new_vreg(&mut self) -> Reg {
        let res = self.regs_count;
        self.regs_count += 1;
        res
    }

    #[must_use]
    pub fn get_regs_count(&self) -> usize {
        self.regs_count as usize
    }

    #[must_use]
    pub fn get_entry_block(&self) -> Option<Block> {
        if self.blocks.is_empty() {
            None
        } else {
            Some(Block::new(0))
        }
    }

    pub fn blocks_iter(&self) -> impl Iterator<Item=(Block, &BlockData<I>)> {
        self.blocks.iter()
    }

    #[must_use]
    pub fn blocks_count(&self) -> usize {
        self.blocks.len()
    }
}

impl<I: Inst> Display for Func<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}:", self.name)?;

        for (id, data) in self.blocks.iter() {
            write!(f, "{id}")?;
            write!(f, "\n{data}")?;
        }

        Ok(())
    }
}
