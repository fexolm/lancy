use std::fmt::Display;
use std::io::empty;

use crate::codegen::tir::{CFG, reg_name};
use crate::support::slotmap::{Key, PrimaryMap};

use super::{Block, BlockData, Inst, TirError};

pub type Reg = u32;

pub struct Func<I: Inst> {
    name: String,
    blocks: PrimaryMap<Block, BlockData<I>>,
    regs_count: u32,
    cfg: Option<CFG>,
}

impl<I: Inst> Func<I> {
    pub fn new(name: String) -> Self {
        let mut regs_count = I::preg_count() as u32;

        Func {
            name,
            regs_count,
            blocks: PrimaryMap::new(),
            cfg: None,
        }
    }

    pub fn add_block(&mut self, data: BlockData<I>) -> Block {
        self.invalidate_dfg();
        self.blocks.insert(data)
    }

    pub fn add_empty_block(&mut self) -> Block {
        self.invalidate_dfg();
        self.blocks.insert(Default::default())
    }

    pub fn get_block_data_mut(&mut self, block: Block) -> &mut BlockData<I> {
        self.invalidate_dfg();
        &mut self.blocks[block]
    }

    pub fn get_block_data(&self, block: Block) -> &BlockData<I> {
        &self.blocks[block]
    }

    pub fn new_vreg(&mut self) -> Reg {
        let res = self.regs_count;
        self.regs_count += 1;
        res
    }

    pub fn construct_cfg(&mut self) -> Result<(), TirError> {
        let entry = self.get_entry_block().ok_or(TirError::EmptyFunctionBody)?;
        let mut cfg = CFG::new(entry, self.blocks.len());
        for (block, data) in self.blocks.iter() {
            if let Some(term) = data.get_terminator() {
                if term.is_branch() {
                    let targets = term.get_branch_targets();
                    for t in targets {
                        cfg.add_edge(t, block);
                    }
                }
            } else {
                return Err(TirError::BlockNotTerminated(block));
            }
        }
        self.cfg = Some(cfg);
        Ok(())
    }

    fn invalidate_dfg(&mut self) {
        self.cfg = None;
    }

    pub fn get_cfg(&self) -> &CFG {
        self.cfg.as_ref().unwrap()
    }

    pub fn get_regs_count(&self) -> usize {
        self.regs_count as usize
    }

    pub fn get_entry_block(&self) -> Option<Block> {
        if self.blocks.len() > 0 {
            Some(Block::new(0))
        } else {
            None
        }
    }

    pub fn blocks_iter(&self) -> impl Iterator<Item = (Block, &BlockData<I>)> {
        self.blocks.iter()
    }
}

impl<I: Inst> Display for Func<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:\n", self.name)?;

        for (id, data) in self.blocks.iter() {
            write!(f, "{id}")?;

            if let Some(dfg) = &self.cfg {
                write!(
                    f,
                    "    ; preds {:?}, succs: {:?}",
                    dfg.preds(id),
                    dfg.succs(id)
                )?;
            }

            write!(f, "\n{data}")?;
        }

        Ok(())
    }
}
