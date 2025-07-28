use std::fmt::Display;
use std::io::empty;

use crate::codegen::tir::{CFG, Reg, RegClass, RegType, reg_name};
use crate::support::slotmap::{Key, PrimaryMap};

use super::{Block, BlockData, Inst, TirError};

pub struct Func<I: Inst> {
    name: String,
    blocks: PrimaryMap<Block, BlockData<I>>,
    vregs_count: u32,
    args: Vec<Reg>,
    results: Vec<Reg>,
    cfg: Option<CFG>,
}

impl<I: Inst> Func<I> {
    pub fn new(name: String, arg_types: Vec<RegClass>, result_types: Vec<RegClass>) -> Self {
        let mut vregs_count = 0;

        let args: Vec<_> = arg_types
            .iter()
            .enumerate()
            .map(|(i, &c)| Reg::new(RegType::Virtual, c, i as u32))
            .collect();

        vregs_count += args.len() as u32;

        let results: Vec<_> = result_types
            .iter()
            .enumerate()
            .map(|(i, &c)| Reg::new(RegType::Virtual, c, i as u32 + vregs_count))
            .collect();

        vregs_count += results.len() as u32;

        Func {
            name,
            vregs_count,
            blocks: PrimaryMap::new(),
            args,
            results,
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

    pub fn new_vreg(&mut self, cls: RegClass) -> Reg {
        let res = Reg::new(RegType::Virtual, cls, self.vregs_count);
        self.vregs_count += 1;
        res
    }

    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    pub fn get_arg(&self, i: usize) -> Reg {
        self.args[i]
    }

    pub fn num_results(&self) -> usize {
        self.results.len()
    }

    pub fn get_result(&self, i: usize) -> Reg {
        self.results[i]
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

    pub fn get_vregs_count(&self) -> usize {
        self.vregs_count as usize
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
        write!(f, "fn {}(", self.name)?;

        if self.args.len() > 0 {
            for &arg in &self.args[..self.args.len() - 1] {
                write!(f, "{}, ", reg_name::<I>(arg))?;
            }
            write!(f, "{})", reg_name::<I>(*self.args.last().unwrap()))?;
        }

        if self.results.len() > 0 {
            write!(f, " -> (")?;
            for &res in &self.results[..self.results.len() - 1] {
                write!(f, "{}, ", reg_name::<I>(res))?;
            }
            write!(f, "{})", reg_name::<I>(*self.results.last().unwrap()))?;
        }

        write!(f, "\n")?;

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
