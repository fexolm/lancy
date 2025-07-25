use std::fmt::Display;

use crate::codegen::tir::backend::{Backend, reg_name};
use crate::codegen::tir::{Reg, RegClass};
use crate::support::slotmap::PrimaryMap;

use super::{Block, BlockData, Inst};

pub struct Func<B: Backend> {
    name: String,
    blocks: PrimaryMap<Block, BlockData<B::Inst>>,
    max_vreg: u32,
    args: Vec<Reg>,
    results: Vec<Reg>,
}

impl<B: Backend> Func<B> {
    pub fn new(name: String, arg_types: Vec<RegClass>, result_types: Vec<RegClass>) -> Self {
        let mut max_vreg = 0;

        let args: Vec<_> = arg_types
            .iter()
            .enumerate()
            .map(|(i, &c)| Reg::virt(c, i as u32))
            .collect();

        max_vreg += args.len() as u32;

        let results: Vec<_> = result_types
            .iter()
            .enumerate()
            .map(|(i, &c)| Reg::virt(c, i as u32 + max_vreg))
            .collect();

        max_vreg += results.len() as u32;

        Func {
            name,
            max_vreg,
            blocks: PrimaryMap::new(),
            args,
            results,
        }
    }

    pub fn add_block(&mut self, data: BlockData<B::Inst>) -> Block {
        self.blocks.insert(data)
    }

    pub fn add_empty_block(&mut self) -> Block {
        self.blocks.insert(Default::default())
    }

    pub fn get_block_data_mut(&mut self, block: Block) -> &mut BlockData<B::Inst> {
        &mut self.blocks[block]
    }

    pub fn get_block_data(&self, block: Block) -> &BlockData<B::Inst> {
        &self.blocks[block]
    }

    pub fn new_vreg(&mut self, cls: RegClass) -> Reg {
        let res = Reg::virt(cls, self.max_vreg);
        self.max_vreg += 1;
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
}

impl<B: Backend> Display for Func<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn {}(", self.name)?;

        if self.args.len() > 0 {
            for &arg in &self.args[..self.args.len() - 1] {
                write!(f, "{}, ", reg_name::<B>(arg))?;
            }
            write!(f, "{})", reg_name::<B>(*self.args.last().unwrap()))?;
        }

        if self.results.len() > 0 {
            write!(f, " -> (")?;
            for &res in &self.results[..self.results.len() - 1] {
                write!(f, "{}, ", reg_name::<B>(res))?;
            }
            write!(f, "{})", reg_name::<B>(*self.results.last().unwrap()))?;
        }

        write!(f, "\n")?;

        for (id, data) in self.blocks.iter() {
            write!(f, "{id}\n{data}")?;
        }

        Ok(())
    }
}
