use crate::slotmap_key;
use std::fmt::{Debug, Display};

use super::{Inst, Instruction, PseudoInstruction};

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
    insts: Vec<Instruction<I>>,
}

impl<I: Inst> Default for BlockData<I> {
    fn default() -> Self {
        Self {
            insts: Vec::default(),
        }
    }
}

impl<I: Inst> BlockData<I> {
    #[must_use]
    pub fn new() -> Self {
        BlockData { insts: Vec::new() }
    }

    pub fn push_target_inst(&mut self, inst: I) {
        self.insts.push(Instruction::Target(inst));
    }
    pub fn push_pseudo_inst(&mut self, inst: PseudoInstruction) {
        self.insts.push(Instruction::Pseudo(inst));
    }

    #[must_use]
    pub fn get_terminator(&self) -> Option<Instruction<I>> {
        if let Some(inst) = self.insts.last()
            && inst.is_term()
        {
            Some(*inst)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item=&Instruction<I>> {
        self.insts.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.insts.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.insts.is_empty()
    }
}

impl<I: Inst> Display for BlockData<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for inst in &self.insts {
            writeln!(f, "    {inst}")?;
        }
        Ok(())
    }
}
