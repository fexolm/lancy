use smallvec::SmallVec;

use crate::{codegen::tir::Block, support::slotmap::SecondaryMap};

#[derive(Default, Clone)]
struct DFGNode {
    successors: SmallVec<[Block; 2]>,
    predecessors: SmallVec<[Block; 2]>,
}

pub struct DFG {
    nodes: SecondaryMap<Block, DFGNode>,
}

impl DFG {
    pub fn new(size: usize) -> Self {
        Self {
            nodes: SecondaryMap::with_size(size),
        }
    }

    pub fn empty() -> Self {
        Self {
            nodes: SecondaryMap::new(),
        }
    }

    pub fn add_edge(&mut self, successor: Block, predecessor: Block) {
        self.nodes[successor].predecessors.push(predecessor);
        self.nodes[predecessor].successors.push(successor);
    }

    pub fn preds(&self, block: Block) -> &[Block] {
        &self.nodes[block].predecessors
    }

    pub fn succs(&self, block: Block) -> &[Block] {
        &self.nodes[block].successors
    }
}
