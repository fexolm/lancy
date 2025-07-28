use smallvec::SmallVec;

use crate::codegen::tir::Block;
use crate::support::slotmap::{SecondaryMap, SecondaryMapExt};

#[derive(Default, Clone)]
struct CFGNode {
    successors: SmallVec<[Block; 2]>,
    predecessors: SmallVec<[Block; 2]>,
}

pub struct CFG {
    nodes: SecondaryMap<Block, CFGNode>,
    entry: Block,
}

impl CFG {
    pub fn new(entry: Block, size: usize) -> Self {
        Self {
            nodes: SecondaryMap::with_default(size),
            entry,
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

    pub fn blocks_count(&self) -> usize {
        self.nodes.capacity()
    }

    pub fn get_entry_block(&self) -> Block {
        self.entry
    }
}
#[cfg(test)]
mod tests {
    use crate::support::slotmap::Key;

    use super::*;

    #[test]
    fn test_add_edge_and_query() {
        let mut cfg = CFG::new(Block::new(0), 3);
        let b0 = Block::new(0);
        let b1 = Block::new(1);
        let b2 = Block::new(2);

        cfg.add_edge(b1, b0);
        cfg.add_edge(b2, b1);

        assert_eq!(cfg.succs(b0), &[b1]);
        assert_eq!(cfg.preds(b1), &[b0]);
        assert_eq!(cfg.succs(b1), &[b2]);
        assert_eq!(cfg.preds(b2), &[b1]);
        assert!(cfg.preds(b0).is_empty());
        assert!(cfg.succs(b2).is_empty());
    }

    #[test]
    fn test_multiple_edges() {
        let mut cfg = CFG::new(Block::new(0), 4);
        let b0 = Block::new(0);
        let b1 = Block::new(1);
        let b2 = Block::new(2);
        let b3 = Block::new(3);

        cfg.add_edge(b1, b0);
        cfg.add_edge(b2, b0);
        cfg.add_edge(b3, b0);

        let mut succs = cfg.succs(b0).to_vec();
        succs.sort();
        assert_eq!(succs, vec![b1, b2, b3]);

        for &b in &[b1, b2, b3] {
            assert_eq!(cfg.preds(b), &[b0]);
        }
    }

    #[test]
    fn test_no_edges() {
        let cfg = CFG::new(Block::new(0), 2);
        let b0 = Block::new(0);
        let b1 = Block::new(1);

        assert!(cfg.succs(b0).is_empty());
        assert!(cfg.preds(b0).is_empty());
        assert!(cfg.succs(b1).is_empty());
        assert!(cfg.preds(b1).is_empty());
    }
}
