use smallvec::SmallVec;

use crate::codegen::tir::{Block, Func, Inst, TirError};
use crate::support::bitset::FixedBitSet;
use crate::support::slotmap::{Key, SecondaryMap};

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
        let mut nodes = SecondaryMap::new(size);
        nodes.fill(CFGNode::default());
        Self {
            nodes,
            entry,
        }
    }
    pub fn compute<I: Inst>(func: &Func<I>) -> Result<CFG, TirError> {
        let size = func.blocks_count();
        let entry = func.get_entry_block().ok_or(TirError::EmptyFunctionBody)?;

        let mut cfg = Self::new(entry, size);

        for (block, data) in func.blocks_iter() {
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

        Ok(cfg)
    }

    pub fn add_edge(&mut self, successor: Block, predecessor: Block) {
        self.nodes.get_mut(successor).unwrap().predecessors.push(predecessor);
        self.nodes.get_mut(predecessor).unwrap().successors.push(successor);
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

pub fn reverse_post_order(cfg: &CFG) -> Vec<Block> {
    let mut visited = FixedBitSet::zeroes(cfg.blocks_count());

    let mut stack = Vec::new();
    let entry = cfg.get_entry_block();
    stack.push(entry);

    let mut rpo = Vec::with_capacity(cfg.blocks_count());

    while let Some(block) = stack.pop() {
        if visited.has(block.index()) {
            continue;
        }
        visited.add(block.index());

        rpo.push(block);

        for &succ in cfg.succs(block) {
            if !visited.has(succ.index()) {
                stack.push(succ);
            }
        }
    }
    rpo
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
