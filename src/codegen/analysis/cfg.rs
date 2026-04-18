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
    #[must_use] 
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
                        cfg.add_edge(block, t);
                    }
                }
            } else {
                return Err(TirError::BlockNotTerminated(block));
            }
        }

        Ok(cfg)
    }

    /// Add a directed edge `from → to`. Updates both the predecessor list of
    /// `to` (which gains `from`) and the successor list of `from` (which
    /// gains `to`).
    pub fn add_edge(&mut self, from: Block, to: Block) {
        self.nodes.get_mut(to).unwrap().predecessors.push(from);
        self.nodes.get_mut(from).unwrap().successors.push(to);
    }

    #[must_use] 
    pub fn preds(&self, block: Block) -> &[Block] {
        &self.nodes[block].predecessors
    }

    #[must_use] 
    pub fn succs(&self, block: Block) -> &[Block] {
        &self.nodes[block].successors
    }

    #[must_use] 
    pub fn blocks_count(&self) -> usize {
        self.nodes.capacity()
    }

    #[must_use] 
    pub fn get_entry_block(&self) -> Block {
        self.entry
    }
}

/// Reverse post-order traversal of the CFG starting from the entry block.
///
/// The returned vector has the property that every block appears after all
/// its non-back-edge predecessors — this makes it the standard iteration
/// order for forward dataflow analyses and the natural block layout order.
///
/// Implemented iteratively via a work-stack with a `processed` flag: on
/// first visit we push self (with processed=true) plus successors (with
/// processed=false); when we pop with processed=true all descendants have
/// already been emitted, so this is the post-order step. Reversing the
/// accumulated post-order gives RPO.
#[must_use]
pub fn reverse_post_order(cfg: &CFG) -> Vec<Block> {
    let mut visited = FixedBitSet::zeroes(cfg.blocks_count());
    let mut post: Vec<Block> = Vec::with_capacity(cfg.blocks_count());
    let mut stack: Vec<(Block, bool)> = vec![(cfg.get_entry_block(), false)];
    while let Some((block, processed)) = stack.pop() {
        if processed {
            post.push(block);
            continue;
        }
        if visited.has(block.index()) {
            continue;
        }
        visited.add(block.index());
        stack.push((block, true));
        for &succ in cfg.succs(block) {
            if !visited.has(succ.index()) {
                stack.push((succ, false));
            }
        }
    }
    post.reverse();
    post
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

        cfg.add_edge(b0, b1);
        cfg.add_edge(b1, b2);

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

        cfg.add_edge(b0, b1);
        cfg.add_edge(b0, b2);
        cfg.add_edge(b0, b3);

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

    #[test]
    fn rpo_has_every_block_after_all_non_back_edge_predecessors() {
        // Diamond: 0 → {1, 2} → 3. In RPO, 0 must come first, 3 last, and
        // both 1 and 2 must come strictly between. This is the defining
        // property of RPO on an acyclic CFG.
        let mut cfg = CFG::new(Block::new(0), 4);
        let [b0, b1, b2, b3] = [0, 1, 2, 3].map(Block::new);
        cfg.add_edge(b0, b1);
        cfg.add_edge(b0, b2);
        cfg.add_edge(b1, b3);
        cfg.add_edge(b2, b3);

        let rpo = reverse_post_order(&cfg);
        assert_eq!(rpo.len(), 4);
        let pos = |b: Block| rpo.iter().position(|&x| x == b).unwrap();
        assert!(pos(b0) < pos(b1));
        assert!(pos(b0) < pos(b2));
        assert!(pos(b1) < pos(b3));
        assert!(pos(b2) < pos(b3));
    }

    #[test]
    fn rpo_over_a_back_edge_still_visits_loop_header_before_body() {
        // 0 → 1 → 2 → 3, with back edge 3 → 1. Loop header `1` must still
        // come before `2` and `3` in RPO; the back edge doesn't re-order.
        let mut cfg = CFG::new(Block::new(0), 4);
        let [b0, b1, b2, b3] = [0, 1, 2, 3].map(Block::new);
        cfg.add_edge(b0, b1);
        cfg.add_edge(b1, b2);
        cfg.add_edge(b2, b3);
        cfg.add_edge(b3, b1);

        let rpo = reverse_post_order(&cfg);
        let pos = |b: Block| rpo.iter().position(|&x| x == b).unwrap();
        assert!(pos(b0) < pos(b1));
        assert!(pos(b1) < pos(b2));
        assert!(pos(b2) < pos(b3));
    }
}
