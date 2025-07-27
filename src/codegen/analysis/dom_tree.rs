use crate::{
    codegen::tir::{Block, CFG},
    support::{
        bitset::FixedBitSet,
        slotmap::{Key, SecondaryMap},
    },
};

#[derive(Clone, Default)]
struct Node {
    rpo: u32,
    idom: Option<Block>,
}

pub struct DomTree {
    nodes: SecondaryMap<Block, Node>,
    reverse_postorder: Vec<Block>,
}

impl DomTree {
    pub fn build(cfg: &CFG) -> Self {
        let mut res = Self {
            nodes: SecondaryMap::with_capacity(cfg.blocks_count()),
            reverse_postorder: Vec::new(),
        };
        res.compute(cfg);
        res
    }

    fn compute(&mut self, cfg: &CFG) {
        self.compute_postorder(cfg);
        self.compute_domtree(cfg);
    }

    fn compute_postorder(&mut self, cfg: &CFG) {
        let mut visited = FixedBitSet::new(cfg.blocks_count());

        let mut stack = Vec::new();
        let entry = Block::new(0);
        stack.push(entry);

        while let Some(block) = stack.pop() {
            if visited.has(block.index()) {
                continue;
            }
            visited.add(block.index());
            self.reverse_postorder.push(block);

            for &succ in cfg.succs(block) {
                if !visited.has(succ.index()) {
                    stack.push(succ);
                }
            }
        }
    }

    fn compute_domtree(&mut self, cfg: &CFG) {
        const STRIDE: u32 = 4;
        let (entry_block, reverse_postorder) = match self.reverse_postorder.as_slice().split_first()
        {
            Some((&eb, rest)) => (eb, rest),
            None => return,
        };

        self.nodes[entry_block].rpo = 2 * STRIDE;

        for (rpo, &block) in reverse_postorder.iter().enumerate() {
            self.nodes[block] = Node {
                idom: self.compute_idom(block, cfg).into(),
                rpo: (rpo as u32 + 3) * STRIDE,
            }
        }

        let mut changed = true;
        while changed {
            changed = false;

            for block in reverse_postorder.iter() {
                let new_idom = self.compute_idom(*block, cfg).into();
                if self.nodes[*block].idom != new_idom {
                    self.nodes[*block].idom = new_idom;
                    changed = true;
                }
            }
        }
    }

    fn compute_idom(&self, block: Block, cfg: &CFG) -> Block {
        let mut reachable_preds = cfg
            .preds(block)
            .iter()
            .copied()
            .filter(|&pred| self.nodes[pred].rpo > 1);

        let mut idom = reachable_preds.next().unwrap();

        for pred in reachable_preds {
            idom = self.common_dominator(idom, pred);
        }

        idom
    }

    fn common_dominator(&self, mut a: Block, mut b: Block) -> Block {
        loop {
            let a_rpo = self.nodes[a].rpo;
            let b_rpo = self.nodes[b].rpo;

            if a_rpo < b_rpo {
                let idom = self.nodes[b].idom.unwrap();
                b = idom;
            } else if a_rpo > b_rpo {
                let idom = self.nodes[a].idom.unwrap();
                a = idom;
            } else {
                return a;
            }
        }
    }

    pub fn dominates(&self, a: Block, mut b: Block) -> bool {
        if a == b {
            return true;
        }

        let a_rpo = self.nodes[a].rpo;

        while a_rpo < self.nodes[b].rpo {
            if let Some(idom) = self.nodes[b].idom {
                b = idom;
            } else {
                return false; // b has no dominator
            }
        }

        a == b
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn simple_cfg() -> CFG {
        // Construct a simple CFG:
        // 0 -> 1 -> 2
        //      \-> 3
        let mut cfg = CFG::new(4);
        let b0 = Block::new(0);
        let b1 = Block::new(1);
        let b2 = Block::new(2);
        let b3 = Block::new(3);

        cfg.add_edge(b1, b0);
        cfg.add_edge(b2, b1);
        cfg.add_edge(b3, b1);

        cfg
    }

    fn diamond_cfg() -> CFG {
        // Construct a diamond CFG:
        //   0
        //  / \
        // 1   2
        //  \ /
        //   3
        let mut cfg = CFG::new(4);
        let b0 = Block::new(0);
        let b1 = Block::new(1);
        let b2 = Block::new(2);
        let b3 = Block::new(3);

        cfg.add_edge(b1, b0);
        cfg.add_edge(b2, b0);
        cfg.add_edge(b3, b1);
        cfg.add_edge(b3, b2);

        cfg
    }

    #[test]
    fn test_simple_cfg_domtree() {
        let cfg = simple_cfg();
        let domtree = DomTree::build(&cfg);

        let b0 = Block::new(0);
        let b1 = Block::new(1);
        let b2 = Block::new(2);
        let b3 = Block::new(3);

        assert!(domtree.dominates(b0, b1));
        assert!(domtree.dominates(b0, b2));
        assert!(domtree.dominates(b0, b3));
        assert!(domtree.dominates(b1, b2));
        assert!(domtree.dominates(b1, b3));
        assert!(!domtree.dominates(b2, b3));
        assert!(!domtree.dominates(b3, b2));
    }

    #[test]
    fn test_diamond_cfg_domtree() {
        let cfg = diamond_cfg();
        let domtree = DomTree::build(&cfg);

        let b0 = Block::new(0);
        let b1 = Block::new(1);
        let b2 = Block::new(2);
        let b3 = Block::new(3);

        assert!(domtree.dominates(b0, b1));
        assert!(domtree.dominates(b0, b2));
        assert!(domtree.dominates(b0, b3));
        assert!(!domtree.dominates(b1, b2));
        assert!(!domtree.dominates(b2, b1));
        assert!(!domtree.dominates(b1, b3));
        assert!(!domtree.dominates(b2, b3));
    }

    #[test]
    fn test_self_dominance() {
        let cfg = simple_cfg();
        let domtree = DomTree::build(&cfg);

        for i in 0..4 {
            let b = Block::new(i);
            assert!(domtree.dominates(b, b));
        }
    }

    // #[test]
    // fn test_unreachable_block() {
    //     // 0 -> 1
    //     // 2 (unreachable)
    //     let mut cfg = CFG::new(3);
    //     let b0 = Block::new(0);
    //     let b1 = Block::new(1);
    //     let b2 = Block::new(2);

    //     cfg.add_edge(b0, b1);

    //     let domtree = DomTree::build(&cfg);

    //     assert!(domtree.dominates(b0, b1));
    //     // Unreachable block should not dominate anything except itself
    //     assert!(domtree.dominates(b2, b2));
    //     assert!(!domtree.dominates(b2, b0));
    //     assert!(!domtree.dominates(b2, b1));
    // }
}
