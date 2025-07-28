use smallvec::SmallVec;

use crate::{
    codegen::tir::{Block, CFG, Func, Inst, RegType},
    support::{
        bitset::FixedBitSet,
        slotmap::{Key, SecondaryMap, SecondaryMapExt},
    },
};

pub struct LivenessAnalysis {
    live_in: SecondaryMap<Block, FixedBitSet>,
    live_out: SecondaryMap<Block, FixedBitSet>,
    uses: SecondaryMap<Block, FixedBitSet>,
    defs: SecondaryMap<Block, FixedBitSet>,
    live_ranges: SecondaryMap<u32, Vec<Block>>,
}

impl LivenessAnalysis {
    pub fn new<I: Inst>(func: &Func<I>, cfg: &CFG) -> Self {
        let live_in = SecondaryMap::new(
            cfg.blocks_count(),
            FixedBitSet::zeroes(func.get_vregs_count()),
        );
        let live_out = SecondaryMap::new(
            cfg.blocks_count(),
            FixedBitSet::zeroes(func.get_vregs_count()),
        );
        let uses = SecondaryMap::new(
            cfg.blocks_count(),
            FixedBitSet::zeroes(func.get_vregs_count()),
        );
        let defs = SecondaryMap::new(
            cfg.blocks_count(),
            FixedBitSet::zeroes(func.get_vregs_count()),
        );

        let live_ranges = SecondaryMap::with_default(func.get_vregs_count() as usize);

        let mut analysis = Self {
            live_in,
            live_out,
            uses,
            defs,
            live_ranges,
        };

        analysis.construct(func, cfg);
        analysis.compute_live_ranges(func, cfg);

        analysis
    }

    fn compute_reverse_postorder<I: Inst>(
        func: &Func<I>,
        cfg: &CFG,
    ) -> smallvec::SmallVec<[Block; 16]> {
        let mut visited = FixedBitSet::zeroes(cfg.blocks_count());

        let mut stack = Vec::new();
        let entry = func.get_entry_block().unwrap();
        stack.push(entry);

        let mut postorder = SmallVec::new();
        postorder.reserve(cfg.blocks_count());

        while let Some(block) = stack.pop() {
            if visited.has(block.index()) {
                continue;
            }
            visited.add(block.index());

            postorder.push(block);

            for &succ in cfg.succs(block) {
                if !visited.has(succ.index()) {
                    stack.push(succ);
                }
            }
        }
        postorder
    }

    fn init_block<I: Inst>(&mut self, block: Block, func: &Func<I>, cfg: &CFG) {
        for inst in func.get_block_data(block).iter() {
            let uses = inst.get_uses();
            let defs = inst.get_defs();

            let block_defs = &mut self.defs[block];
            let block_uses = &mut self.uses[block];

            for r in uses {
                let id = r.get_id() as usize;
                if r.get_type() == RegType::Virtual && block_defs.has(id) {
                    block_uses.add(id);
                }
            }

            for r in defs {
                let id = r.get_id() as usize;
                if r.get_type() == RegType::Virtual {
                    block_defs.add(id);
                }
            }
        }
    }

    fn construct<I: Inst>(&mut self, func: &Func<I>, cfg: &CFG) {
        let mut worklist = Self::compute_reverse_postorder(&func, cfg);
        let mut processed = FixedBitSet::zeroes(cfg.blocks_count());

        let mut changed = true;

        for (b, _) in func.blocks_iter() {
            self.init_block(b, func, cfg);
        }

        while let Some(block) = worklist.pop() {
            if processed.has(block.index()) {
                continue;
            }
            processed.add(block.index());

            let line_ins_count = self.live_in[block].ones_count();

            for &s in cfg.succs(block) {
                self.live_out[block].union(&self.live_in[s]);
            }

            self.live_in[block].union(&self.live_out[block]);
            self.live_in[block].difference(&self.defs[block]);
            self.live_in[block].union(&self.uses[block]);

            if self.live_in[block].ones_count() != line_ins_count {
                worklist.extend_from_slice(cfg.preds(block));
            }
        }
    }

    fn compute_live_ranges<I: Inst>(&mut self, func: &Func<I>, cfg: &CFG) {
        for (block, _) in func.blocks_iter() {
            for reg_id in 0..func.get_vregs_count() {
                if self.live_in[block].has(reg_id) || self.live_out[block].has(reg_id) {
                    self.live_ranges[reg_id as u32].push(block);
                }
            }
        }
    }

    pub fn get_life_range(&self, reg_id: u32) -> &[Block] {
        &self.live_ranges[reg_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{
        isa::x64::inst::X64Inst,
        tir::{BlockData, Func, Inst, Reg, RegClass},
    };

    #[test]
    fn simple_test() {
        let mut func = Func::new(
            "foo".to_string(),
            vec![RegClass::Int(8)],
            vec![RegClass::Int(8)],
        );

        let block1 = func.add_empty_block();

        let block2 = {
            let mut block_data = BlockData::new();

            let src = func.get_arg(0);
            let dst = func.get_result(0);

            block_data.push(X64Inst::Mov64rr { src, dst });
            block_data.push(X64Inst::Ret);

            func.add_block(block_data)
        };

        {
            let b1_data = func.get_block_data_mut(block1);
            b1_data.push(X64Inst::Jmp { dst: block2 });
        }

        func.construct_cfg().unwrap();
        let analysis = LivenessAnalysis::new(&func, &func.get_cfg());

        let live_range = analysis.get_life_range(0);
        assert_eq!(live_range, &[block1, block2]);
    }
}
