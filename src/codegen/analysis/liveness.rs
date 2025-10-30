use crate::codegen::tir::reverse_post_order;
use crate::{
    codegen::tir::{Block, Func, Inst, Reg, CFG},
    support::{
        bitset::FixedBitSet,
        slotmap::{Key, SecondaryMap},
    },
};
use std::ops::Index;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ProgramPoint {
    pub block: Block,
    pub inst_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct LiveRange {
    pub start: ProgramPoint,
    pub end: ProgramPoint,
}

struct UseDefs {
    uses: SecondaryMap<Block, FixedBitSet>,
    defs: SecondaryMap<Block, FixedBitSet>,
}

struct LivenessAnalysis {
    live_in: SecondaryMap<Block, FixedBitSet>,
    live_out: SecondaryMap<Block, FixedBitSet>,
}

#[derive(Default)]
pub struct LiveRanges {
    ranges: SecondaryMap<Reg, Vec<LiveRange>>,
}

impl UseDefs {
    pub fn compute<I: Inst>(func: &Func<I>) -> Self {
        let mut uses = SecondaryMap::new(func.blocks_count());
        uses.fill(FixedBitSet::zeroes(func.get_regs_count()));
        let mut defs = SecondaryMap::new(func.blocks_count());
        defs.fill(FixedBitSet::zeroes(func.get_regs_count()));

        let mut res = Self { uses, defs };

        for (b, _) in func.blocks_iter() {
            res.compute_block(b, func);
        }

        res
    }

    fn compute_block<I: Inst>(&mut self, block: Block, func: &Func<I>) {
        for inst in func.get_block_data(block).iter() {
            let uses = inst.get_uses();
            let defs = inst.get_defs();

            let block_defs = &mut self.defs.get_mut(block).unwrap();
            let block_uses = &mut self.uses.get_mut(block).unwrap();

            for r in uses {
                let id = r as usize;
                if !block_defs.has(id) {
                    block_uses.add(id);
                }
            }

            for r in defs {
                let id = r as usize;
                block_defs.add(id);
            }
        }
    }

    pub fn get_uses(&self, block: Block) -> &FixedBitSet {
        &self.uses[block]
    }

    pub fn get_defs(&self, block: Block) -> &FixedBitSet {
        &self.defs[block]
    }
}

impl LivenessAnalysis {
    pub fn compute<I: Inst>(func: &Func<I>, cfg: &CFG) -> Self {
        let regs_count = func.get_regs_count() as usize;
        let mut live_in = SecondaryMap::new(cfg.blocks_count());
        live_in.fill(FixedBitSet::zeroes(regs_count));
        let mut live_out = SecondaryMap::new(cfg.blocks_count());
        live_out.fill(FixedBitSet::zeroes(regs_count));

        let mut analysis = Self {
            live_in,
            live_out,
        };

        analysis.do_compute(func, cfg);
        analysis
    }

    fn do_compute<I: Inst>(&mut self, func: &Func<I>, cfg: &CFG) {
        let mut worklist = reverse_post_order(cfg);
        let mut changed = true;

        let usedefs = UseDefs::compute(func);

        while let Some(block) = worklist.pop() {
            let line_ins_count = self.live_in[block].ones_count();
            let line_outs_count = self.live_out[block].ones_count();

            let mut live_out = self.live_out.get_mut(block).unwrap();

            for &s in cfg.succs(block) {
                live_out.union(&self.live_in[s]);
            }

            let mut live_in = self.live_in.get_mut(block).unwrap();

            live_in.union(&self.live_out[block]);
            live_in.difference(&usedefs.get_defs(block));
            live_in.union(&usedefs.get_uses(block));

            if self.live_in[block].ones_count() != line_ins_count
                || self.live_out[block].ones_count() != line_outs_count
            {
                worklist.extend_from_slice(cfg.preds(block));
            }
        }
    }
}

impl LiveRanges {
    pub fn compute<I: Inst>(func: &Func<I>, cfg: &CFG) -> Self {
        let entry = if let Some(entry) = func.get_entry_block() {
            entry
        } else {
            return Self::default();
        };

        let liveness = LivenessAnalysis::compute(func, cfg);

        let mut ranges: SecondaryMap<Reg, Vec<LiveRange>> = SecondaryMap::new(func.get_regs_count());

        let mut live_vars = liveness.live_in[entry].clone();

        for r in live_vars.iter_ones() {
            ranges[r as Reg].push(LiveRange {
                start: ProgramPoint {
                    block: entry,
                    inst_index: 0,
                },
                end: ProgramPoint {
                    block: entry,
                    inst_index: 0,
                },
            });
        }

        for (block, block_data) in func.blocks_iter() {
            let live_in = &liveness.live_in[block];
            let live_out = &liveness.live_out[block];

            live_vars.union(live_in);

            for (idx, inst) in block_data.iter().enumerate() {
                let p = ProgramPoint {
                    block,
                    inst_index: idx as u32,
                };

                for r in inst.get_uses() {
                    ranges[r].last_mut().unwrap().end = p;
                }

                for r in inst.get_defs() {
                    if live_vars.has(r as usize) && live_out.has(r as usize) {
                        continue;
                    }

                    ranges[r].push(LiveRange {
                        start: p,
                        end: p,
                    });
                }
            }
            for r in live_out.iter_ones() {
                ranges[r as Reg].last_mut().unwrap().end = ProgramPoint {
                    block,
                    inst_index: block_data.len() as u32,
                };
            }
        }

        Self {
            ranges
        }
    }
}

impl Index<Reg> for LiveRanges {
    type Output = [LiveRange];

    fn index(&self, index: Reg) -> &Self::Output {
        &self.ranges[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{
        isa::x64::inst::X64Inst,
        tir::{BlockData, Func},
    };


    #[test]
    fn usedef_test() {
        // foo:
        // @0
        //     mov v0 v1
        //     jmp @1
        // @1
        //     mov v1 v0
        //     ret
        let mut func = Func::<X64Inst>::new("foo".to_string());

        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let b1 = {
            let mut block_data = BlockData::new();

            block_data.push(X64Inst::Mov64rr { dst: v1, src: v0 });
            block_data.push(X64Inst::Ret { src: v1 });

            func.add_block(block_data)
        };

        {
            let block_data = func.get_block_data_mut(b0);
            block_data.push(X64Inst::Mov64rr { dst: v0, src: v1 });

            block_data.push(X64Inst::Jmp { dst: b1 });
        }

        let usedefs = UseDefs::compute(&func);

        assert_eq!(
            usedefs.get_uses(b0).iter_ones().collect::<Vec<_>>(),
            vec![v1 as usize]
        );
        assert_eq!(
            usedefs.get_defs(b0).iter_ones().collect::<Vec<_>>(),
            vec![v0 as usize]
        );

        assert_eq!(
            usedefs.get_uses(b1).iter_ones().collect::<Vec<_>>(),
            vec![v0 as usize]
        );
        assert_eq!(
            usedefs.get_defs(b1).iter_ones().collect::<Vec<_>>(),
            vec![v1 as usize]
        );
    }


    #[test]
    fn simple_test() {
        // foo:
        // @0
        //     mov v0 v1
        //     jmp @1
        // @1
        //     mov v1 v0
        //     ret v1
        let mut func = Func::<X64Inst>::new("foo".to_string());

        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();

        let b1 = {
            let mut block_data = BlockData::new();

            block_data.push(X64Inst::Mov64rr { dst: v1, src: v0 });
            block_data.push(X64Inst::Ret { src: v1 });

            func.add_block(block_data)
        };

        {
            let block_data = func.get_block_data_mut(b0);
            block_data.push(X64Inst::Mov64rr { dst: v0, src: v1 });

            block_data.push(X64Inst::Jmp { dst: b1 });
        }

        let cfg = CFG::compute(&func).unwrap();

        let live_ranges = LiveRanges::compute(&func, &cfg);


        let v0_ranges = &live_ranges[v0];
        let rax_ranges = &live_ranges[v1];

        assert_eq!(
            rax_ranges,
            [LiveRange {
                start: ProgramPoint {
                    block: b0,
                    inst_index: 0,
                },
                end: ProgramPoint {
                    block: b0,
                    inst_index: 0,
                },
            },
                LiveRange {
                    start: ProgramPoint {
                        block: b1,
                        inst_index: 0,
                    },
                    end: ProgramPoint {
                        block: b1,
                        inst_index: 1,
                    },
                }]
        );

        assert_eq!(
            v0_ranges,
            [LiveRange {
                start: ProgramPoint {
                    block: b0,
                    inst_index: 0,
                },
                end: ProgramPoint {
                    block: b1,
                    inst_index: 0,
                },
            }]
        );
    }

    #[test]
    fn test_loop() {
        // foo:
        // @0
        //     mov v0 v2
        //     jmp @1
        // @1
        //     mov v1 v0
        //     jmp @2
        // @2
        //     mov v2 v1
        //     jmp @0

        let mut func = Func::<X64Inst>::new("foo".to_string());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();

        let b1 = func.add_empty_block();
        let b2 = func.add_empty_block();

        {
            let block_data = func.get_block_data_mut(b0);
            block_data.push(X64Inst::Mov64rr { dst: v0, src: v2 });
            block_data.push(X64Inst::Jmp { dst: b1 });
        }

        {
            let block_data = func.get_block_data_mut(b1);
            block_data.push(X64Inst::Mov64rr { dst: v1, src: v0 });
            block_data.push(X64Inst::Jmp { dst: b2 });
        }

        {
            let block_data = func.get_block_data_mut(b2);
            block_data.push(X64Inst::Mov64rr { dst: v2, src: v1 });
            block_data.push(X64Inst::Jmp { dst: b0 });
        }

        let cfg = CFG::compute(&func).unwrap();
        let live_ranges = LiveRanges::compute(&func, &cfg);
        let v0_ranges = &live_ranges[v0];
        let v1_ranges = &live_ranges[v1];
        let v2_ranges = &live_ranges[v2];

        let liveness = LivenessAnalysis::compute(&func, &cfg);

        assert_eq!(
            liveness.live_in[b0].iter_ones().collect::<Vec<_>>(),
            vec![v2 as usize]
        );

        assert_eq!(
            liveness.live_out[b2].iter_ones().collect::<Vec<_>>(),
            vec![v2 as usize]
        );

        assert_eq!(
            v2_ranges,
            [LiveRange {
                start: ProgramPoint {
                    block: b0,
                    inst_index: 0,
                },
                end: ProgramPoint {
                    block: b2,
                    inst_index: 2,
                },
            }]
        );

        assert_eq!(
            v0_ranges,
            [LiveRange {
                start: ProgramPoint {
                    block: b0,
                    inst_index: 0,
                },
                end: ProgramPoint {
                    block: b1,
                    inst_index: 0,
                },
            }]
        );

        assert_eq!(
            v1_ranges,
            [LiveRange {
                start: ProgramPoint {
                    block: b1,
                    inst_index: 0,
                },
                end: ProgramPoint {
                    block: b2,
                    inst_index: 0,
                },
            }]
        );
    }
}
