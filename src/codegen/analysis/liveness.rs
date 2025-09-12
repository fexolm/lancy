use smallvec::SmallVec;

use crate::{
    codegen::tir::{Block, Func, Inst, Reg, CFG},
    support::{
        bitset::FixedBitSet,
        slotmap::{Key, SecondaryMap, SecondaryMapExt},
    },
};
use crate::codegen::tir::reverse_post_order;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramPoint {
    pub block: Block,
    pub inst_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LiveRange {
    pub reg: Reg,
    pub start: ProgramPoint,
    pub end: ProgramPoint,
}

pub struct LivenessAnalysis {
    live_in: SecondaryMap<Block, FixedBitSet>,
    live_out: SecondaryMap<Block, FixedBitSet>,
    live_ranges: SecondaryMap<u32, Vec<LiveRange>>,
    regs_count: usize,
    pregs_count: usize,
}

struct UseDefs {
    uses: SecondaryMap<Block, FixedBitSet>,
    defs: SecondaryMap<Block, FixedBitSet>,
}

impl UseDefs {
    pub fn compute<I: Inst>(func: &Func<I>) -> Self {
        let mut res = Self {
            uses: SecondaryMap::new(func.blocks_count(), FixedBitSet::zeroes(func.get_regs_count())),
            defs: SecondaryMap::new(func.blocks_count(), FixedBitSet::zeroes(func.get_regs_count())),
        };

        for (b, _) in func.blocks_iter() {
            res.compute_block(b, func);
        }

        res
    }

    fn compute_block<I: Inst>(&mut self, block: Block, func: &Func<I>) {
        for inst in func.get_block_data(block).iter() {
            let uses = inst.get_uses();
            let defs = inst.get_defs();

            let block_defs = &mut self.defs[block];
            let block_uses = &mut self.uses[block];

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
        let pregs_count = I::preg_count() as usize;
        let live_in = SecondaryMap::new(cfg.blocks_count(), FixedBitSet::zeroes(regs_count));
        let live_out = SecondaryMap::new(cfg.blocks_count(), FixedBitSet::zeroes(regs_count));

        let live_ranges = SecondaryMap::with_default(regs_count as usize);

        let mut analysis = Self {
            live_in,
            live_out,
            live_ranges,
            regs_count,
            pregs_count,
        };
        
        analysis.do_compute(func, cfg);
        analysis.compute_live_ranges(func, cfg);

        analysis
    }

    fn do_compute<I: Inst>(&mut self, func: &Func<I>, cfg: &CFG) {
        let mut worklist = reverse_post_order(cfg);
        let mut changed = true;

        let usedefs = UseDefs::compute(func);

        while let Some(block) = worklist.pop() {
            let line_ins_count = self.live_in[block].ones_count();
            let line_outs_count = self.live_out[block].ones_count();

            for &s in cfg.succs(block) {
                self.live_out[block].union(&self.live_in[s]);
            }

            self.live_in[block].union(&self.live_out[block]);
            self.live_in[block].difference(&usedefs.get_defs(block));
            self.live_in[block].union(&usedefs.get_uses(block));

            if self.live_in[block].ones_count() != line_ins_count
                || self.live_out[block].ones_count() != line_outs_count
            {
                worklist.extend_from_slice(cfg.preds(block));
            }
        }
    }

    fn merge_intervals(&mut self, func: &Func<impl Inst>) {
        for i in 0..self.live_ranges.capacity() {
            let mut ranges = &mut self.live_ranges[i as u32];

            if ranges.is_empty() {
                continue;
            }

            ranges.sort();

            let mut merged = Vec::new();
            let mut current = ranges[0];

            for &next in &ranges[1..] {
                let block_data = func.get_block_data(current.end.block);

                if next.start <= current.end
                    || block_data.len() as u32 >= current.end.inst_index
                    && next.start.inst_index == 0
                {
                    current.end = next.end;
                } else {
                    merged.push(current);
                    current = next;
                }
            }
            merged.push(current);
            *ranges = merged;
        }
    }

    fn compute_live_ranges<I: Inst>(&mut self, func: &Func<I>, cfg: &CFG) {
        let mut prev_block_len = 0;

        for (block, block_data) in func.blocks_iter() {
            for r in self.live_in[block].iter_ones() {
                let end = if self.live_out[block].has(r as usize) {
                    block_data.len() as u32
                } else {
                    0
                };

                self.live_ranges[r as u32].push(LiveRange {
                    reg: r as Reg,
                    start: ProgramPoint {
                        block: block,
                        inst_index: 0,
                    },
                    end: ProgramPoint {
                        block: block,
                        inst_index: end,
                    },
                });
            }

            for (inst_index, inst) in block_data.iter().enumerate() {
                let point = ProgramPoint {
                    block,
                    inst_index: inst_index as u32,
                };

                for reg in inst.get_uses() {
                    let last = self.live_ranges[reg].last_mut().unwrap();
                    if last.end < point {
                        last.end = point;
                    }
                }

                for reg in inst.get_defs() {
                    if let Some(last) = self.live_ranges[reg].last() {
                        if last.end >= point {
                            continue; // Already has a range that covers this point
                        }
                    }

                    self.live_ranges[reg].push(LiveRange {
                        reg: reg as Reg,
                        start: point,
                        end: point,
                    });
                }
            }

            for r in self.live_out[block].iter_ones() {
                self.live_ranges[r as u32].last_mut().unwrap().end = ProgramPoint {
                    block: block,
                    inst_index: block_data.len() as u32,
                };
            }

            self.merge_intervals(func);
        }
    }

    pub fn get_live_ranges_for(&self, reg: Reg) -> &[LiveRange] {
        &self.live_ranges[reg]
    }

    pub fn get_vreg_live_ranges(&self) -> Vec<LiveRange> {
        let mut res = Vec::new();
        for r in self.pregs_count..self.regs_count {
            res.extend_from_slice(&self.live_ranges[r as u32]);
        }

        res.sort_by_key(|r| r.start);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{
        isa::x64::{inst::X64Inst, regs::*},
        tir::{BlockData, Func},
    };

    
    #[test]
    fn usedef_test() {
        // foo:
        // @0
        //     mov v0 rax
        //     jmp @1
        // @1
        //     mov rax v0
        //     ret
        let mut func = Func::<X64Inst>::new("foo".to_string());

        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();

        let b1 = {
            let mut block_data = BlockData::new();

            block_data.push(X64Inst::Mov64rr { dst: RAX, src: v0 });
            block_data.push(X64Inst::Ret);

            func.add_block(block_data)
        };

        {
            let block_data = func.get_block_data_mut(b0);
            block_data.push(X64Inst::Mov64rr { dst: v0, src: RAX });

            block_data.push(X64Inst::Jmp { dst: b1 });
        }

        let usedefs = UseDefs::compute(&func);

        assert_eq!(
            usedefs.get_uses(b0).iter_ones().collect::<Vec<_>>(),
            vec![RAX as usize]
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
            vec![RAX as usize]
        );
    }


    #[test]
    fn simple_test() {
        // foo:
        // @0
        //     mov v0 rax
        //     jmp @1
        // @1
        //     mov rax v0
        //     ret
        let mut func = Func::<X64Inst>::new("foo".to_string());

        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();

        let b1 = {
            let mut block_data = BlockData::new();

            block_data.push(X64Inst::Mov64rr { dst: RAX, src: v0 });
            block_data.push(X64Inst::Ret);

            func.add_block(block_data)
        };

        {
            let block_data = func.get_block_data_mut(b0);
            block_data.push(X64Inst::Mov64rr { dst: v0, src: RAX });

            block_data.push(X64Inst::Jmp { dst: b1 });
        }

        let cfg = CFG::compute(&func).unwrap();
        let analysis = LivenessAnalysis::compute(&func, &cfg);

        let v0_ranges = analysis.get_live_ranges_for(v0);
        let rax_ranges = analysis.get_live_ranges_for(RAX);

        assert_eq!(
            rax_ranges,
            [LiveRange {
                reg: RAX,
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
            v0_ranges,
            [LiveRange {
                reg: v0,
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
        //     mov v0 rax
        //     jmp @1
        // @1
        //     mov v1 v0
        //     jmp @2
        // @2
        //     mov rax v1
        //     jmp @0

        let mut func = Func::<X64Inst>::new("foo".to_string());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let b1 = func.add_empty_block();
        let b2 = func.add_empty_block();

        {
            let block_data = func.get_block_data_mut(b0);
            block_data.push(X64Inst::Mov64rr { dst: v0, src: RAX });
            block_data.push(X64Inst::Jmp { dst: b1 });
        }

        {
            let block_data = func.get_block_data_mut(b1);
            block_data.push(X64Inst::Mov64rr { dst: v1, src: v0 });
            block_data.push(X64Inst::Jmp { dst: b2 });
        }

        {
            let block_data = func.get_block_data_mut(b2);
            block_data.push(X64Inst::Mov64rr { dst: RAX, src: v1 });
            block_data.push(X64Inst::Jmp { dst: b0 });
        }

        let cfg = CFG::compute(&func).unwrap();
        let analysis = LivenessAnalysis::compute(&func, &cfg);
        let v0_ranges = analysis.get_live_ranges_for(v0);
        let v1_ranges = analysis.get_live_ranges_for(v1);
        let rax_ranges = analysis.get_live_ranges_for(RAX);

        assert_eq!(
            analysis.live_in[b0].iter_ones().collect::<Vec<_>>(),
            vec![RAX as usize]
        );

        assert_eq!(
            analysis.live_out[b2].iter_ones().collect::<Vec<_>>(),
            vec![RAX as usize]
        );

        assert_eq!(
            rax_ranges,
            [LiveRange {
                reg: RAX,
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
                reg: v0,
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
                reg: v1,
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
