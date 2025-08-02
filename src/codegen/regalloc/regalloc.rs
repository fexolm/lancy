use std::collections::{BTreeSet, LinkedList};

use crate::{
    codegen::{
        analysis::{LiveRange, LivenessAnalysis, ProgramPoint},
        tir::{Block, BlockData, CFG, Func, Inst, Reg},
    },
    support::{
        bitset::FixedBitSet,
        slotmap::{Key, SecondaryMap, SecondaryMapExt},
    },
};

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AllocatedSlot {
    Reg(Reg),
    Stack(u32),
}

#[derive(Debug, PartialEq)]
pub struct RegAllocResult {
    pub range: LiveRange,
    pub allocated_slot: AllocatedSlot,
}

pub struct RegAlloc<'i, I: Inst> {
    func: &'i Func<I>,
    cfg: &'i CFG,
    liveness: &'i LivenessAnalysis,
    active: FixedBitSet,
    expire_range: BTreeSet<(ProgramPoint, Reg)>,
    stack_slots: u32,
}

impl<'i, I: Inst> RegAlloc<'i, I> {
    pub fn new(func: &'i Func<I>, cfg: &'i CFG, liveness: &'i LivenessAnalysis) -> Self {
        Self {
            func,
            cfg,
            liveness,
            active: FixedBitSet::zeroes(I::preg_count() as usize),
            expire_range: BTreeSet::new(),
            stack_slots: 0,
        }
    }

    fn expire(&mut self, p: ProgramPoint) {
        while let Some((end, reg)) = self.expire_range.first() {
            if *end < p {
                self.active.del(*reg as usize);
                self.expire_range.pop_first();
            } else {
                return;
            }
        }
    }

    fn lookup_available_reg(&mut self, cur: ProgramPoint) -> Option<Reg> {
        let active_copy: FixedBitSet = self.active.clone();
        'outer: for r in active_copy.iter_zeroes() {
            for range in self.liveness.get_live_ranges_for(r as Reg) {
                if range.start <= cur && cur <= range.end {
                    self.active.add(r);
                    self.expire_range.insert((cur, r as Reg));
                    continue 'outer;
                }
            }
            return Some(r as Reg);
        }

        None
    }

    pub fn run(&mut self) -> Vec<RegAllocResult> {
        let mut res: Vec<RegAllocResult> = Vec::new();
        let live_ranges = self.liveness.get_vreg_live_ranges();

        for lr in live_ranges {
            self.expire(lr.start);

            if let Some(reg) = self.lookup_available_reg(lr.start) {
                res.push(RegAllocResult {
                    range: lr,
                    allocated_slot: AllocatedSlot::Reg(reg),
                });
            } else {
                res.push(RegAllocResult {
                    range: lr,
                    allocated_slot: AllocatedSlot::Stack(self.stack_slots),
                });
                self.stack_slots += 1;
            }
        }

        res
    }
}

pub fn apply_regalloc_result<I: Inst>(func: &mut Func<I>, mut ra_intervals: Vec<RegAllocResult>) {
    let mut slots = Vec::new();
    let mut new_blocks = SecondaryMap::with_default(func.blocks_count());
    slots.resize(func.get_regs_count() - I::preg_count() as usize, None);

    ra_intervals.sort_by_key(|i| i.range.start);
    ra_intervals.reverse();

    for (block, data) in func.blocks_iter() {
        let mut new_block = BlockData::new();
        for (idx, &i) in data.iter().enumerate() {
            let p = ProgramPoint {
                block: block,
                inst_index: idx as u32,
            };

            while let Some(interval) = ra_intervals.last() {
                if interval.range.start <= p && interval.range.end >= p {
                    slots[interval.range.reg as usize - I::preg_count() as usize] =
                        Some(interval.allocated_slot);
                    ra_intervals.pop();
                } else {
                    break;
                }
            }

            let mut new_inst = i;
            let defs = i.get_defs();
            let uses = i.get_uses();
            for &old in defs.iter().chain(uses.iter()) {
                if old >= I::preg_count() {
                    if let Some(AllocatedSlot::Reg(new)) =
                        slots[old as usize - I::preg_count() as usize]
                    {
                        new_inst = new_inst.replace(old, new)
                    } else {
                        todo!();
                    }
                }
            }

            new_block.push(new_inst);
        }
        new_blocks[block] = new_block;
    }

    let blocks_count = func.blocks_count();
    for b in 0..blocks_count {
        let b = Block::new(b);
        *func.get_block_data_mut(b) = new_blocks[b].clone();
    }
}

#[cfg(test)]
mod tests {
    use crate::codegen::{
        analysis::{LiveRange, LivenessAnalysis, ProgramPoint},
        isa::x64::{
            inst::X64Inst,
            regs::{RAX, RBX},
        },
        regalloc::{AllocatedSlot, RegAlloc, RegAllocResult},
        tir::{BlockData, Func},
    };

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

        func.construct_cfg().unwrap();
        let analysis = LivenessAnalysis::new(&func, &func.get_cfg());

        let mut regalloc = RegAlloc::new(&func, &func.get_cfg(), &analysis);

        assert_eq!(
            regalloc.run(),
            vec![RegAllocResult {
                range: LiveRange {
                    reg: v0,
                    start: ProgramPoint {
                        block: b0,
                        inst_index: 0
                    },
                    end: ProgramPoint {
                        block: b1,
                        inst_index: 0
                    }
                },
                allocated_slot: AllocatedSlot::Reg(RBX),
            }]
        );
    }
}
