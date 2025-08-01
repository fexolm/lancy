use std::collections::{BTreeSet, LinkedList};

use crate::{
    codegen::{
        analysis::{LiveRange, LivenessAnalysis, ProgramPoint},
        tir::{CFG, Func, Inst, Reg},
    },
    support::bitset::FixedBitSet,
};

#[derive(Debug, PartialEq)]
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
