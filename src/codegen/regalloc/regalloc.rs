use crate::codegen::analysis::LiveRanges;
use crate::support::slotmap::SecondaryMap;
use crate::{
    codegen::{
        analysis::{LiveRange, ProgramPoint, cfg::CFG},
        tir::{Func, Inst, Reg},
    },
    support::{bitset::FixedBitSet, slotmap::Key},
};
use std::collections::{BTreeSet, HashMap, LinkedList};
use std::ptr::null;
use crate::codegen::tir::Block;

pub type StackSlot = u32;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AllocatedSlot {
    Reg(Reg),
    Stack(StackSlot),
}

pub struct RegAllocResult {
    pub coloring: SecondaryMap<Reg, AllocatedSlot>,
    pub frame_layout: SecondaryMap<StackSlot, usize>
}

impl RegAllocResult {
    pub fn new(
        coloring: SecondaryMap<Reg, AllocatedSlot>,
        frame_layout: SecondaryMap<StackSlot, usize>,
    ) -> Self {
        Self {
            coloring,
            frame_layout,
        }
    }
}

#[derive(Clone)]
pub struct RegAllocConfig {
    pub preg_count: usize,
    pub allocatable_regs: Vec<Reg>,
    pub scratch_regs: Vec<Reg>,
    pub reg_bind: HashMap<Reg, Reg>,
}

pub struct RegAlloc<'i, I: Inst> {
    func: &'i Func<I>,
    cfg: &'i CFG,
    free_regs: FixedBitSet,
    scratch_regs: Vec<Reg>,
    last_preg_use: SecondaryMap<Reg, Reg>, // Physical register -> Virtual register
    config: RegAllocConfig,
    expire_range: BTreeSet<(ProgramPoint, Reg)>,
    stack_slots: u32,
}

impl<'i, I: Inst> RegAlloc<'i, I> {
    pub fn new(func: &'i Func<I>, cfg: &'i CFG, config: RegAllocConfig) -> Self {
        let mut free_regs = FixedBitSet::zeroes(config.preg_count);

        for preg in &config.allocatable_regs {
            free_regs.add(*preg as usize);
        }

        Self {
            func,
            cfg,
            free_regs,
            scratch_regs: config.scratch_regs.clone(),
            last_preg_use: SecondaryMap::new(config.preg_count),
            config,
            expire_range: BTreeSet::new(),
            stack_slots: 0,
        }
    }

    fn expire(&mut self, p: ProgramPoint) {
        while let Some((end, reg)) = self.expire_range.first() {
            if *end < p {
                self.free_regs.add(*reg as usize);
                self.expire_range.pop_first();
            } else {
                return;
            }
        }
    }

    fn lookup_available_reg(&mut self, cur: ProgramPoint) -> Option<Reg> {
        if let Some(reg) = self.free_regs.iter_ones().next() {
            Some(reg as Reg)
        } else {
            None
        }
    }

    pub fn run(&mut self) -> RegAllocResult {
        let live_ranges: LiveRanges = LiveRanges::compute(&self.func, &self.cfg);
        let mut sorted_live_ranges: Vec<(Reg, &LiveRange)> = live_ranges.iter().collect();
        sorted_live_ranges.sort_by_key(|(_, range)| range.start);

        for (key, range) in sorted_live_ranges.iter() {
            println!("{:?}: {:?}..{:?}", key, range.start, range.end);
        }

        let mut coloring: SecondaryMap<Reg, AllocatedSlot> =
            SecondaryMap::new(self.func.get_regs_count()); // (Virtual register, Physical Register)
        let mut frame_layout: SecondaryMap<StackSlot, usize> = SecondaryMap::new(self.func.get_regs_count());
        for (vreg, lr) in sorted_live_ranges.iter() {
            self.expire(lr.start);

            if self.config.reg_bind.contains_key(vreg) {
                let preg = self.config.reg_bind[vreg];
                if !self.free_regs.has(preg as usize) {
                    let spill_reg = self.last_preg_use[preg];
                    coloring.set(spill_reg, AllocatedSlot::Stack(self.stack_slots));
                    frame_layout.set(self.stack_slots, (self.stack_slots * 8) as usize);
                    self.stack_slots += 1;
                }
                self.free_regs.del(preg as usize);
                self.expire_range.insert((lr.end, preg));
                coloring.set(*vreg, AllocatedSlot::Reg(preg));
                self.last_preg_use.set(preg, *vreg);
                continue;
            }

            if let Some(preg) = self.lookup_available_reg(lr.start) {
                self.free_regs.del(preg as usize);
                self.expire_range.insert((lr.end, preg));
                coloring.set(*vreg, AllocatedSlot::Reg(preg));
                self.last_preg_use.set(preg, *vreg);
            } else {
                coloring.set(*vreg, AllocatedSlot::Stack(self.stack_slots));
                frame_layout.set(self.stack_slots, (self.stack_slots * 8) as usize);
                self.stack_slots += 1;
            }
        }

        RegAllocResult::new(coloring, frame_layout)
    }
}

#[cfg(test)]
mod tests {
    use crate::codegen::analysis::cfg::CFG;
    use crate::codegen::isa::x64::inst::X64Inst;
    use crate::codegen::isa::x64::regs::*;
    use crate::codegen::regalloc::{RegAlloc, RegAllocConfig};
    use crate::codegen::tir::{BlockData, Func, PseudoInstruction};
    use std::collections::{HashMap, LinkedList};

    #[test]
    fn simple_test() {
        // foo:
        // @0
        //     arg v0
        //     mov v1 v0
        //     jmp @1
        // @1
        //     mov v2 v1
        //     jmp @2
        // @2
        //     mov v3 v2
        //     ret v3
        let mut func = Func::<X64Inst>::new("foo".to_string());

        let b0 = func.add_empty_block();
        let b1 = func.add_empty_block();
        let b2 = func.add_empty_block();

        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();

        let mut reg_bind = HashMap::new();
        reg_bind.insert(v0, RAX);
        reg_bind.insert(v3, RAX);

        {
            let block_data = func.get_block_data_mut(b0);
            block_data.push_pseudo_inst(PseudoInstruction::Arg { dst: v0 });
            block_data.push_target_inst(X64Inst::Mov64rr { dst: v1, src: v0 });
            block_data.push_target_inst(X64Inst::Jmp { dst: b1 });
        }

        {
            let block_data = func.get_block_data_mut(b1);
            block_data.push_target_inst(X64Inst::Mov64rr { dst: v2, src: v1 });
            block_data.push_target_inst(X64Inst::Jmp { dst: b2 });
        }

        {
            let block_data = func.get_block_data_mut(b2);
            block_data.push_target_inst(X64Inst::Mov64rr { dst: v3, src: v2 });
            block_data.push_target_inst(X64Inst::Ret { src: v3 });
        }

        println!("{func}");

        let cfg = CFG::compute(&func).unwrap();
        let mut allocatable_regs = vec![RAX, RBX, RCX, RDX];
        let mut scratch_regs = vec![R12, R13];
        let reg_alloc_config = RegAllocConfig {
            preg_count: 32,
            allocatable_regs,
            scratch_regs,
            reg_bind,
        };
        let mut regalloc = RegAlloc::new(&func, &cfg, reg_alloc_config);
        let res = regalloc.run();

        for (vir_reg, phys_reg) in res.coloring.iter() {
            println!("v{:?}->{:?}", vir_reg, phys_reg);
        }

        for (stack_slot, offest) in res.frame_layout.iter() {
            println!("slot {:?} offset {:?}", stack_slot, offest);
        }

        assert_eq!(true, true);
    }
}
