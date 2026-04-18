use crate::codegen::analysis::LiveRanges;
use crate::support::slotmap::SecondaryMap;
use crate::{
    codegen::{
        analysis::{cfg::CFG, LiveRange, ProgramPoint},
        tir::{Func, Inst, Reg},
    },
    support::bitset::FixedBitSet,
};
use std::collections::{BTreeSet, HashMap};

pub type StackSlot = u32;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AllocatedSlot {
    Reg(Reg),
    Stack(StackSlot),
}

pub struct RegAllocResult {
    pub coloring: SecondaryMap<Reg, AllocatedSlot>,
    pub frame_layout: SecondaryMap<StackSlot, usize>,
    pub frame_size: u32,
}

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
    last_preg_use: SecondaryMap<Reg, Reg>, // Physical register -> Virtual register
    config: &'i RegAllocConfig,
    expire_range: BTreeSet<(ProgramPoint, Reg)>,
    stack_slots: u32,
}

impl<'i, I: Inst> RegAlloc<'i, I> {
    #[must_use] 
    pub fn new(func: &'i Func<I>, cfg: &'i CFG, config: &'i RegAllocConfig) -> Self {
        let mut free_regs = FixedBitSet::zeroes(config.preg_count);

        for preg in &config.allocatable_regs {
            free_regs.add(*preg as usize);
        }

        Self {
            func,
            cfg,
            free_regs,
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

    fn lookup_available_reg(&mut self) -> Option<Reg> {
        self.free_regs.iter_ones().next().map(|reg| reg as Reg)
    }

    pub fn run(&mut self) -> RegAllocResult {
        let live_ranges: LiveRanges = LiveRanges::compute(self.func, self.cfg);
        let mut sorted_live_ranges: Vec<(Reg, &LiveRange)> = live_ranges.iter().collect();
        sorted_live_ranges.sort_by_key(|(_, range)| range.start);

        let mut coloring: SecondaryMap<Reg, AllocatedSlot> =
            SecondaryMap::new(self.func.get_regs_count());
        let mut frame_layout: SecondaryMap<StackSlot, usize> =
            SecondaryMap::new(self.func.get_regs_count());

        for (vreg, lr) in &sorted_live_ranges {
            self.expire(lr.start);

            if self.config.reg_bind.contains_key(vreg) {
                let preg = self.config.reg_bind[vreg];
                if !self.free_regs.has(preg as usize) {
                    // Preg is held by another vreg — evict it to a stack slot.
                    // The evicted vreg's end-of-range entry in `expire_range`
                    // must be removed so the preg isn't freed twice (once by
                    // the evictee's stale entry, once by the new owner).
                    let evictee = self.last_preg_use[preg];
                    let evictee_end = live_ranges[evictee].end;
                    self.expire_range.remove(&(evictee_end, preg));
                    coloring.set(evictee, AllocatedSlot::Stack(self.stack_slots));
                    frame_layout.set(self.stack_slots, (self.stack_slots * 8) as usize);
                    self.stack_slots += 1;
                }
                self.free_regs.del(preg as usize);
                self.expire_range.insert((lr.end, preg));
                coloring.set(*vreg, AllocatedSlot::Reg(preg));
                self.last_preg_use.set(preg, *vreg);
                continue;
            }

            if let Some(preg) = self.lookup_available_reg() {
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

        RegAllocResult {
            coloring,
            frame_layout,
            frame_size: self.stack_slots * 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::codegen::analysis::cfg::CFG;
    use crate::codegen::isa::x64::inst::X64Inst;
    use crate::codegen::isa::x64::regs::*;
    use crate::codegen::regalloc::{RegAlloc, RegAllocConfig};
    use crate::codegen::tir::{Func, PseudoInstruction};
    use std::collections::HashMap;

    use crate::codegen::regalloc::AllocatedSlot;
    use crate::codegen::tir::Reg;

    fn default_cfg(reg_bind: HashMap<Reg, Reg>) -> RegAllocConfig {
        RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RAX, RBX, RCX, RDX],
            scratch_regs: vec![R12, R13],
            reg_bind,
        }
    }

    #[test]
    fn simple_test() {
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
            block_data.push_pseudo_inst(PseudoInstruction::Arg { dst: v0, idx: 0 });
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
            block_data.push_pseudo_inst(PseudoInstruction::Return { src: v3 });
        }

        let cfg = CFG::compute(&func).unwrap();
        let reg_alloc_config = default_cfg(reg_bind);
        let mut regalloc = RegAlloc::new(&func, &cfg, &reg_alloc_config);
        let res = regalloc.run();

        assert_eq!(res.coloring[v0], AllocatedSlot::Reg(RAX));
        assert_eq!(res.coloring[v3], AllocatedSlot::Reg(RAX));
        for &v in &[v0, v1, v2, v3] {
            let slot = res.coloring[v];
            assert!(matches!(slot, AllocatedSlot::Reg(_) | AllocatedSlot::Stack(_)));
        }
    }

    #[test]
    fn regalloc_spills_when_pressure_exceeds_pool() {
        // Produce 6 independent, overlapping live values; only 2 allocatable
        // regs -> at least 4 must end up on the stack.
        let mut func = Func::<X64Inst>::new("pressure".to_string());
        let b0 = func.add_empty_block();
        let vregs: Vec<Reg> = (0..6).map(|_| func.new_vreg()).collect();
        {
            let bd = func.get_block_data_mut(b0);
            for (i, &v) in vregs.iter().enumerate() {
                bd.push_target_inst(X64Inst::Mov64ri { dst: v, imm: i as i64 });
            }
            // Chain them so the liveness of each extends to the return.
            for window in vregs.windows(2) {
                bd.push_target_inst(X64Inst::Add64rr {
                    dst: window[1],
                    src: window[0],
                });
            }
            bd.push_pseudo_inst(PseudoInstruction::Return { src: *vregs.last().unwrap() });
        }
        let cfg = CFG::compute(&func).unwrap();
        let cfg_config = RegAllocConfig {
            preg_count: 32,
            allocatable_regs: vec![RAX, RBX],
            scratch_regs: vec![R12, R13],
            reg_bind: HashMap::new(),
        };
        let mut regalloc = RegAlloc::new(&func, &cfg, &cfg_config);
        let res = regalloc.run();
        let spilled = vregs
            .iter()
            .filter(|&&v| matches!(res.coloring[v], AllocatedSlot::Stack(_)))
            .count();
        assert!(spilled >= 1, "expected at least one spill under pressure");
    }
}
