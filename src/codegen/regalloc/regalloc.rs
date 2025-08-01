use crate::{
    codegen::{
        analysis::LivenessAnalysis,
        tir::{CFG, Func, Inst, Reg},
    },
    support::bitset::FixedBitSet,
};

pub struct RegAllocResult {
    pub allocated_registers: Vec<Reg>,
}

pub fn allocate_registers<I: Inst>(func: &Func<I>, cfg: &CFG, liveness: &LivenessAnalysis) {
    // let mut active = FixedBitSet::zeroes(I::preg_count());
}
