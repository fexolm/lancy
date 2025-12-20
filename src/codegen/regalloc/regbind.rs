use crate::codegen::tir::Reg;
use crate::support::slotmap::SecondaryMap;

pub struct RegisterBinding {
    vreg_to_preg: SecondaryMap<Reg, Reg>,
}

impl RegisterBinding {
    pub fn new(reg_count: usize) -> Self {
        let vreg_to_preg = SecondaryMap::new(reg_count);
        Self { vreg_to_preg }
    }
    
    pub fn vreg_to_preg(&self, vreg: Reg) -> Reg  {
        self.vreg_to_preg[vreg]
    }
    
    pub fn add(&mut self, vreg: Reg, preg: Reg) {
        self.vreg_to_preg[vreg] = preg;
    }
}