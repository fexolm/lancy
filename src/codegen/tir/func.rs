use std::collections::HashMap;
use std::fmt::Display;

use crate::support::slotmap::{Key, PrimaryMap};

use super::{Block, BlockData, CallData, CallId, Inst, PhiData, PhiId};

pub type Reg = u32;

pub struct Func<I: Inst> {
    name: String,
    blocks: PrimaryMap<Block, BlockData<I>>,
    phis: PrimaryMap<PhiId, PhiData>,
    calls: PrimaryMap<CallId, CallData>,
    regs_count: u32,
    /// Frontend-declared pre-bindings: `vreg -> preg` pins that the
    /// allocator must honor for the vreg's whole life. Typically used
    /// for ABI-visible shims (arg/ret registers, IDIV's RAX/RDX, shift
    /// counts in RCX, etc.) that the frontend emits before ABI
    /// lowering. The pipeline merges these with `AbiLowerResult::reg_bind`
    /// before handing the config to the regalloc.
    pre_binds: HashMap<Reg, Reg>,
}

impl<I: Inst> Func<I> {
    #[must_use]
    pub fn new(name: String) -> Self {
        Func {
            name,
            regs_count: 0,
            blocks: PrimaryMap::new(),
            phis: PrimaryMap::new(),
            calls: PrimaryMap::new(),
            pre_binds: HashMap::new(),
        }
    }

    pub fn add_block(&mut self, data: BlockData<I>) -> Block {
        self.blocks.insert(data)
    }

    pub fn add_empty_block(&mut self) -> Block {
        self.blocks.insert(BlockData::default())
    }

    pub fn get_block_data_mut(&mut self, block: Block) -> &mut BlockData<I> {
        &mut self.blocks[block]
    }

    #[must_use]
    pub fn get_block_data(&self, block: Block) -> &BlockData<I> {
        &self.blocks[block]
    }

    pub fn new_vreg(&mut self) -> Reg {
        let res = self.regs_count;
        self.regs_count += 1;
        res
    }

    #[must_use]
    pub fn get_regs_count(&self) -> usize {
        self.regs_count as usize
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn get_entry_block(&self) -> Option<Block> {
        if self.blocks.is_empty() {
            None
        } else {
            Some(Block::new(0))
        }
    }

    pub fn blocks_iter(&self) -> impl Iterator<Item=(Block, &BlockData<I>)> {
        self.blocks.iter()
    }

    #[must_use]
    pub fn blocks_count(&self) -> usize {
        self.blocks.len()
    }

    /// Register a phi node's incoming operands and return an opaque id
    /// to stamp into `PseudoInstruction::Phi { id }`.
    pub fn new_phi(&mut self, incoming: Vec<(Block, Reg)>) -> PhiId {
        self.phis.insert(PhiData { incoming })
    }

    #[must_use]
    pub fn phi_operands(&self, id: PhiId) -> &PhiData {
        &self.phis[id]
    }

    pub fn phi_operands_mut(&mut self, id: PhiId) -> &mut PhiData {
        &mut self.phis[id]
    }

    /// Register a call's callee / args / rets and return an id to
    /// stamp into `PseudoInstruction::CallPseudo { id }`.
    pub fn new_call(&mut self, data: CallData) -> CallId {
        self.calls.insert(data)
    }

    #[must_use]
    pub fn call_operands(&self, id: CallId) -> &CallData {
        &self.calls[id]
    }

    pub fn call_operands_mut(&mut self, id: CallId) -> &mut CallData {
        &mut self.calls[id]
    }

    /// Declare a frontend-level pre-bind: `vreg` must occupy physical
    /// register `preg` for its entire live range. Disagreement with any
    /// later source (ABI lowering, `RegDef` pseudo) triggers the
    /// allocator's pre-bind conflict check.
    pub fn pre_bind(&mut self, vreg: Reg, preg: Reg) {
        if let Some(prev) = self.pre_binds.insert(vreg, preg)
            && prev != preg
        {
            panic!(
                "vreg {vreg} pre-bound to two different pregs: {prev} vs {preg}"
            );
        }
    }

    #[must_use]
    pub fn pre_binds(&self) -> &HashMap<Reg, Reg> {
        &self.pre_binds
    }
}

impl<I: Inst> Display for Func<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}:", self.name)?;

        for (id, data) in self.blocks.iter() {
            write!(f, "{id}")?;
            write!(f, "\n{data}")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::isa::x64::inst::X64Inst;
    use crate::codegen::tir::CallData;
    use crate::codegen::tir::CallTarget;

    #[test]
    fn new_phi_round_trips_incoming_edges() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b0 = func.add_empty_block();
        let b1 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let id = func.new_phi(vec![(b0, v0), (b1, v1)]);
        let data = func.phi_operands(id);
        assert_eq!(data.incoming, vec![(b0, v0), (b1, v1)]);
    }

    #[test]
    fn distinct_phis_have_distinct_ids_and_operands() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let id_a = func.new_phi(vec![(b, v0)]);
        let id_b = func.new_phi(vec![(b, v1)]);
        assert_ne!(id_a, id_b);
        assert_eq!(func.phi_operands(id_a).incoming, vec![(b, v0)]);
        assert_eq!(func.phi_operands(id_b).incoming, vec![(b, v1)]);
    }

    #[test]
    fn phi_operands_mut_allows_editing_in_place() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let id = func.new_phi(vec![(b, v0)]);
        func.phi_operands_mut(id).incoming.push((b, v1));
        assert_eq!(func.phi_operands(id).incoming.len(), 2);
    }

    #[test]
    fn new_call_round_trips_symbol_and_args() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let ret = func.new_vreg();
        let id = func.new_call(CallData {
            callee: CallTarget::Symbol("puts".to_string()),
            args: vec![v0, v1],
            rets: vec![ret],
        });
        let data = func.call_operands(id);
        assert!(matches!(&data.callee, CallTarget::Symbol(s) if s == "puts"));
        assert_eq!(data.args, vec![v0, v1]);
        assert_eq!(data.rets, vec![ret]);
    }

    #[test]
    fn new_call_round_trips_indirect_target() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let fn_ptr = func.new_vreg();
        let id = func.new_call(CallData {
            callee: CallTarget::Indirect(fn_ptr),
            args: Vec::new(),
            rets: Vec::new(),
        });
        match &func.call_operands(id).callee {
            CallTarget::Indirect(r) => assert_eq!(*r, fn_ptr),
            CallTarget::Symbol(_) => panic!("expected indirect callee"),
        }
    }
}
