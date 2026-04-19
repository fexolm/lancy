//! Erases `MakeAggregate` / `InsertValue` / `ExtractValue` pseudos
//! by tracking each aggregate vreg's current element list and
//! rewriting `ExtractValue` into a scalar `Copy`. Must run before
//! regalloc — aggregate vregs carry no machine value.

use std::collections::HashMap;

use crate::codegen::tir::{Func, Inst, Instruction, PseudoInstruction, Reg};

/// Lower aggregate pseudos in place. See module docs for the contract.
pub fn lower_aggregates<I: Inst>(func: &mut Func<I>) {
    if !func.has_aggregates() {
        return;
    }
    // Working map: current element list per aggregate vreg.
    let mut elems: HashMap<Reg, Vec<Reg>> = HashMap::new();

    let blocks: Vec<_> = func.blocks_iter().map(|(b, _)| b).collect();
    for block in blocks {
        let old = func.get_block_data_mut(block).take_insts();
        let mut new: Vec<Instruction<I>> = Vec::with_capacity(old.len());
        for inst in old {
            match inst {
                Instruction::Pseudo(PseudoInstruction::MakeAggregate { dst, id }) => {
                    let v = func.aggregate_operands(id).elems.clone();
                    elems.insert(dst, v);
                    // Erase — aggregate handles have no machine value.
                }
                Instruction::Pseudo(PseudoInstruction::InsertValue {
                    dst,
                    agg,
                    val,
                    idx,
                }) => {
                    let mut v = elems
                        .get(&agg)
                        .cloned()
                        .expect("InsertValue on unknown aggregate vreg");
                    let i = idx as usize;
                    assert!(
                        i < v.len(),
                        "InsertValue index {i} out of bounds (aggregate has {} elements)",
                        v.len()
                    );
                    v[i] = val;
                    elems.insert(dst, v);
                }
                Instruction::Pseudo(PseudoInstruction::ExtractValue { dst, agg, idx }) => {
                    let v = elems
                        .get(&agg)
                        .expect("ExtractValue on unknown aggregate vreg");
                    let i = idx as usize;
                    assert!(
                        i < v.len(),
                        "ExtractValue index {i} out of bounds (aggregate has {} elements)",
                        v.len()
                    );
                    let src = v[i];
                    new.push(Instruction::Pseudo(PseudoInstruction::Copy { dst, src }));
                }
                other => new.push(other),
            }
        }
        func.get_block_data_mut(block).set_insts(new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::isa::x64::inst::X64Inst;

    #[test]
    fn extract_lowers_to_copy() {
        let mut func = Func::<X64Inst>::new("t".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let agg = func.new_vreg();
        let id = func.new_aggregate(vec![v0, v1]);
        let extracted = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::MakeAggregate { dst: agg, id });
            bd.push_pseudo_inst(PseudoInstruction::ExtractValue {
                dst: extracted,
                agg,
                idx: 1,
            });
        }

        lower_aggregates(&mut func);

        let insts: Vec<_> = func.get_block_data(b0).iter().copied().collect();
        // MakeAggregate erased; ExtractValue → Copy from v1.
        assert_eq!(insts.len(), 1);
        match insts[0] {
            Instruction::Pseudo(PseudoInstruction::Copy { dst, src }) => {
                assert_eq!(dst, extracted);
                assert_eq!(src, v1);
            }
            other => panic!("expected Copy, got {other:?}"),
        }
    }

    #[test]
    fn insert_replaces_element_in_fresh_aggregate() {
        // agg0 = {v0, v1}; agg1 = insertvalue agg0[0] <- v2; x = extract agg1[0]
        // => x copies v2 (not v0).
        let mut func = Func::<X64Inst>::new("t".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let agg0 = func.new_vreg();
        let agg1 = func.new_vreg();
        let extracted = func.new_vreg();
        let id = func.new_aggregate(vec![v0, v1]);
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::MakeAggregate { dst: agg0, id });
            bd.push_pseudo_inst(PseudoInstruction::InsertValue {
                dst: agg1,
                agg: agg0,
                val: v2,
                idx: 0,
            });
            bd.push_pseudo_inst(PseudoInstruction::ExtractValue {
                dst: extracted,
                agg: agg1,
                idx: 0,
            });
        }

        lower_aggregates(&mut func);

        let insts: Vec<_> = func.get_block_data(b0).iter().copied().collect();
        assert_eq!(insts.len(), 1);
        match insts[0] {
            Instruction::Pseudo(PseudoInstruction::Copy { dst, src }) => {
                assert_eq!(dst, extracted);
                assert_eq!(src, v2);
            }
            other => panic!("expected Copy, got {other:?}"),
        }
    }

    #[test]
    fn chained_insert_follows_latest_replacement_per_index() {
        // agg0 = {v0, v1}
        // agg1 = insertvalue agg0[0] <- v2   // now {v2, v1}
        // agg2 = insertvalue agg1[0] <- v3   // now {v3, v1}
        // extract agg2[0] should copy v3 (most recent write wins), not v2 or v0.
        let mut func = Func::<X64Inst>::new("chain".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();
        let agg0 = func.new_vreg();
        let agg1 = func.new_vreg();
        let agg2 = func.new_vreg();
        let extracted = func.new_vreg();
        let id = func.new_aggregate(vec![v0, v1]);
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::MakeAggregate { dst: agg0, id });
            bd.push_pseudo_inst(PseudoInstruction::InsertValue {
                dst: agg1,
                agg: agg0,
                val: v2,
                idx: 0,
            });
            bd.push_pseudo_inst(PseudoInstruction::InsertValue {
                dst: agg2,
                agg: agg1,
                val: v3,
                idx: 0,
            });
            bd.push_pseudo_inst(PseudoInstruction::ExtractValue {
                dst: extracted,
                agg: agg2,
                idx: 0,
            });
        }

        lower_aggregates(&mut func);

        let insts: Vec<_> = func.get_block_data(b0).iter().copied().collect();
        assert_eq!(insts.len(), 1);
        match insts[0] {
            Instruction::Pseudo(PseudoInstruction::Copy { dst, src }) => {
                assert_eq!(dst, extracted);
                assert_eq!(
                    src, v3,
                    "latest insertvalue wins at index 0"
                );
            }
            other => panic!("expected Copy, got {other:?}"),
        }
    }

    #[test]
    fn insert_preserves_unchanged_elements() {
        // agg0 = {v0, v1, v2}; agg1 = insertvalue agg0[1] <- v3;
        // extract agg1[2] should still copy v2.
        let mut func = Func::<X64Inst>::new("t".into());
        let b0 = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        let v2 = func.new_vreg();
        let v3 = func.new_vreg();
        let agg0 = func.new_vreg();
        let agg1 = func.new_vreg();
        let extracted = func.new_vreg();
        let id = func.new_aggregate(vec![v0, v1, v2]);
        {
            let bd = func.get_block_data_mut(b0);
            bd.push_pseudo_inst(PseudoInstruction::MakeAggregate { dst: agg0, id });
            bd.push_pseudo_inst(PseudoInstruction::InsertValue {
                dst: agg1,
                agg: agg0,
                val: v3,
                idx: 1,
            });
            bd.push_pseudo_inst(PseudoInstruction::ExtractValue {
                dst: extracted,
                agg: agg1,
                idx: 2,
            });
        }

        lower_aggregates(&mut func);

        let insts: Vec<_> = func.get_block_data(b0).iter().copied().collect();
        match insts[0] {
            Instruction::Pseudo(PseudoInstruction::Copy { src, .. }) => {
                assert_eq!(src, v2);
            }
            other => panic!("expected Copy, got {other:?}"),
        }
    }
}
