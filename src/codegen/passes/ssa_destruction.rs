//! SSA destruction with critical-edge splitting.
//!
//! **Requires:** SSA-form target IR with `PseudoInstruction::Phi { dst, id }`
//! in blocks that merge multiple incoming edges. Variable-length
//! `(Block, Reg)` incoming lists live at `Func::phi_operands(id)`.
//!
//! **Preserves:** CFG reachability (not block identity — critical edges
//! get a fresh intermediate block).
//!
//! **Invalidates:** The `Phi` pseudo — all phi instructions are removed
//! after this pass. SSA no longer holds: a vreg may be defined in
//! multiple predecessors.
//!
//! **Effect:** For each phi `dst = phi [(pred_i, src_i)...]`:
//!
//! * If the edge `pred_i → target` is a *critical edge* (pred has
//!   multiple successors **and** target has multiple predecessors),
//!   insert a fresh intermediate "landing" block between them. Rewrite
//!   `pred`'s terminator to branch into the landing block, and make the
//!   landing block `jmp target`.
//! * Emit the phi-materializing Copies at the end of the insertion
//!   block (either `pred` itself, if the edge isn't critical, or the
//!   freshly-created landing block). To avoid stomping on a
//!   simultaneously-read source, the copies first stage every incoming
//!   source into a fresh temp and then move each temp into its final
//!   destination.

use std::collections::HashMap;

use crate::codegen::tir::{Block, Func, Inst, Instruction, PseudoInstruction, Reg};

/// A single phi's state after being stripped from its block: the vreg
/// it defined and the list of `(predecessor, incoming_source)` pairs.
type StrippedPhi = (Reg, Vec<(Block, Reg)>);

/// Destroy SSA in place. See module docs for the contract.
pub fn destroy_ssa<I: Inst>(func: &mut Func<I>) {
    // ---- Phase 1: strip `Phi` pseudos from every block, remembering
    // per-block phi headers so we can synthesize copies later. ----
    let blocks: Vec<Block> = func.blocks_iter().map(|(b, _)| b).collect();
    let mut phi_headers: HashMap<Block, Vec<StrippedPhi>> = HashMap::new();
    for b in &blocks {
        let insts = func.get_block_data_mut(*b).take_insts();
        let mut kept = Vec::with_capacity(insts.len());
        let mut here: Vec<StrippedPhi> = Vec::new();
        for inst in insts {
            if let Instruction::Pseudo(PseudoInstruction::Phi { dst, id }) = inst {
                let incoming = func.phi_operands(id).incoming.clone();
                here.push((dst, incoming));
            } else {
                kept.push(inst);
            }
        }
        func.get_block_data_mut(*b).set_insts(kept);
        if !here.is_empty() {
            phi_headers.insert(*b, here);
        }
    }

    if phi_headers.is_empty() {
        return;
    }

    // ---- Phase 2: resolve each (pred, target) edge that feeds a phi
    // into its insertion block (pred for non-critical edges, a freshly
    // created landing block for critical edges). ----
    let mut per_landing: HashMap<Block, Vec<(Reg, Reg)>> = HashMap::new();

    // Successor count per block. A conditional jump with both arms to
    // the same target still counts as one successor.
    let mut succ_count: HashMap<Block, usize> = HashMap::new();
    for b in &blocks {
        let bd = func.get_block_data(*b);
        let count = bd.get_terminator().map_or(0, |term| {
            let t = term.get_branch_targets();
            match t.as_slice() {
                [] => 0,
                [_] => 1,
                [x, y] if x == y => 1,
                _ => t.len(),
            }
        });
        succ_count.insert(*b, count);
    }

    for (&target, phis) in &phi_headers {
        // Every phi of `target` lists the same predecessor set (the
        // target's CFG preds), so take them from the first phi.
        let uniq_preds: Vec<Block> = phis[0].1.iter().map(|(p, _)| *p).collect();

        for pred in uniq_preds {
            let pred_has_multi_succ = succ_count.get(&pred).copied().unwrap_or(0) > 1;
            // Target always has >=2 preds here (otherwise we wouldn't
            // need phis), so the edge is critical iff pred has >1 succ.
            let insertion_block = if pred_has_multi_succ {
                let landing = func.add_empty_block();
                if let Some(last) = func.get_block_data_mut(pred).insts_mut().last_mut()
                    && last.is_term()
                {
                    last.rewrite_branch_target(target, landing);
                }
                func.get_block_data_mut(landing)
                    .push_inst(Instruction::new_jmp(target));
                succ_count.insert(landing, 1);
                landing
            } else {
                pred
            };
            for (dst, incoming) in phis {
                let src = incoming
                    .iter()
                    .find(|(p, _)| *p == pred)
                    .map(|(_, s)| *s)
                    .expect("phi must list every predecessor exactly once");
                per_landing
                    .entry(insertion_block)
                    .or_default()
                    .push((*dst, src));
            }
        }
    }

    // ---- Phase 3: emit staged copies in each insertion block ----
    for (insertion, pairs) in per_landing {
        // Pre-allocate temps so borrows don't alias with block mutation.
        let temps: Vec<Reg> = (0..pairs.len()).map(|_| func.new_vreg()).collect();
        let insts = func.get_block_data_mut(insertion).insts_mut();
        let insert_at = insts
            .iter()
            .rposition(Inst::is_term)
            .unwrap_or(insts.len());
        let mut prelude: Vec<Instruction<I>> = Vec::with_capacity(pairs.len() * 2);
        for (i, (_dst, src)) in pairs.iter().enumerate() {
            prelude.push(Instruction::Pseudo(PseudoInstruction::Copy {
                dst: temps[i],
                src: *src,
            }));
        }
        for (i, (dst, _src)) in pairs.iter().enumerate() {
            prelude.push(Instruction::Pseudo(PseudoInstruction::Copy {
                dst: *dst,
                src: temps[i],
            }));
        }
        insts.splice(insert_at..insert_at, prelude);
    }
}

