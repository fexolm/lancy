//! Pseudo cleanup pass — runs after regalloc.
//!
//! **Requires:** Regalloc result applied (every vreg has a physical register
//! or stack slot via `RegAllocResult::coloring`). Remaining pseudos are only
//! `Arg` (for pinned-shim defs) and `Copy` (coalescing/move survivors).
//!
//! **Effect:**
//! * `Arg` is erased — by this point the shim is already "in" the ABI reg.
//! * `Copy { dst, src }`:
//!   * If `dst` and `src` are both in registers and coalesced (same preg), erase.
//!   * Otherwise rewrite to `X64Inst::Mov64rr { dst, src }`. The downstream MC
//!     emitter loads/stores to/from a stack slot when either side is spilled.
//! * `Return` must not reach this pass (`abi_lower` converts it to `RawRet`).
//!
//! **Preserves:** CFG shape, terminators, regalloc result.

use crate::codegen::isa::x64::inst::X64Inst;
use crate::codegen::regalloc::{AllocatedSlot, RegAllocResult};
use crate::codegen::tir::{Func, Instruction, PseudoInstruction};

pub fn run(func: &mut Func<X64Inst>, ra: &RegAllocResult) {
    let block_ids: Vec<_> = func.blocks_iter().map(|(b, _)| b).collect();
    for block in block_ids {
        let old = func.get_block_data_mut(block).take_insts();
        let mut new: Vec<Instruction<X64Inst>> = Vec::with_capacity(old.len());
        for inst in old {
            match inst {
                Instruction::Pseudo(PseudoInstruction::Arg { .. }) => {
                    // Pinned shim def is now a no-op: the caller put the value
                    // in the arg register for us.
                }
                Instruction::Pseudo(PseudoInstruction::Copy { dst, src }) => {
                    let coalesced = matches!(
                        (ra.coloring[dst], ra.coloring[src]),
                        (AllocatedSlot::Reg(a), AllocatedSlot::Reg(b)) if a == b
                    );
                    if !coalesced {
                        new.push(Instruction::Target(X64Inst::Mov64rr { dst, src }));
                    }
                }
                Instruction::Pseudo(PseudoInstruction::Return { .. }) => {
                    panic!(
                        "Return pseudo should have been lowered to RawRet before pseudo cleanup"
                    );
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
    use crate::codegen::regalloc::{AllocatedSlot, RegAllocResult};
    use crate::codegen::tir::Reg;
    use crate::support::slotmap::SecondaryMap;

    fn mk_ra(entries: &[(Reg, AllocatedSlot)], cap: usize) -> RegAllocResult {
        let mut coloring = SecondaryMap::new(cap);
        for (v, slot) in entries {
            coloring.set(*v, *slot);
        }
        RegAllocResult {
            coloring,
            frame_layout: SecondaryMap::new(cap),
            frame_size: 0,
        }
    }

    #[test]
    fn copy_is_erased_when_source_and_dest_share_register() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: v1, src: v0 });
            bd.push_target_inst(X64Inst::RawRet);
        }
        let ra = mk_ra(
            &[
                (v0, AllocatedSlot::Reg(0)),
                (v1, AllocatedSlot::Reg(0)),
            ],
            func.get_regs_count(),
        );
        run(&mut func, &ra);
        let insts: Vec<_> = func.get_block_data(b).iter().collect();
        assert_eq!(insts.len(), 1);
        assert!(matches!(insts[0], Instruction::Target(X64Inst::RawRet)));
    }

    #[test]
    fn copy_becomes_mov_when_registers_differ() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let v0 = func.new_vreg();
        let v1 = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Copy { dst: v1, src: v0 });
            bd.push_target_inst(X64Inst::RawRet);
        }
        let ra = mk_ra(
            &[
                (v0, AllocatedSlot::Reg(0)),
                (v1, AllocatedSlot::Reg(1)),
            ],
            func.get_regs_count(),
        );
        run(&mut func, &ra);
        let insts: Vec<_> = func.get_block_data(b).iter().collect();
        assert_eq!(insts.len(), 2);
        assert!(matches!(
            insts[0],
            Instruction::Target(X64Inst::Mov64rr { .. })
        ));
    }

    #[test]
    fn arg_pseudo_is_erased() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let v0 = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Arg { dst: v0, idx: 0 });
            bd.push_target_inst(X64Inst::RawRet);
        }
        let ra = mk_ra(&[(v0, AllocatedSlot::Reg(0))], func.get_regs_count());
        run(&mut func, &ra);
        let insts: Vec<_> = func.get_block_data(b).iter().collect();
        assert_eq!(insts.len(), 1);
        assert!(matches!(insts[0], Instruction::Target(X64Inst::RawRet)));
    }

    #[test]
    #[should_panic(expected = "Return pseudo should have been lowered to RawRet")]
    fn return_pseudo_in_cleanup_input_panics() {
        let mut func = Func::<X64Inst>::new("t".to_string());
        let b = func.add_empty_block();
        let v0 = func.new_vreg();
        {
            let bd = func.get_block_data_mut(b);
            bd.push_pseudo_inst(PseudoInstruction::Return { src: v0 });
        }
        let ra = mk_ra(&[(v0, AllocatedSlot::Reg(0))], func.get_regs_count());
        run(&mut func, &ra);
    }
}
