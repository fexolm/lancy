//! Trait definitions for swappable compilation passes.
//!
//! Each pass has a trait whose concrete implementations can be plugged into
//! the pipeline. Today there's one implementation per slot (SysV ABI for
//! x64, one regalloc); the trait infrastructure exists so multiple can
//! coexist and be compared.

use std::collections::HashMap;

use crate::codegen::tir::{Func, Inst, Reg};

/// Output of an ABI-lowering pass.
///
/// `reg_bind` maps the pinned shim / return vregs introduced during lowering
/// to their ABI-fixed physical registers. The regalloc consumes this as its
/// pre-bind constraint set.
pub struct AbiLowerResult {
    pub reg_bind: HashMap<Reg, Reg>,
}

/// Lowers target-neutral ABI pseudos (`Arg`, `Return`, eventually
/// `CallPseudo`) into a calling-convention-specific concrete sequence.
///
/// Implementations are ISA-specific (they emit target `I` instructions for
/// return / call terminators) and CC-specific (they pick the right argument
/// / return registers). A given CC on a given ISA is one implementor, e.g.
/// `SysVAmd64Lowering: AbiLowering<X64Inst>`.
pub trait AbiLowering<I: Inst> {
    fn lower(&self, func: &mut Func<I>) -> AbiLowerResult;
}
