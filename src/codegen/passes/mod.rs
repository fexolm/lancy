//! Trait definitions for swappable compilation passes.
//!
//! Each pass has a trait whose concrete implementations can be plugged into
//! the pipeline. Today there's one implementation per slot (SysV ABI for
//! x64, one regalloc); the trait infrastructure exists so multiple can
//! coexist and be compared.

pub mod ssa_destruction;

pub use ssa_destruction::destroy_ssa;

use std::collections::HashMap;

use crate::codegen::tir::{Func, Inst, Reg};

/// Output of an ABI-lowering pass.
///
/// `reg_bind` maps the pinned shim / return vregs introduced during lowering
/// to their ABI-fixed physical registers. The regalloc consumes this as its
/// pre-bind constraint set. `call_sites` records each CallPseudo's
/// synthesized call-target `Mov64ri` so later stages can patch its
/// 64-bit immediate with a real symbol address.
pub struct AbiLowerResult {
    pub reg_bind: HashMap<Reg, Reg>,
    pub call_sites: Vec<CallSite>,
}

/// One expanded call site. `addr_vreg` is the vreg that holds the
/// callee's address (set via `Mov64ri` with a placeholder). `symbol`
/// is the target symbol name; an empty string means indirect call.
#[derive(Clone, Debug)]
pub struct CallSite {
    pub addr_vreg: Reg,
    pub symbol: String,
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
