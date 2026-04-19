//! Register allocation.
//!
//! Shared types (`AllocatedSlot`, `Assignment`, `RegAllocConfig`,
//! `RegAllocResult`, `StackSlot`, `SplitMove`) and the `RegAllocator` trait.
//! Concrete allocators live in submodules and plug in by implementing the
//! trait; the pipeline can swap algorithms for comparison or benchmarking
//! without rewiring emission.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::codegen::analysis::cfg::CFG;
use crate::codegen::analysis::layout::ProgramPoint;
use crate::codegen::analysis::liveness::Segment;
use crate::codegen::tir::{Func, Inst, Reg};
use crate::support::slotmap::SecondaryMap;

pub type StackSlot = u32;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum AllocatedSlot {
    Reg(Reg),
    Stack(StackSlot),
}

/// Per-vreg assignment. Most vregs have a single piece spanning their entire
/// live range. Vregs that were evicted mid-life carry multiple pieces — a
/// `Reg(p)` piece up to the split point followed by a `Stack(s)` piece after,
/// with a corresponding `SplitMove` in the result to save `p` into `s`.
///
/// Pieces are sorted by segment start and are non-overlapping. The emitter
/// queries `at(program_point)` for each use / def it emits to figure out
/// where the value is *at that specific point*.
#[derive(Default, Debug, Clone)]
pub struct Assignment {
    pub pieces: SmallVec<[(Segment, AllocatedSlot); 1]>,
}

impl Assignment {
    /// Single-piece assignment across `[start, end)`. The common case: a
    /// vreg never evicted.
    #[must_use]
    pub fn uniform(slot: AllocatedSlot, start: ProgramPoint, end: ProgramPoint) -> Self {
        let mut pieces = SmallVec::new();
        if start < end {
            pieces.push((Segment { start, end }, slot));
        }
        Self { pieces }
    }

    /// Slot that contains the vreg at `pt`, or `None` if `pt` is outside
    /// every piece (i.e. the vreg isn't live there).
    #[must_use]
    pub fn at(&self, pt: ProgramPoint) -> Option<AllocatedSlot> {
        // Linear scan: most vregs have 1-2 pieces. Not worth a binary
        // search until proven bottlenecked.
        for (seg, slot) in &self.pieces {
            if seg.contains(pt) {
                return Some(*slot);
            }
        }
        None
    }

    /// The single slot if the assignment has exactly one piece and that
    /// piece's slot is it. Useful for callers that don't track program
    /// points and just want "what slot is this vreg in" when it's uniform.
    #[must_use]
    pub fn uniform_slot(&self) -> Option<AllocatedSlot> {
        if self.pieces.len() == 1 {
            Some(self.pieces[0].1)
        } else {
            None
        }
    }

    /// Iterator over every *distinct* slot the vreg visits. Used by the MC
    /// emitter to compute the callee-saved-regs set without caring about
    /// when the slot is active.
    pub fn slots(&self) -> impl Iterator<Item = AllocatedSlot> + '_ {
        self.pieces.iter().map(|(_, s)| *s)
    }
}

/// A store of a live value from a preg into a stack slot, to be inserted by
/// the emitter immediately before the instruction whose early point equals
/// `at_point`. Generated whenever the allocator splits a live range: the
/// vreg held `from_preg` up to `at_point`, after which it lives in
/// `to_slot` — so the preg's value must be preserved before the reuse.
#[derive(Clone, Copy, Debug)]
pub struct SplitMove {
    pub at_point: ProgramPoint,
    pub from_preg: Reg,
    pub to_slot: StackSlot,
}

/// Per-function output of a `RegAllocator`. Consumed by `pseudo_cleanup` and
/// the MC emitter. `frame_layout[s]` is the byte offset of slot `s` from the
/// frame pointer (see the MC emitter); slots are dense `0..frame_size/8`.
pub struct RegAllocResult {
    pub assignments: SecondaryMap<Reg, Assignment>,
    pub frame_layout: Vec<usize>,
    pub frame_size: u32,
    pub split_moves: Vec<SplitMove>,
}

impl RegAllocResult {
    /// `at(vreg, pt)` is the slot holding `vreg` at program point `pt`, or
    /// `None` if the vreg isn't live there (or has no assignment at all).
    #[must_use]
    pub fn at(&self, vreg: Reg, pt: ProgramPoint) -> Option<AllocatedSlot> {
        self.assignments.get(vreg).and_then(|a| a.at(pt))
    }
}

/// Target-neutral inputs to allocation.
///
/// * `preg_count` — size of the physical register space; `Reg` values in the
///   allocation result are < this.
/// * `allocatable_regs` — GPR-class pool (integer / pointer vregs). Ordered
///   roughly by preference (callee-saved last so caller-saved wins on ties).
/// * `scratch_regs` — GPR scratches reserved for the MC emitter's spill
///   reload/spill. Must not overlap `allocatable_regs`.
/// * `allocatable_fp_regs` — XMM-class pool (float / vector vregs). Routed by
///   `Func::vreg_type(v).is_fp_or_vector()`. May be empty if the frontend
///   doesn't emit any FP vregs.
/// * `scratch_fp_regs` — XMM scratches for FP spill reload/spill. Similar
///   disjointness rule.
/// * `reg_bind` — pre-binds: `vreg -> preg` constraints. The allocator must
///   honor these even if it means evicting.
pub struct RegAllocConfig {
    pub preg_count: usize,
    pub allocatable_regs: Vec<Reg>,
    pub scratch_regs: Vec<Reg>,
    pub allocatable_fp_regs: Vec<Reg>,
    pub scratch_fp_regs: Vec<Reg>,
    pub reg_bind: HashMap<Reg, Reg>,
}

/// A register-allocation algorithm. Static-dispatch trait — callers pick the
/// implementation by type (e.g. `LinearScan::allocate(&f, &cfg, &c)`).
pub trait RegAllocator<I: Inst> {
    fn allocate(func: &Func<I>, cfg: &CFG, config: &RegAllocConfig) -> RegAllocResult;
}

pub mod linear_scan;
pub use linear_scan::LinearScan;
