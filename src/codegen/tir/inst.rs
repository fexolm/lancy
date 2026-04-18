use smallvec::{smallvec, SmallVec};
use std::fmt::{Debug, Display, Formatter};

use super::Reg;
use crate::codegen::tir::Block;
use crate::slotmap_key;

pub trait Inst: Sized + Copy + Display {
    fn is_branch(&self) -> bool;
    fn is_ret(&self) -> bool;

    fn is_term(&self) -> bool {
        self.is_branch() || self.is_ret()
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]>;
    fn get_defs(&self) -> SmallVec<[Reg; 1]>;

    fn get_branch_targets(&self) -> SmallVec<[Block; 2]>;
}

slotmap_key!(PhiId(u32));
slotmap_key!(CallId(u32));

impl Display for PhiId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "phi#{}", self.0)
    }
}

impl Debug for PhiId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for CallId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "call#{}", self.0)
    }
}

impl Debug for CallId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

/// Target-neutral pseudo instructions. Closed set.
///
/// Most pseudos are erased (`Kill`, `ImplicitDef`), lowered to targets
/// (`Arg`, `Return`, `CallPseudo`, `Phi`, `StackAlloc`, `FrameSetup`,
/// `FrameDestroy`), or honored as regalloc constraints (`RegDef`) by
/// earlier passes before machine-code emission. Two exceptions are
/// `Copy` (survives as a MOV candidate) and `Arg` (stays as a pinned
/// def shim after ABI lowering).
///
/// Variable-length operands — phi incoming edges and call arg/result
/// lists — live in side tables on `Func`, keyed by `PhiId` / `CallId`.
/// The enum itself stays `Copy` so instruction arrays can be moved and
/// pattern-matched cheaply.
#[derive(Copy, Clone, Debug)]
pub enum PseudoInstruction {
    /// Incoming argument `idx`. Lowered by the ABI pass.
    Arg { dst: Reg, idx: u32 },
    /// Typed value move. Primary coalescing candidate.
    Copy { dst: Reg, src: Reg },
    /// Abstract return. Lowered by the ABI pass into a `Copy` to the
    /// return register plus a target `Ret`-style instruction.
    Return { src: Reg },
    /// SSA merge. Variable-length `(Block, Reg)` list lives at
    /// `Func::phi_operands(id)`. Lowered by SSA destruction into
    /// parallel `Copy`s in predecessors.
    Phi { dst: Reg, id: PhiId },
    /// Reserve a stack slot. The ABI / prologue pass picks the slot; at
    /// that point `dst` holds the slot's address. Used for `alloca`.
    StackAlloc { dst: Reg, size: u32, align: u32 },
    /// Abstract call. Variable-length arg/result lists live at
    /// `Func::call_operands(id)`. Lowered by the ABI pass.
    CallPseudo { id: CallId },
    /// Marker for prologue insertion. Erased by prologue/epilogue pass.
    FrameSetup,
    /// Marker for epilogue insertion. Erased by prologue/epilogue pass.
    FrameDestroy,
    /// Defines `dst` as undef. Regalloc sees a def with no cost; pseudo
    /// cleanup erases it.
    ImplicitDef { dst: Reg },
    /// Explicit end-of-live-range marker. Regalloc consumes it; pseudo
    /// cleanup erases it.
    Kill { src: Reg },
    /// Pre-bind `vreg` to physical `preg`. Regalloc merges in-stream
    /// `RegDef` pins with `RegAllocConfig.reg_bind`; both mechanisms
    /// are equivalent and honored. A vreg pinned by both must agree —
    /// the allocator panics on disagreement. Erased after allocation.
    RegDef { vreg: Reg, preg: Reg },
}

impl Display for PseudoInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PseudoInstruction::Arg { dst, idx } => {
                write!(f, "{} = arg {idx}", reg_name(*dst))
            }
            PseudoInstruction::Copy { dst, src } => {
                write!(f, "{} = copy {}", reg_name(*dst), reg_name(*src))
            }
            PseudoInstruction::Return { src } => write!(f, "return {}", reg_name(*src)),
            PseudoInstruction::Phi { dst, id } => {
                write!(f, "{} = phi {id}", reg_name(*dst))
            }
            PseudoInstruction::StackAlloc { dst, size, align } => {
                write!(f, "{} = stackalloc size={size} align={align}", reg_name(*dst))
            }
            PseudoInstruction::CallPseudo { id } => write!(f, "call {id}"),
            PseudoInstruction::FrameSetup => f.write_str("frame_setup"),
            PseudoInstruction::FrameDestroy => f.write_str("frame_destroy"),
            PseudoInstruction::ImplicitDef { dst } => {
                write!(f, "{} = implicit_def", reg_name(*dst))
            }
            PseudoInstruction::Kill { src } => write!(f, "kill {}", reg_name(*src)),
            PseudoInstruction::RegDef { vreg, preg } => {
                write!(f, "regdef {} = p{preg}", reg_name(*vreg))
            }
        }
    }
}

impl Inst for PseudoInstruction {
    fn is_branch(&self) -> bool {
        false
    }

    fn is_ret(&self) -> bool {
        matches!(self, PseudoInstruction::Return { .. })
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]> {
        match self {
            PseudoInstruction::Copy { src, .. } | PseudoInstruction::Return { src } => {
                smallvec![*src]
            }
            PseudoInstruction::Kill { src } => smallvec![*src],
            // Phi and CallPseudo uses live in the side table on `Func`.
            // Callers that need those operands (SSA destruction, ABI
            // lowering) consult `Func::phi_operands` / `call_operands`
            // directly rather than going through `get_uses`.
            PseudoInstruction::Arg { .. }
            | PseudoInstruction::Phi { .. }
            | PseudoInstruction::StackAlloc { .. }
            | PseudoInstruction::CallPseudo { .. }
            | PseudoInstruction::FrameSetup
            | PseudoInstruction::FrameDestroy
            | PseudoInstruction::ImplicitDef { .. }
            | PseudoInstruction::RegDef { .. } => smallvec![],
        }
    }

    fn get_defs(&self) -> SmallVec<[Reg; 1]> {
        match self {
            PseudoInstruction::Arg { dst, .. }
            | PseudoInstruction::Copy { dst, .. }
            | PseudoInstruction::Phi { dst, .. }
            | PseudoInstruction::StackAlloc { dst, .. }
            | PseudoInstruction::ImplicitDef { dst } => smallvec![*dst],
            PseudoInstruction::RegDef { vreg, .. } => smallvec![*vreg],
            PseudoInstruction::Return { .. }
            | PseudoInstruction::CallPseudo { .. }
            | PseudoInstruction::FrameSetup
            | PseudoInstruction::FrameDestroy
            | PseudoInstruction::Kill { .. } => smallvec![],
        }
    }

    fn get_branch_targets(&self) -> SmallVec<[Block; 2]> {
        smallvec![]
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Instruction<I: Inst> {
    Target(I),
    Pseudo(PseudoInstruction),
}

impl<I: Inst> Display for Instruction<I> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Pseudo(inst) => write!(f, "{inst}"),
            Instruction::Target(inst) => write!(f, "{inst}"),
        }
    }
}

impl<I: Inst> Inst for Instruction<I> {
    fn is_branch(&self) -> bool {
        match self {
            Instruction::Target(inst) => inst.is_branch(),
            Instruction::Pseudo(inst) => inst.is_branch(),
        }
    }

    fn is_ret(&self) -> bool {
        match self {
            Instruction::Target(inst) => inst.is_ret(),
            Instruction::Pseudo(inst) => inst.is_ret(),
        }
    }

    fn get_uses(&self) -> SmallVec<[Reg; 2]> {
        match self {
            Instruction::Target(inst) => inst.get_uses(),
            Instruction::Pseudo(inst) => inst.get_uses(),
        }
    }

    fn get_defs(&self) -> SmallVec<[Reg; 1]> {
        match self {
            Instruction::Target(inst) => inst.get_defs(),
            Instruction::Pseudo(inst) => inst.get_defs(),
        }
    }

    fn get_branch_targets(&self) -> SmallVec<[Block; 2]> {
        match self {
            Instruction::Target(inst) => inst.get_branch_targets(),
            Instruction::Pseudo(inst) => inst.get_branch_targets(),
        }
    }
}

/// Side-table payload for `PseudoInstruction::Phi`. Owned by `Func`.
#[derive(Clone, Debug, Default)]
pub struct PhiData {
    /// `(predecessor_block, incoming_reg)` pairs — one per predecessor
    /// edge. Order matches predecessors in CFG iteration.
    pub incoming: Vec<(Block, Reg)>,
}

/// Side-table payload for `PseudoInstruction::CallPseudo`. Owned by
/// `Func`. `callee` is either a direct symbol name (`CallTarget::Symbol`,
/// resolved by the JIT at load time) or an indirect register holding a
/// function pointer (`CallTarget::Indirect`).
#[derive(Clone, Debug)]
pub struct CallData {
    pub callee: CallTarget,
    pub args: Vec<Reg>,
    pub rets: Vec<Reg>,
}

#[derive(Clone, Debug)]
pub enum CallTarget {
    /// Direct call resolved by symbol name at JIT load time.
    Symbol(String),
    /// Indirect call through a register holding a function pointer.
    Indirect(Reg),
}

#[must_use]
pub fn reg_name(reg: Reg) -> String {
    format!("v{reg}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::slotmap::Key;

    #[test]
    fn pseudo_arg_defs_dst_uses_nothing() {
        let p = PseudoInstruction::Arg { dst: 7, idx: 0 };
        assert_eq!(p.get_defs().as_slice(), &[7]);
        assert!(p.get_uses().is_empty());
        assert!(!p.is_term());
    }

    #[test]
    fn pseudo_copy_has_src_use_and_dst_def() {
        let p = PseudoInstruction::Copy { dst: 3, src: 2 };
        assert_eq!(p.get_defs().as_slice(), &[3]);
        assert_eq!(p.get_uses().as_slice(), &[2]);
    }

    #[test]
    fn pseudo_return_is_terminator_and_uses_src() {
        let p = PseudoInstruction::Return { src: 9 };
        assert!(p.is_term());
        assert!(p.is_ret());
        assert_eq!(p.get_uses().as_slice(), &[9]);
        assert!(p.get_defs().is_empty());
    }

    #[test]
    fn pseudo_phi_defs_dst_uses_empty_operand_list_lives_off_inst() {
        let p = PseudoInstruction::Phi {
            dst: 5,
            id: PhiId::new(0),
        };
        assert_eq!(p.get_defs().as_slice(), &[5]);
        assert!(p.get_uses().is_empty());
        assert!(!p.is_term());
    }

    #[test]
    fn pseudo_stackalloc_defs_dst() {
        let p = PseudoInstruction::StackAlloc {
            dst: 4,
            size: 16,
            align: 8,
        };
        assert_eq!(p.get_defs().as_slice(), &[4]);
        assert!(p.get_uses().is_empty());
    }

    #[test]
    fn pseudo_callpseudo_defs_and_uses_live_off_inst() {
        let p = PseudoInstruction::CallPseudo { id: CallId::new(0) };
        assert!(p.get_defs().is_empty());
        assert!(p.get_uses().is_empty());
    }

    #[test]
    fn pseudo_frame_markers_have_no_uses_or_defs() {
        let a = PseudoInstruction::FrameSetup;
        let b = PseudoInstruction::FrameDestroy;
        assert!(a.get_defs().is_empty() && a.get_uses().is_empty());
        assert!(b.get_defs().is_empty() && b.get_uses().is_empty());
    }

    #[test]
    fn pseudo_implicit_def_defs_dst_uses_nothing() {
        let p = PseudoInstruction::ImplicitDef { dst: 2 };
        assert_eq!(p.get_defs().as_slice(), &[2]);
        assert!(p.get_uses().is_empty());
    }

    #[test]
    fn pseudo_kill_uses_src_defs_nothing() {
        let p = PseudoInstruction::Kill { src: 8 };
        assert_eq!(p.get_uses().as_slice(), &[8]);
        assert!(p.get_defs().is_empty());
    }

    #[test]
    fn pseudo_regdef_defs_vreg_uses_nothing() {
        let p = PseudoInstruction::RegDef { vreg: 3, preg: 1 };
        assert_eq!(p.get_defs().as_slice(), &[3]);
        assert!(p.get_uses().is_empty());
    }

    #[test]
    fn display_format_for_each_new_pseudo() {
        let phi = PseudoInstruction::Phi {
            dst: 4,
            id: PhiId::new(2),
        };
        assert_eq!(format!("{phi}"), "v4 = phi phi#2");

        let sa = PseudoInstruction::StackAlloc {
            dst: 5,
            size: 16,
            align: 8,
        };
        assert_eq!(format!("{sa}"), "v5 = stackalloc size=16 align=8");

        let call = PseudoInstruction::CallPseudo { id: CallId::new(7) };
        assert_eq!(format!("{call}"), "call call#7");

        assert_eq!(format!("{}", PseudoInstruction::FrameSetup), "frame_setup");
        assert_eq!(format!("{}", PseudoInstruction::FrameDestroy), "frame_destroy");

        let idef = PseudoInstruction::ImplicitDef { dst: 8 };
        assert_eq!(format!("{idef}"), "v8 = implicit_def");

        let kill = PseudoInstruction::Kill { src: 9 };
        assert_eq!(format!("{kill}"), "kill v9");

        let rd = PseudoInstruction::RegDef { vreg: 1, preg: 3 };
        assert_eq!(format!("{rd}"), "regdef v1 = p3");
    }
}
