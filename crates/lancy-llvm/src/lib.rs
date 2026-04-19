//! LLVM-IR frontend for lancy.
//!
//! Reads LLVM IR (textual or bitcode) from a memory buffer via inkwell,
//! picks a named function, and lowers its body into a `Func<X64Inst>` that
//! lancy's x64 pipeline can compile and JIT.
//!
//! The converter intentionally supports only the subset of LLVM IR that
//! lancy itself can currently encode — integer arithmetic on i64 (and
//! narrower ints widened to i64), signed compares, unconditional and
//! conditional branches terminating in per-block `ret`, and integer
//! constants. Programs that LLVM would lower to `phi`, `select`,
//! `alloca`, or calls are rejected with a `ConvertError`.

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::match_same_arms,
    clippy::must_use_candidate
)]

mod converter;
mod error;

pub use converter::{compile_ir_to_jit, convert_function};
pub use error::ConvertError;

// Re-export lancy's x64 condition codes so the converter can reference
// them via `crate::Cond` without threading through x64-specific paths.
pub(crate) use lancy::codegen::isa::x64::inst::Cond;
