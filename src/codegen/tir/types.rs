//! Type system for virtual registers.
//!
//! Every `Reg` carries a `Type`. For scalar values the type names the machine
//! width and the integer/float flavor; for vectors the type names the lane
//! scalar plus the vector width in bits. `Agg` identifies an SSA aggregate
//! whose element decomposition lives in `Func::aggregate_operands(id)`.

use crate::slotmap_key;
use std::fmt::{Debug, Display, Formatter};

slotmap_key!(AggregateId(u32));

impl Display for AggregateId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "agg#{}", self.0)
    }
}

impl Debug for AggregateId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

/// Scalar type of a register or of a vector lane.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Ptr,
}

impl ScalarType {
    /// Lane width in bits.
    #[must_use]
    pub fn bits(self) -> u32 {
        match self {
            ScalarType::I8 => 8,
            ScalarType::I16 => 16,
            ScalarType::I32 => 32,
            ScalarType::I64 | ScalarType::F64 | ScalarType::Ptr => 64,
            ScalarType::F32 => 32,
        }
    }

    #[must_use]
    pub fn is_float(self) -> bool {
        matches!(self, ScalarType::F32 | ScalarType::F64)
    }
}

impl Display for ScalarType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ScalarType::I8 => "i8",
            ScalarType::I16 => "i16",
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::Ptr => "ptr",
        };
        f.write_str(s)
    }
}

/// Full register type: scalar, vector, or aggregate handle.
///
/// Aggregate values have no single machine slot — the `AggregateId` points
/// into a side table of element vregs, and the `lower_aggregates` pass
/// rewrites `ExtractValue`/`InsertValue` pseudos into scalar `Copy`s before
/// regalloc ever sees the aggregate vreg.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Ptr,
    /// 128-bit vector of lanes.
    V128(ScalarType),
    /// 256-bit vector of lanes.
    V256(ScalarType),
    /// 512-bit vector of lanes.
    V512(ScalarType),
    /// SSA aggregate. Its composition lives in `Func`'s aggregate side
    /// table; the vreg itself never reaches codegen.
    Agg(AggregateId),
}

impl Type {
    /// Promote a scalar type to the `Type` enum.
    #[must_use]
    pub fn scalar(s: ScalarType) -> Self {
        match s {
            ScalarType::I8 => Type::I8,
            ScalarType::I16 => Type::I16,
            ScalarType::I32 => Type::I32,
            ScalarType::I64 => Type::I64,
            ScalarType::F32 => Type::F32,
            ScalarType::F64 => Type::F64,
            ScalarType::Ptr => Type::Ptr,
        }
    }

    /// Whether a value of this type lives in a floating-point / vector
    /// register (XMM/YMM/ZMM on x86-64). Used by the allocator to pick
    /// the right physical-register class.
    #[must_use]
    pub fn is_fp_or_vector(self) -> bool {
        matches!(
            self,
            Type::F32
                | Type::F64
                | Type::V128(_)
                | Type::V256(_)
                | Type::V512(_)
        )
    }

    #[must_use]
    pub fn is_aggregate(self) -> bool {
        matches!(self, Type::Agg(_))
    }

    /// Machine-word width in bytes for scalar types. Aggregates and
    /// vectors return `None` — they do not occupy a single GPR slot.
    #[must_use]
    pub fn scalar_bytes(self) -> Option<u32> {
        match self {
            Type::I8 => Some(1),
            Type::I16 => Some(2),
            Type::I32 | Type::F32 => Some(4),
            Type::I64 | Type::F64 | Type::Ptr => Some(8),
            Type::V128(_) | Type::V256(_) | Type::V512(_) | Type::Agg(_) => None,
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::I8 => f.write_str("i8"),
            Type::I16 => f.write_str("i16"),
            Type::I32 => f.write_str("i32"),
            Type::I64 => f.write_str("i64"),
            Type::F32 => f.write_str("f32"),
            Type::F64 => f.write_str("f64"),
            Type::Ptr => f.write_str("ptr"),
            Type::V128(s) => write!(f, "v128<{s}>"),
            Type::V256(s) => write!(f, "v256<{s}>"),
            Type::V512(s) => write!(f, "v512<{s}>"),
            Type::Agg(id) => write!(f, "{id}"),
        }
    }
}

/// Side-table payload for aggregates. Owned by `Func`.
///
/// `elems[i]` is the vreg carrying aggregate element `i`. All element
/// vregs must be scalar (non-aggregate) — nested aggregates are modeled
/// by listing their leaves directly.
#[derive(Clone, Debug, Default)]
pub struct AggregateData {
    pub elems: Vec<crate::codegen::tir::Reg>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_promotion_round_trips() {
        for s in [
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::F32,
            ScalarType::F64,
            ScalarType::Ptr,
        ] {
            let t = Type::scalar(s);
            assert!(t.scalar_bytes().is_some());
        }
    }

    #[test]
    fn fp_and_vector_types_report_fp_class() {
        assert!(Type::F32.is_fp_or_vector());
        assert!(Type::F64.is_fp_or_vector());
        assert!(Type::V128(ScalarType::I32).is_fp_or_vector());
        assert!(!Type::I64.is_fp_or_vector());
        assert!(!Type::Ptr.is_fp_or_vector());
    }

    #[test]
    fn scalar_bytes_match_bit_widths() {
        assert_eq!(Type::I8.scalar_bytes(), Some(1));
        assert_eq!(Type::I16.scalar_bytes(), Some(2));
        assert_eq!(Type::I32.scalar_bytes(), Some(4));
        assert_eq!(Type::I64.scalar_bytes(), Some(8));
        assert_eq!(Type::F32.scalar_bytes(), Some(4));
        assert_eq!(Type::F64.scalar_bytes(), Some(8));
        assert_eq!(Type::Ptr.scalar_bytes(), Some(8));
        assert!(Type::V128(ScalarType::I32).scalar_bytes().is_none());
    }

    #[test]
    fn display_forms_are_stable() {
        assert_eq!(format!("{}", Type::I64), "i64");
        assert_eq!(format!("{}", Type::F32), "f32");
        assert_eq!(
            format!("{}", Type::V256(ScalarType::F64)),
            "v256<f64>"
        );
    }
}
