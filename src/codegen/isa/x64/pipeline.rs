//! End-to-end x64 compilation pipeline.
//!
//! `compile(func)` runs every required pass in order and returns the emitted
//! machine-code bytes. `jit(func)` additionally loads the bytes into an
//! executable mapping.

use crate::codegen::analysis::cfg::CFG;
use crate::codegen::isa::x64::inst::X64Inst;
use crate::codegen::isa::x64::mc::emit_mc::FnMCWriter;
use crate::codegen::isa::x64::passes::abi_lower::SysVAmd64Lowering;
use crate::codegen::isa::x64::regs::{
    R8, R9, R10, R11, R12, R13, R14, R15, RAX, RBX, RCX, RDI, RDX, RSI,
};
use crate::codegen::jit::{Module, Relocation};
use crate::codegen::passes::{AbiLowering, destroy_ssa};
use crate::codegen::regalloc::{LinearScan, RegAllocConfig, RegAllocator};
use crate::codegen::tir::{Func, Reg};
use std::collections::HashMap;

/// Build the default `SysV`-flavored `RegAllocConfig`. The allocatable pool is
/// the nine caller-saved integer registers (`RAX/RCX/RDX/RSI/RDI/R8..R11`) plus
/// two callee-saved (`R14, R15`). The MC emitter's prologue saves/restores any
/// callee-saved regs the allocator actually hands out. `RBX`, `R12`, `R13` are
/// reserved as spill scratches (callee-saved, also saved by the emitter when
/// touched). Three scratches is the worst case needed by
/// `Mov64rm { src: Mem { base, index: Some, .. }, dst }` when `base`,
/// `index`, and `dst` are all spilled simultaneously.
///
/// Ordering matters: allocation uses `max_by_key` over a per-preg free-until
/// point, and `max_by_key` returns the **last** element on ties. Callee-saved
/// are placed *first* so that when every preg is equally free, a caller-saved
/// register wins — keeping prologue/epilogue push/pop traffic minimal.
#[must_use]
pub fn default_ra_config(reg_bind: HashMap<Reg, Reg>) -> RegAllocConfig {
    RegAllocConfig {
        preg_count: 32,
        allocatable_regs: vec![R14, R15, RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11],
        scratch_regs: vec![RBX, R12, R13],
        reg_bind,
    }
}

/// Output of the full compile pipeline: executable bytes plus any
/// call-site relocations requiring symbol resolution at load time.
pub struct Compiled {
    pub name: String,
    pub bytes: Vec<u8>,
    pub relocations: Vec<Relocation>,
}

/// Compile a function end-to-end. Returns the emitted bytes.
#[must_use]
pub fn compile(func: Func<X64Inst>) -> Vec<u8> {
    compile_full(func).bytes
}

/// Full compile pipeline including call-site relocation capture.
#[must_use]
pub fn compile_full(mut func: Func<X64Inst>) -> Compiled {
    let name = func.name().to_string();
    // Phi → parallel Copies before anything else. Subsequent passes
    // assume the IR is phi-free.
    destroy_ssa(&mut func);
    let abi = SysVAmd64Lowering.lower(&mut func);
    let cfg = CFG::compute(&func).expect("CFG compute on valid function");
    let mut reg_bind = abi.reg_bind;
    for (&v, &p) in func.pre_binds() {
        match reg_bind.insert(v, p) {
            Some(prev) if prev != p => panic!(
                "vreg {v} pre-bound to two different pregs: {prev} (from ABI) vs {p} (from frontend)"
            ),
            _ => {}
        }
    }
    let ra_cfg = default_ra_config(reg_bind);
    let ra_res = LinearScan::allocate(&func, &cfg, &ra_cfg);
    let mut w = FnMCWriter::new(&func, &ra_cfg, &ra_res);
    let emitted = w.emit_fn_with_relocs(&abi.call_sites);
    let relocations = emitted
        .relocations
        .into_iter()
        .map(|r| Relocation {
            offset: r.imm_offset,
            symbol: r.symbol,
        })
        .collect();
    Compiled {
        name,
        bytes: emitted.bytes,
        relocations,
    }
}

/// Compile a function and load the resulting bytes into an executable mapping.
/// Returns the `Module` (which must outlive any derived function pointers).
///
/// # Errors
/// Propagates `io::Error` from `mmap` / `mprotect` / `dlsym` in the JIT runtime.
pub fn jit(func: Func<X64Inst>) -> std::io::Result<Module> {
    let compiled = compile_full(func);
    Module::load_with_relocs(&compiled.bytes, &compiled.relocations, &compiled.name)
}

#[cfg(test)]
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
mod tests {
    use super::*;
    use crate::codegen::isa::x64::builder::FuncBuilder;

    #[allow(non_camel_case_types)]
    type FnI64_I64 = unsafe extern "sysv64" fn(i64) -> i64;
    #[allow(non_camel_case_types)]
    type FnI64I64_I64 = unsafe extern "sysv64" fn(i64, i64) -> i64;

    #[test]
    fn jit_identity_returns_argument() {
        let mut b = FuncBuilder::new("id");
        let a = b.arg();
        b.ret(a);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [-3, 0, 1, 42, 1_000_000_007] {
            assert_eq!(unsafe { f(x) }, x);
        }
    }

    #[test]
    fn jit_constant_function_returns_constant() {
        let mut b = FuncBuilder::new("k");
        let _a = b.arg();
        let k = b.iconst64(12345);
        b.ret(k);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(0) }, 12345);
        assert_eq!(unsafe { f(-99) }, 12345);
    }

    #[test]
    fn jit_add_two_args() {
        let mut b = FuncBuilder::new("add");
        let x = b.arg();
        let y = b.arg();
        let s = b.add(x, y);
        b.ret(s);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(1, 2) }, 3);
        assert_eq!(unsafe { f(-5, 5) }, 0);
        assert_eq!(unsafe { f(i64::MAX, 0) }, i64::MAX);
        assert_eq!(unsafe { f(100, -50) }, 50);
    }

    #[test]
    fn jit_sub_two_args() {
        let mut b = FuncBuilder::new("sub");
        let x = b.arg();
        let y = b.arg();
        let s = b.sub(x, y);
        b.ret(s);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(10, 3) }, 7);
        assert_eq!(unsafe { f(0, 1) }, -1);
        assert_eq!(unsafe { f(i64::MIN, 0) }, i64::MIN);
    }

    #[test]
    fn jit_imul_two_args() {
        let mut b = FuncBuilder::new("mul");
        let x = b.arg();
        let y = b.arg();
        let s = b.imul(x, y);
        b.ret(s);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(6, 7) }, 42);
        assert_eq!(unsafe { f(-3, 4) }, -12);
        assert_eq!(unsafe { f(0, 100) }, 0);
    }

    #[test]
    fn jit_branch_max_of_two() {
        use crate::codegen::isa::x64::inst::Cond;
        let mut b = FuncBuilder::new("max");
        let a = b.arg();
        let c = b.arg();
        let then_blk = b.new_block();
        let else_blk = b.new_block();
        b.branch_icmp(Cond::GE, a, c, then_blk, else_blk);
        b.switch_to_block(then_blk);
        b.ret(a);
        b.switch_to_block(else_blk);
        b.ret(c);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(5, 3) }, 5);
        assert_eq!(unsafe { f(3, 5) }, 5);
        assert_eq!(unsafe { f(-10, -20) }, -10);
        assert_eq!(unsafe { f(7, 7) }, 7);
    }

    #[test]
    fn jit_chain_of_adds_forces_regalloc_to_hold_many_live_values() {
        let mut b = FuncBuilder::new("chain");
        let a = b.arg();
        let c = b.arg();
        let mut acc = b.add(a, c);
        for _ in 0..10 {
            acc = b.add(acc, c);
        }
        b.ret(acc);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (x, y) in [(0, 1), (3, 4), (-10, 2), (100, -1)] {
            assert_eq!(unsafe { f(x, y) }, x + 11 * y);
        }
    }

    #[test]
    fn jit_under_register_pressure_producing_correct_result() {
        let mut b = FuncBuilder::new("pressure");
        let a = b.arg();
        let c = b.arg();
        let step = b.add(a, c);
        let mut acc = step;
        for _ in 0..19 {
            acc = b.add(acc, step);
        }
        b.ret(acc);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (x, y) in [(1_i64, 1_i64), (-5, 7), (100, -101)] {
            assert_eq!(unsafe { f(x, y) }, 20 * (x + y));
        }
    }

    #[test]
    fn jit_deep_chain_forces_spills_to_stack() {
        let mut b = FuncBuilder::new("many_sums");
        let a = b.arg();
        let c = b.arg();
        let mut vals = vec![a, c];
        for _ in 0..15 {
            let s = b.add(vals[vals.len() - 1], vals[vals.len() - 2]);
            vals.push(s);
        }
        let mut acc = vals[0];
        for v in &vals[1..] {
            acc = b.add(acc, *v);
        }
        b.ret(acc);

        fn reference(a: i64, c: i64) -> i64 {
            let mut v = vec![a, c];
            for _ in 0..15 {
                let s = v[v.len() - 1].wrapping_add(v[v.len() - 2]);
                v.push(s);
            }
            v.iter().copied().fold(0_i64, i64::wrapping_add)
        }

        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (x, y) in [(1_i64, 1_i64), (2, 3), (-1, -2), (7, -11), (0, 5)] {
            let got = unsafe { f(x, y) };
            let want = reference(x, y);
            assert_eq!(got, want, "f({x},{y}) disagrees with reference");
        }
    }

    #[test]
    fn jit_uses_all_six_sysv_arg_registers() {
        #[allow(non_camel_case_types)]
        type Fn6 = unsafe extern "sysv64" fn(i64, i64, i64, i64, i64, i64) -> i64;
        let mut b = FuncBuilder::new("six");
        let a = b.arg();
        let bb = b.arg();
        let c = b.arg();
        let d = b.arg();
        let e = b.arg();
        let g = b.arg();
        let t1 = b.add(a, bb);
        let t2 = b.sub(t1, c);
        let t3 = b.imul(t2, d);
        let t4 = b.add(t3, e);
        let t5 = b.sub(t4, g);
        b.ret(t5);
        let m = jit(b.build()).unwrap();
        let f: Fn6 = unsafe { m.entry() };
        let reference = |a: i64, bb: i64, c: i64, d: i64, e: i64, g: i64| -> i64 {
            (((a.wrapping_add(bb)).wrapping_sub(c)).wrapping_mul(d))
                .wrapping_add(e)
                .wrapping_sub(g)
        };
        for inputs in [
            (1, 2, 3, 4, 5, 6),
            (-1, -2, -3, -4, -5, -6),
            (10, 20, 5, 3, 100, -50),
            (0, 0, 0, 1, 0, 0),
        ] {
            let (a, bb, c, d, e, g) = inputs;
            assert_eq!(
                unsafe { f(a, bb, c, d, e, g) },
                reference(a, bb, c, d, e, g),
                "inputs = {inputs:?}"
            );
        }
    }

    #[test]
    fn jit_branch_sign_reports_minus_one_zero_or_one() {
        use crate::codegen::isa::x64::inst::Cond;
        let mut b = FuncBuilder::new("sign");
        let x = b.arg();
        let zero = b.iconst64(0);
        let pos_blk = b.new_block();
        let neg_or_zero_blk = b.new_block();
        let neg_blk = b.new_block();
        let zero_blk = b.new_block();
        b.branch_icmp(Cond::G, x, zero, pos_blk, neg_or_zero_blk);
        b.switch_to_block(neg_or_zero_blk);
        let zero2 = b.iconst64(0);
        b.branch_icmp(Cond::L, x, zero2, neg_blk, zero_blk);
        b.switch_to_block(pos_blk);
        let one = b.iconst64(1);
        b.ret(one);
        b.switch_to_block(neg_blk);
        let neg_one = b.iconst64(-1);
        b.ret(neg_one);
        b.switch_to_block(zero_blk);
        let zero3 = b.iconst64(0);
        b.ret(zero3);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(0) }, 0);
        assert_eq!(unsafe { f(1) }, 1);
        assert_eq!(unsafe { f(-1) }, -1);
        assert_eq!(unsafe { f(i64::MAX) }, 1);
        assert_eq!(unsafe { f(i64::MIN) }, -1);
        assert_eq!(unsafe { f(42) }, 1);
        assert_eq!(unsafe { f(-42) }, -1);
    }

    #[test]
    fn jit_randomized_add_mul_matches_rust_reference() {
        let mut b = FuncBuilder::new("diff_squares");
        let a = b.arg();
        let c = b.arg();
        let sum = b.add(a, c);
        let dif = b.sub(a, c);
        let prod = b.imul(sum, dif);
        b.ret(prod);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };

        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state as i64
        };
        for _ in 0..200 {
            let (x, y) = (next() >> 32, next() >> 32);
            let got = unsafe { f(x, y) };
            let want = x.wrapping_mul(x).wrapping_sub(y.wrapping_mul(y));
            assert_eq!(got, want, "x={x}, y={y}");
        }
    }

    #[test]
    fn compile_produces_deterministic_bytes() {
        let build = || {
            let mut b = FuncBuilder::new("det");
            let x = b.arg();
            let y = b.arg();
            let s = b.add(x, y);
            b.ret(s);
            b.build()
        };
        let a = compile(build());
        let bb = compile(build());
        assert_eq!(a, bb);
        assert!(!a.is_empty());
    }

    #[test]
    fn compile_emits_prologue_and_epilogue_markers() {
        let mut b = FuncBuilder::new("t");
        let x = b.arg();
        b.ret(x);
        let bytes = compile(b.build());
        assert_eq!(bytes[0], 0x55, "prologue must open with push rbp");
        let n = bytes.len();
        assert_eq!(&bytes[n - 2..], &[0x5D, 0xC3]);
    }

    // -------------- Integer op coverage --------------

    #[test]
    fn jit_and_computes_bitwise_and() {
        let mut b = FuncBuilder::new("and");
        let x = b.arg();
        let y = b.arg();
        let r = b.and(x, y);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, b) in [(0_i64, 0), (0xFF, 0x0F), (-1, 0x12345678), (i64::MAX, 42)] {
            assert_eq!(unsafe { f(a, b) }, a & b, "({a},{b})");
        }
    }

    #[test]
    fn jit_or_computes_bitwise_or() {
        let mut b = FuncBuilder::new("or");
        let x = b.arg();
        let y = b.arg();
        let r = b.or(x, y);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, b) in [(0_i64, 0), (0x0F, 0xF0), (-1, 0), (0x8000_0000_0000_0000_u64 as i64, 1)] {
            assert_eq!(unsafe { f(a, b) }, a | b, "({a},{b})");
        }
    }

    #[test]
    fn jit_xor_computes_bitwise_xor() {
        let mut b = FuncBuilder::new("xor");
        let x = b.arg();
        let y = b.arg();
        let r = b.xor(x, y);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, b) in [(0_i64, 0), (-1, -1), (0xDEAD_BEEF, 0xCAFE), (i64::MAX, i64::MIN)] {
            assert_eq!(unsafe { f(a, b) }, a ^ b, "({a},{b})");
        }
    }

    #[test]
    fn jit_not_flips_all_bits() {
        let mut b = FuncBuilder::new("not");
        let x = b.arg();
        let r = b.not(x);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [0_i64, -1, 42, i64::MAX, i64::MIN] {
            assert_eq!(unsafe { f(x) }, !x, "x={x}");
        }
    }

    #[test]
    fn jit_neg_negates() {
        let mut b = FuncBuilder::new("neg");
        let x = b.arg();
        let r = b.neg(x);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [0_i64, 1, -1, i64::MAX] {
            assert_eq!(unsafe { f(x) }, x.wrapping_neg(), "x={x}");
        }
    }

    #[test]
    fn jit_shl_imm_shifts_left_by_constant() {
        let mut b = FuncBuilder::new("shl_imm");
        let x = b.arg();
        let r = b.shl_imm(x, 3);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [0_i64, 1, 7, -1, i64::MAX / 8] {
            assert_eq!(unsafe { f(x) }, x.wrapping_shl(3), "x={x}");
        }
    }

    #[test]
    fn jit_shr_imm_logical_shifts_right_by_constant() {
        let mut b = FuncBuilder::new("shr_imm");
        let x = b.arg();
        let r = b.shr_imm(x, 4);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [0_i64, 1, 0xFF, -1] {
            let want = ((x as u64) >> 4) as i64;
            assert_eq!(unsafe { f(x) }, want, "x={x}");
        }
    }

    #[test]
    fn jit_sar_imm_arithmetic_shifts_right_by_constant() {
        let mut b = FuncBuilder::new("sar_imm");
        let x = b.arg();
        let r = b.sar_imm(x, 5);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [0_i64, 1, -1, i64::MIN, i64::MAX] {
            assert_eq!(unsafe { f(x) }, x >> 5, "x={x}");
        }
    }

    #[test]
    fn jit_shl_rr_shifts_left_by_register() {
        let mut b = FuncBuilder::new("shl_rr");
        let x = b.arg();
        let cnt = b.arg();
        let r = b.shl(x, cnt);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, c) in [(1_i64, 0), (1, 3), (-1, 1), (0xFF, 16), (0xABCD, 32)] {
            let want = (a as u64).wrapping_shl(c as u32) as i64;
            assert_eq!(unsafe { f(a, c) }, want, "({a},{c})");
        }
    }

    #[test]
    fn jit_shr_rr_logical_shifts_right_by_register() {
        let mut b = FuncBuilder::new("shr_rr");
        let x = b.arg();
        let cnt = b.arg();
        let r = b.shr(x, cnt);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, c) in [(0x1000_i64, 0), (-1, 1), (-1, 63), (0xFF00, 4)] {
            let want = ((a as u64) >> (c as u32 & 63)) as i64;
            assert_eq!(unsafe { f(a, c) }, want, "({a},{c})");
        }
    }

    #[test]
    fn jit_sar_rr_arithmetic_shifts_right_by_register() {
        let mut b = FuncBuilder::new("sar_rr");
        let x = b.arg();
        let cnt = b.arg();
        let r = b.sar(x, cnt);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, c) in [(-8_i64, 2), (8, 1), (-1, 10), (i64::MIN, 1)] {
            assert_eq!(unsafe { f(a, c) }, a >> (c & 63), "({a},{c})");
        }
    }

    #[test]
    fn jit_sdiv_signed_division() {
        let mut b = FuncBuilder::new("sdiv");
        let x = b.arg();
        let y = b.arg();
        let r = b.sdiv(x, y);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, b) in [(10_i64, 3), (-10, 3), (10, -3), (-10, -3), (7, 2), (0, 5)] {
            assert_eq!(unsafe { f(a, b) }, a / b, "({a},{b})");
        }
    }

    #[test]
    fn jit_srem_signed_remainder() {
        let mut b = FuncBuilder::new("srem");
        let x = b.arg();
        let y = b.arg();
        let r = b.srem(x, y);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, b) in [(10_i64, 3), (-10, 3), (10, -3), (-10, -3), (7, 2)] {
            assert_eq!(unsafe { f(a, b) }, a % b, "({a},{b})");
        }
    }

    #[test]
    fn jit_udiv_unsigned_division() {
        let mut b = FuncBuilder::new("udiv");
        let x = b.arg();
        let y = b.arg();
        let r = b.udiv(x, y);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, b) in [(10_u64, 3), (1_000_000, 7), (u64::MAX, 2)] {
            let want = (a / b) as i64;
            assert_eq!(unsafe { f(a as i64, b as i64) }, want, "({a},{b})");
        }
    }

    #[test]
    fn jit_urem_unsigned_remainder() {
        let mut b = FuncBuilder::new("urem");
        let x = b.arg();
        let y = b.arg();
        let r = b.urem(x, y);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, b) in [(10_u64, 3), (1_000_000, 7), (u64::MAX, 2)] {
            let want = (a % b) as i64;
            assert_eq!(unsafe { f(a as i64, b as i64) }, want, "({a},{b})");
        }
    }

    #[test]
    fn jit_select_picks_true_or_false_value() {
        use crate::codegen::isa::x64::inst::Cond;
        let mut b = FuncBuilder::new("select_lt");
        let x = b.arg();
        let y = b.arg();
        // result = if x < y { 100 } else { 200 }
        let t = b.iconst64(100);
        let f = b.iconst64(200);
        let r = b.select(Cond::L, x, y, t, f);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(1, 2) }, 100);
        assert_eq!(unsafe { f(2, 1) }, 200);
        assert_eq!(unsafe { f(5, 5) }, 200);
        assert_eq!(unsafe { f(-5, -3) }, 100);
    }

    // -------------- Phi / SSA destruction coverage --------------

    #[test]
    fn jit_phi_merges_two_paths_and_returns_correct_value() {
        use crate::codegen::isa::x64::inst::Cond;
        // fn max(a, b) via phi:
        //   entry: if a >= b { goto then } else { goto else }
        //   then: let r_a = a; goto merge
        //   else: let r_b = b; goto merge
        //   merge: r = phi(r_a from then, r_b from else); ret r
        let mut b = FuncBuilder::new("max_phi");
        let a = b.arg();
        let c = b.arg();
        let then_blk = b.new_block();
        let else_blk = b.new_block();
        let merge_blk = b.new_block();
        b.branch_icmp(Cond::GE, a, c, then_blk, else_blk);
        b.switch_to_block(then_blk);
        b.jmp(merge_blk);
        b.switch_to_block(else_blk);
        b.jmp(merge_blk);
        b.switch_to_block(merge_blk);
        let r = b.phi(vec![(then_blk, a), (else_blk, c)]);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(5, 3) }, 5);
        assert_eq!(unsafe { f(3, 5) }, 5);
        assert_eq!(unsafe { f(-2, -5) }, -2);
    }

    #[test]
    fn jit_phi_counts_down_loop_to_zero() {
        use crate::codegen::isa::x64::inst::Cond;
        // fn sum_to_n(n) via loop with phi:
        //   entry: jmp header
        //   header: i = phi(n from entry, i_next from body)
        //           acc = phi(0 from entry, acc_next from body)
        //           if i <= 0 goto exit else goto body
        //   body: acc_next = acc + i; i_next = i - 1; jmp header
        //   exit: ret acc
        let mut b = FuncBuilder::new("sum_loop");
        let n = b.arg();
        let zero = b.iconst64(0);
        let one = b.iconst64(1);
        let entry_blk = b.entry_block();
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        b.switch_to_block(entry_blk);
        b.jmp(header);

        // Header: phis + test
        b.switch_to_block(header);
        // Declare phi without all operands yet; fill in body's
        // contribution after we emit body.
        // We need to know body's outputs up front, so we sketch:
        // first emit body "ahead" by pre-creating fresh vregs.
        let i_next = b.new_vreg();
        let acc_next = b.new_vreg();
        let i_phi = b.phi(vec![(entry_blk, n), (body, i_next)]);
        let acc_phi = b.phi(vec![(entry_blk, zero), (body, acc_next)]);
        b.branch_icmp(Cond::LE, i_phi, zero, exit, body);

        // Body
        b.switch_to_block(body);
        // acc_next = acc + i
        let acc_next_computed = b.add(acc_phi, i_phi);
        b.copy_into(acc_next, acc_next_computed);
        // i_next = i - 1
        let i_next_computed = b.sub(i_phi, one);
        b.copy_into(i_next, i_next_computed);
        b.jmp(header);

        // Exit
        b.switch_to_block(exit);
        b.ret(acc_phi);

        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        for n in [0_i64, 1, 5, 10, 100] {
            let want: i64 = (1..=n).sum();
            assert_eq!(unsafe { f(n) }, want, "n={n}");
        }
    }

    // -------------- Conversion coverage --------------

    #[test]
    fn jit_sext_i32_to_i64_sign_extends_low_dword() {
        let mut b = FuncBuilder::new("sext32");
        let x = b.arg();
        let r = b.sext_i32_to_i64(x);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        // Only the low 32 bits of the i64 arg matter for the instruction;
        // we feed test values that span the sign-bit boundary.
        for x in [0_i32, 1, -1, i32::MAX, i32::MIN, -42] {
            let got = unsafe { f(i64::from(x as u32)) }; // pass zero-ext so upper=0
            assert_eq!(got, i64::from(x), "x={x}");
        }
    }

    #[test]
    fn jit_zext_i32_to_i64_zeros_upper_dword() {
        let mut b = FuncBuilder::new("zext32");
        let x = b.arg();
        let r = b.zext_i32_to_i64(x);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        // Garbage upper bits must be dropped.
        let arg = 0xFFFF_FFFF_DEAD_BEEF_u64 as i64;
        assert_eq!(unsafe { f(arg) }, 0xDEAD_BEEF_i64);
    }

    #[test]
    fn jit_sext_i16_to_i64_sign_extends_low_word() {
        let mut b = FuncBuilder::new("sext16");
        let x = b.arg();
        let r = b.sext_i16_to_i64(x);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [0_i16, 1, -1, i16::MAX, i16::MIN] {
            let got = unsafe { f(i64::from(x as u16)) };
            assert_eq!(got, i64::from(x), "x={x}");
        }
    }

    #[test]
    fn jit_zext_i8_to_i64_zeros_upper_bytes() {
        let mut b = FuncBuilder::new("zext8");
        let x = b.arg();
        let r = b.zext_i8_to_i64(x);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        let arg = 0xDEAD_BEEF_CAFE_00FF_u64 as i64;
        assert_eq!(unsafe { f(arg) }, 0xFF);
    }

    #[test]
    fn jit_trunc_to_i32_keeps_low_dword() {
        let mut b = FuncBuilder::new("trunc32");
        let x = b.arg();
        let r = b.trunc_to_i32(x);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        let arg = 0xFFFF_FFFF_DEAD_BEEF_u64 as i64;
        assert_eq!(unsafe { f(arg) }, 0xDEAD_BEEF_i64);
    }

    #[test]
    fn jit_trunc_to_i8_masks_0xff() {
        let mut b = FuncBuilder::new("trunc8");
        let x = b.arg();
        let r = b.trunc_to_i8(x);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64_I64 = unsafe { m.entry() };
        let arg = 0xDEAD_BEEF_i64;
        assert_eq!(unsafe { f(arg) }, 0xEF);
    }

    // -------------- Memory coverage --------------

    #[test]
    fn jit_alloca_store_load_i64() {
        let mut b = FuncBuilder::new("alloca_rt");
        let x = b.arg();
        let ptr = b.stack_alloc(8, 8);
        b.store_i64(ptr, 0, x);
        let y = b.load_i64(ptr, 0);
        b.ret(y);
        let m = jit(b.build()).expect("jit");
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [0_i64, 1, -1, i64::MAX, i64::MIN, 42] {
            assert_eq!(unsafe { f(x) }, x, "x={x}");
        }
    }

    #[test]
    fn jit_alloca_two_slots_store_load() {
        let mut b = FuncBuilder::new("alloca_two");
        let x = b.arg();
        let y = b.arg();
        let p1 = b.stack_alloc(8, 8);
        let p2 = b.stack_alloc(8, 8);
        b.store_i64(p1, 0, x);
        b.store_i64(p2, 0, y);
        let a = b.load_i64(p1, 0);
        let c = b.load_i64(p2, 0);
        let r = b.add(a, c);
        b.ret(r);
        let m = jit(b.build()).expect("jit");
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for (a, c) in [(1_i64, 2), (5, -3), (100, 200), (i64::MAX, 0)] {
            assert_eq!(unsafe { f(a, c) }, a.wrapping_add(c), "({a},{c})");
        }
    }

    #[test]
    fn jit_alloca_i32_narrow_store_load() {
        let mut b = FuncBuilder::new("alloca_i32");
        let x = b.arg();
        let ptr = b.stack_alloc(4, 4);
        b.store_i32(ptr, 0, x);
        let y = b.load_i32(ptr, 0); // zero-extends to 64 on GPR write
        b.ret(y);
        let m = jit(b.build()).expect("jit");
        let f: FnI64_I64 = unsafe { m.entry() };
        // Pass a value with garbage in upper 32 bits — the store+load
        // round-trip should drop those bits.
        let arg = 0xDEAD_BEEF_CAFE_0123_u64 as i64;
        assert_eq!(unsafe { f(arg) }, 0xCAFE_0123_i64);
    }

    #[test]
    fn jit_alloca_array_gep_indexed_store_load() {
        // int[4] arr; arr[i] = x; return arr[i].
        let mut b = FuncBuilder::new("alloca_arr");
        let x = b.arg();
        let i = b.arg();
        let arr = b.stack_alloc(4 * 8, 8); // 4 * i64
        let slot = b.gep_indexed(arr, i, 8, 0);
        b.store_i64(slot, 0, x);
        // Load back.
        let slot2 = b.gep_indexed(arr, i, 8, 0);
        let y = b.load_i64(slot2, 0);
        b.ret(y);
        let m = jit(b.build()).expect("jit");
        let f: FnI64I64_I64 = unsafe { m.entry() };
        for i in 0_i64..4 {
            let v = 100 + i;
            assert_eq!(unsafe { f(v, i) }, v, "i={i}");
        }
    }

    // -------------- Call coverage --------------

    #[test]
    fn jit_call_labs_returns_absolute_value() {
        // fn(x) -> labs(x)
        let mut b = FuncBuilder::new("call_labs");
        let x = b.arg();
        let r = b.call_sym("labs", &[x]);
        b.ret(r);
        let m = jit(b.build()).expect("jit");
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [0_i64, 1, -1, 42, -42, i64::MAX] {
            assert_eq!(unsafe { f(x) }, x.wrapping_abs(), "x={x}");
        }
    }

    #[test]
    fn jit_self_recursive_factorial() {
        use crate::codegen::isa::x64::inst::Cond;
        // fn fact(n: i64) -> i64 {
        //   if n <= 1 { return 1 } else { return n * fact(n - 1) }
        // }
        let mut b = FuncBuilder::new("fact");
        let n = b.arg();
        let one_const = b.iconst64(1);
        let base_blk = b.new_block();
        let rec_blk = b.new_block();
        b.branch_icmp(Cond::LE, n, one_const, base_blk, rec_blk);
        b.switch_to_block(base_blk);
        let one_ret = b.iconst64(1);
        b.ret(one_ret);
        b.switch_to_block(rec_blk);
        let one_for_sub = b.iconst64(1);
        let n_minus_one = b.sub(n, one_for_sub);
        // Recurse: the symbol must match the function's name.
        let rec = b.call_sym("fact", &[n_minus_one]);
        let result = b.imul(n, rec);
        b.ret(result);
        let m = jit(b.build()).expect("jit fact");
        let f: FnI64_I64 = unsafe { m.entry() };
        let reference = |n: i64| -> i64 {
            let mut r: i64 = 1;
            let mut i: i64 = 2;
            while i <= n {
                r = r.wrapping_mul(i);
                i += 1;
            }
            r
        };
        for n in [0_i64, 1, 2, 5, 10, 12] {
            assert_eq!(unsafe { f(n) }, reference(n), "n={n}");
        }
    }

    #[test]
    fn jit_call_preserves_caller_live_vreg_across_call() {
        // fn(x) -> x + labs(x)
        // `x` must survive across the call to labs. The call clobber
        // markers should ensure `x` isn't left in a caller-saved preg.
        let mut b = FuncBuilder::new("call_live");
        let x = b.arg();
        let abs_x = b.call_sym("labs", &[x]);
        let r = b.add(x, abs_x);
        b.ret(r);
        let m = jit(b.build()).expect("jit");
        let f: FnI64_I64 = unsafe { m.entry() };
        for x in [0_i64, 1, -1, 42, -42, 1000] {
            let want = x.wrapping_add(x.wrapping_abs());
            assert_eq!(unsafe { f(x) }, want, "x={x}");
        }
    }

    #[test]
    fn jit_icmp_to_i64_materializes_boolean() {
        use crate::codegen::isa::x64::inst::Cond;
        let mut b = FuncBuilder::new("eq_as_i64");
        let x = b.arg();
        let y = b.arg();
        let r = b.icmp_to_i64(Cond::Z, x, y);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(3, 3) }, 1);
        assert_eq!(unsafe { f(3, 4) }, 0);
        assert_eq!(unsafe { f(0, 0) }, 1);
        assert_eq!(unsafe { f(-1, 1) }, 0);
    }

    // -------------- SysV calling-convention coverage --------------
    //
    // These tests hammer the integer half of SysV AMD64 from every
    // angle: per-arg-index identity, >6-arg functions that exercise
    // incoming stack-args, calls emitting outgoing stack args, deep
    // recursion that mixes both directions, callee-saved preservation
    // across calls, and 16-byte RSP alignment at call sites.

    /// Build a function that returns exactly the nth argument, where
    /// `n_args` is the total number of declared args. Used to verify
    /// that the ABI places each integer arg in the expected slot.
    fn build_return_nth(name: &str, n_args: u32, pick: u32) -> Func<X64Inst> {
        let mut b = FuncBuilder::new(name);
        let mut args = Vec::with_capacity(n_args as usize);
        for _ in 0..n_args {
            args.push(b.arg());
        }
        b.ret(args[pick as usize]);
        b.build()
    }

    #[test]
    fn jit_fn_zero_args_returns_constant() {
        #[allow(non_camel_case_types)]
        type Fn0 = unsafe extern "sysv64" fn() -> i64;
        let mut b = FuncBuilder::new("zero_args");
        let k = b.iconst64(0xDEAD_BEEF);
        b.ret(k);
        let m = jit(b.build()).unwrap();
        let f: Fn0 = unsafe { m.entry() };
        assert_eq!(unsafe { f() }, 0xDEAD_BEEF);
    }

    #[test]
    fn jit_sysv_each_reg_arg_round_trips() {
        #[allow(non_camel_case_types)]
        type Fn6 = unsafe extern "sysv64" fn(i64, i64, i64, i64, i64, i64) -> i64;
        for pick in 0..6 {
            let m = jit(build_return_nth("pick6", 6, pick)).unwrap();
            let f: Fn6 = unsafe { m.entry() };
            let a = [10_i64, 20, 30, 40, 50, 60];
            assert_eq!(
                unsafe { f(a[0], a[1], a[2], a[3], a[4], a[5]) },
                a[pick as usize],
                "pick={pick}"
            );
        }
    }

    #[test]
    fn jit_sysv_7th_arg_comes_from_stack() {
        #[allow(non_camel_case_types)]
        type Fn7 = unsafe extern "sysv64" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;
        let m = jit(build_return_nth("pick7", 7, 6)).unwrap();
        let f: Fn7 = unsafe { m.entry() };
        // Feed distinct sentinels so a misread reg vs. stack slot shows up.
        let r = unsafe { f(1, 2, 3, 4, 5, 6, 0x7777_7777_7777_7777) };
        assert_eq!(r, 0x7777_7777_7777_7777);
    }

    #[test]
    fn jit_sysv_each_arg_position_in_8_arg_fn() {
        #[allow(non_camel_case_types)]
        type Fn8 = unsafe extern "sysv64" fn(
            i64, i64, i64, i64, i64, i64, i64, i64,
        ) -> i64;
        let vals: [i64; 8] = [
            0x1111_1111_1111_1111,
            0x2222_2222_2222_2222,
            0x3333_3333_3333_3333,
            0x4444_4444_4444_4444,
            0x5555_5555_5555_5555,
            0x6666_6666_6666_6666,
            0x7777_7777_7777_7777_u64 as i64,
            0x0123_4567_89AB_CDEF,
        ];
        for pick in 0..8 {
            let m = jit(build_return_nth("pick8", 8, pick)).unwrap();
            let f: Fn8 = unsafe { m.entry() };
            let got = unsafe {
                f(
                    vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7],
                )
            };
            assert_eq!(
                got, vals[pick as usize],
                "pick={pick} (stack-idx={})",
                (pick as i32) - 6
            );
        }
    }

    #[test]
    fn jit_sysv_sum_of_8_args_exercises_both_stack_slots() {
        #[allow(non_camel_case_types)]
        type Fn8 = unsafe extern "sysv64" fn(
            i64, i64, i64, i64, i64, i64, i64, i64,
        ) -> i64;
        let mut b = FuncBuilder::new("sum8");
        let args: Vec<_> = (0..8).map(|_| b.arg()).collect();
        let mut acc = args[0];
        for a in &args[1..] {
            acc = b.add(acc, *a);
        }
        b.ret(acc);
        let m = jit(b.build()).unwrap();
        let f: Fn8 = unsafe { m.entry() };
        let test_inputs: [[i64; 8]; 4] = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [-1, -2, -3, -4, -5, -6, -7, -8],
            [i64::MAX, 0, 0, 0, 0, 0, 0, 1],
            [100, -50, 25, -10, 5, 0, -99, 200],
        ];
        for v in &test_inputs {
            let want: i64 = v.iter().copied().fold(0_i64, i64::wrapping_add);
            let got = unsafe { f(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]) };
            assert_eq!(got, want, "v={v:?}");
        }
    }

    #[test]
    fn jit_sysv_sum_of_12_args_deep_into_stack() {
        #[allow(non_camel_case_types)]
        type Fn12 = unsafe extern "sysv64" fn(
            i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,
        ) -> i64;
        let mut b = FuncBuilder::new("sum12");
        let args: Vec<_> = (0..12).map(|_| b.arg()).collect();
        let mut acc = args[0];
        for a in &args[1..] {
            acc = b.add(acc, *a);
        }
        b.ret(acc);
        let m = jit(b.build()).unwrap();
        let f: Fn12 = unsafe { m.entry() };
        let got = unsafe { f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) };
        assert_eq!(got, (1..=12).sum());
    }

    #[test]
    fn jit_sysv_sum_of_16_args_all_stack_and_reg() {
        #[allow(non_camel_case_types)]
        type Fn16 = unsafe extern "sysv64" fn(
            i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,
        ) -> i64;
        let mut b = FuncBuilder::new("sum16");
        let args: Vec<_> = (0..16).map(|_| b.arg()).collect();
        let mut acc = args[0];
        for a in &args[1..] {
            acc = b.add(acc, *a);
        }
        b.ret(acc);
        let m = jit(b.build()).unwrap();
        let f: Fn16 = unsafe { m.entry() };
        let got = unsafe {
            f(
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            )
        };
        assert_eq!(got, (1..=16).sum());
    }

    #[test]
    fn jit_sysv_stack_arg_preserves_min_max_int64() {
        // If the stack-arg load path truncated or sign-extended the
        // value incorrectly, these sentinels would surface the bug.
        #[allow(non_camel_case_types)]
        type Fn7 = unsafe extern "sysv64" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;
        let m = jit(build_return_nth("pick7_minmax", 7, 6)).unwrap();
        let f: Fn7 = unsafe { m.entry() };
        for v in [i64::MAX, i64::MIN, -1_i64, 0_i64, 1_i64] {
            assert_eq!(unsafe { f(0, 0, 0, 0, 0, 0, v) }, v, "v={v}");
        }
    }

    #[test]
    fn jit_sysv_outgoing_call_with_7_args_recursion() {
        use crate::codegen::isa::x64::inst::Cond;
        // fn sum(n: i64, a, b, c, d, e, f) -> i64
        //   where args are treated as the accumulator set.
        //   if n <= 0 return a+b+c+d+e+f
        //   else return sum(n-1, a, b, c, d, e, f)
        // Exercises outgoing stack args (7th param) in a recursive call.
        let mut b = FuncBuilder::new("rec7");
        let n = b.arg();
        let a = b.arg();
        let c = b.arg();
        let d = b.arg();
        let e = b.arg();
        let g = b.arg();
        let h = b.arg(); // 7th arg — stack-passed
        let zero = b.iconst64(0);
        let base_blk = b.new_block();
        let rec_blk = b.new_block();
        b.branch_icmp(Cond::LE, n, zero, base_blk, rec_blk);
        b.switch_to_block(base_blk);
        let s1 = b.add(a, c);
        let s2 = b.add(s1, d);
        let s3 = b.add(s2, e);
        let s4 = b.add(s3, g);
        let s5 = b.add(s4, h);
        b.ret(s5);
        b.switch_to_block(rec_blk);
        let one = b.iconst64(1);
        let n_minus_one = b.sub(n, one);
        let r = b.call_sym("rec7", &[n_minus_one, a, c, d, e, g, h]);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        #[allow(non_camel_case_types)]
        type Fn7 = unsafe extern "sysv64" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;
        let f: Fn7 = unsafe { m.entry() };
        // Sum of last 6 args, irrespective of `n` depth.
        for n in [0_i64, 1, 5, 20] {
            let got = unsafe { f(n, 1, 2, 3, 4, 5, 6) };
            assert_eq!(got, 21, "n={n} — last six should still sum to 21");
        }
    }

    #[test]
    fn jit_sysv_outgoing_call_with_8_args_recursion() {
        use crate::codegen::isa::x64::inst::Cond;
        // 8-arg recursion: fn(n, a..g) — test that even-count (no
        // padding needed) stack-arg lowering works.
        let mut b = FuncBuilder::new("rec8");
        let n = b.arg();
        let a1 = b.arg();
        let a2 = b.arg();
        let a3 = b.arg();
        let a4 = b.arg();
        let a5 = b.arg();
        let a6 = b.arg(); // 7th: stack-passed
        let a7 = b.arg(); // 8th: stack-passed
        let zero = b.iconst64(0);
        let base = b.new_block();
        let rec = b.new_block();
        b.branch_icmp(Cond::LE, n, zero, base, rec);
        b.switch_to_block(base);
        let t1 = b.add(a1, a2);
        let t2 = b.add(t1, a3);
        let t3 = b.add(t2, a4);
        let t4 = b.add(t3, a5);
        let t5 = b.add(t4, a6);
        let t6 = b.add(t5, a7);
        b.ret(t6);
        b.switch_to_block(rec);
        let one = b.iconst64(1);
        let np = b.sub(n, one);
        let r = b.call_sym("rec8", &[np, a1, a2, a3, a4, a5, a6, a7]);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        #[allow(non_camel_case_types)]
        type Fn8 = unsafe extern "sysv64" fn(
            i64, i64, i64, i64, i64, i64, i64, i64,
        ) -> i64;
        let f: Fn8 = unsafe { m.entry() };
        for n in [0_i64, 1, 3, 10] {
            let got = unsafe { f(n, 1, 2, 3, 4, 5, 6, 7) };
            assert_eq!(got, 28, "n={n} — (1+2+…+7)=28");
        }
    }

    #[test]
    fn jit_sysv_stack_alignment_is_maintained_across_call_sites() {
        // Call a function that itself calls another function that
        // demands 16-byte-aligned rsp at entry. On x86-64 Linux many
        // libc routines rely on alignment via SSE movdqa; `sqrt` is a
        // classic example. If we miscompute the outgoing-args area
        // size (e.g. 8 instead of 16 for a single stack arg), the
        // call in libc would segfault or give garbage.
        //
        // We'll call `labs` through a wrapper that passes 7 args to a
        // JIT-self recursion, then out of that stack-arg level calls
        // `labs` on the stack-passed value.
        use crate::codegen::isa::x64::inst::Cond;
        let mut b = FuncBuilder::new("align_probe");
        let n = b.arg();
        let pad1 = b.arg();
        let pad2 = b.arg();
        let pad3 = b.arg();
        let pad4 = b.arg();
        let pad5 = b.arg();
        let x = b.arg(); // 7th arg — stack-passed
        let zero = b.iconst64(0);
        let base = b.new_block();
        let rec = b.new_block();
        b.branch_icmp(Cond::LE, n, zero, base, rec);
        b.switch_to_block(base);
        let r = b.call_sym("labs", &[x]);
        b.ret(r);
        b.switch_to_block(rec);
        let one = b.iconst64(1);
        let np = b.sub(n, one);
        let rr = b.call_sym("align_probe", &[np, pad1, pad2, pad3, pad4, pad5, x]);
        b.ret(rr);
        let m = jit(b.build()).unwrap();
        #[allow(non_camel_case_types)]
        type Fn7 = unsafe extern "sysv64" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;
        let f: Fn7 = unsafe { m.entry() };
        for depth in [0_i64, 1, 3, 5] {
            for x in [-42_i64, 0, 42, i64::MAX] {
                let got = unsafe { f(depth, 0, 0, 0, 0, 0, x) };
                assert_eq!(got, x.wrapping_abs(), "depth={depth}, x={x}");
            }
        }
    }

    #[test]
    fn jit_sysv_stack_arg_value_lives_across_later_spilling_work() {
        // Force the regalloc to spill heavily AFTER reading a
        // stack-passed arg. The stack-arg load must persist through
        // the spills without being confused with spill slots on the
        // frame.
        let mut b = FuncBuilder::new("stack_arg_surviving_pressure");
        let _r1 = b.arg();
        let _r2 = b.arg();
        let _r3 = b.arg();
        let _r4 = b.arg();
        let _r5 = b.arg();
        let _r6 = b.arg();
        let stk = b.arg(); // 7th arg — stack-passed
        // Create lots of live values competing for caller-saved pregs.
        let mut acc = stk;
        for k in 0..20 {
            let c = b.iconst64(k);
            acc = b.add(acc, c);
        }
        // Subtract the values we added back so the answer is `stk`.
        for k in 0..20 {
            let c = b.iconst64(k);
            acc = b.sub(acc, c);
        }
        b.ret(acc);
        let m = jit(b.build()).unwrap();
        #[allow(non_camel_case_types)]
        type Fn7 = unsafe extern "sysv64" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;
        let f: Fn7 = unsafe { m.entry() };
        for x in [0_i64, 1, -1, 42, i64::MAX, i64::MIN] {
            let got = unsafe { f(0, 0, 0, 0, 0, 0, x) };
            assert_eq!(got, x, "x={x}");
        }
    }

    #[test]
    fn jit_sysv_outgoing_stack_arg_carries_exact_value() {
        // Test that on the caller side, when we pass a value via a
        // stack slot, the callee receives exactly the same bits. We
        // recurse once with 8 args and verify args[6] and args[7] are
        // preserved bit-exactly.
        use crate::codegen::isa::x64::inst::Cond;
        let mut b = FuncBuilder::new("bits8");
        let n = b.arg();
        let a0 = b.arg();
        let a1 = b.arg();
        let a2 = b.arg();
        let a3 = b.arg();
        let a4 = b.arg();
        let a5 = b.arg(); // 7th — stack
        let a6 = b.arg(); // 8th — stack
        let zero = b.iconst64(0);
        let base = b.new_block();
        let rec = b.new_block();
        b.branch_icmp(Cond::LE, n, zero, base, rec);
        b.switch_to_block(base);
        // Base case: XOR the two stack args and return.
        let x = b.xor(a5, a6);
        b.ret(x);
        b.switch_to_block(rec);
        let one = b.iconst64(1);
        let np = b.sub(n, one);
        let r = b.call_sym("bits8", &[np, a0, a1, a2, a3, a4, a5, a6]);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        #[allow(non_camel_case_types)]
        type Fn8 = unsafe extern "sysv64" fn(
            i64, i64, i64, i64, i64, i64, i64, i64,
        ) -> i64;
        let f: Fn8 = unsafe { m.entry() };
        let a5 = 0x0123_4567_89AB_CDEF_i64;
        let a6 = 0xFEDC_BA98_7654_3210_u64 as i64;
        for n in [0_i64, 1, 4] {
            let got = unsafe { f(n, 0, 0, 0, 0, 0, a5, a6) };
            assert_eq!(got, a5 ^ a6, "n={n}");
        }
    }

    // A global-asm wrapper that installs `sentinel` into every SysV
    // integer callee-saved register, calls `fp` via sysv64 with `arg`
    // in RDI, and returns a bitmask in RAX where bit `i` is set iff
    // the corresponding register did not survive the call.
    //
    //   bit 0 = RBX, bit 1 = R12, bit 2 = R13, bit 3 = R14, bit 4 = R15
    //
    // Signature: fn(sentinel: u64, fp: *const (), arg: i64) -> u64
    //            rdi         rsi            rdx
    std::arch::global_asm!(concat!(
        ".global ", "lancy_check_callee_saved", "\n",
        ".type ", "lancy_check_callee_saved", ", @function\n",
        "lancy_check_callee_saved:\n",
        // Prologue: preserve caller's callee-saveds.
        "push rbx\n",
        "push r12\n",
        "push r13\n",
        "push r14\n",
        "push r15\n",
        // Set callee-saveds to the sentinel (rdi). RSI holds fp,
        // RDX holds the arg the JIT fn should receive in RDI.
        "mov rbx, rdi\n",
        "mov r12, rdi\n",
        "mov r13, rdi\n",
        "mov r14, rdi\n",
        "mov r15, rdi\n",
        // Stash sentinel on stack (we'll need it after the call to
        // compare) and shuffle args: rdi <- rdx (JIT's arg0).
        "push rdi\n",
        "mov  rax, rsi\n",          // fp
        "mov  rdi, rdx\n",          // JIT arg0
        // 16-byte alignment check: we've pushed 6 qwords (5 callees
        // + sentinel) + return addr = 7 qwords = 56 bytes. 56 % 16 == 8,
        // so rsp is 8-misaligned — pad.
        "sub  rsp, 8\n",
        "call rax\n",
        "add  rsp, 8\n",
        "pop  rdi\n",               // sentinel back in rdi
        // Compute bitmask into rax.
        "xor  rax, rax\n",
        "cmp  rbx, rdi\n",
        "je   2f\n",
        "or   rax, 1\n",
        "2:\n",
        "cmp  r12, rdi\n",
        "je   3f\n",
        "or   rax, 2\n",
        "3:\n",
        "cmp  r13, rdi\n",
        "je   4f\n",
        "or   rax, 4\n",
        "4:\n",
        "cmp  r14, rdi\n",
        "je   5f\n",
        "or   rax, 8\n",
        "5:\n",
        "cmp  r15, rdi\n",
        "je   6f\n",
        "or   rax, 16\n",
        "6:\n",
        // Epilogue: restore caller's callee-saveds.
        "pop  r15\n",
        "pop  r14\n",
        "pop  r13\n",
        "pop  r12\n",
        "pop  rbx\n",
        "ret\n",
    ));

    unsafe extern "sysv64" {
        fn lancy_check_callee_saved(sentinel: u64, fp: *const (), arg: i64) -> u64;
    }

    /// Thin Rust wrapper around the asm helper.
    unsafe fn check_callee_saved_preserved(
        fp: unsafe extern "sysv64" fn(i64) -> i64,
        sentinel: u64,
    ) -> u64 {
        unsafe { lancy_check_callee_saved(sentinel, fp as *const (), 0) }
    }

    #[test]
    fn jit_preserves_callee_saved_regs_across_a_simple_fn() {
        // A trivial JIT function still goes through prologue/epilogue.
        // If the emitter forgot to pair push/pop for any callee-saved
        // reg, the sentinel check would catch it.
        let mut b = FuncBuilder::new("trivial");
        let _a = b.arg();
        let k = b.iconst64(42);
        b.ret(k);
        let m = jit(b.build()).unwrap();
        let fp: unsafe extern "sysv64" fn(i64) -> i64 = unsafe { m.entry() };
        let bad = unsafe { check_callee_saved_preserved(fp, 0xCAFE_F00D_1234_5678) };
        assert_eq!(bad, 0, "callee-saved bits clobbered: 0x{bad:x}");
    }

    #[test]
    fn jit_preserves_callee_saved_regs_under_heavy_spills() {
        // Force the emitter to use its callee-saved scratch regs
        // (RBX/R12/R13 in the default config) by generating spills.
        // This mirrors `jit_deep_chain_forces_spills_to_stack` but
        // directly checks that the prologue/epilogue save/restore
        // pair kept the caller's sentinels intact.
        let mut b = FuncBuilder::new("many_sums_cs");
        let a = b.arg();
        let mut vals = vec![a];
        for _ in 0..20 {
            let c = b.iconst64(1);
            let s = b.add(vals[vals.len() - 1], c);
            vals.push(s);
        }
        let mut acc = vals[0];
        for v in &vals[1..] {
            acc = b.add(acc, *v);
        }
        b.ret(acc);
        let m = jit(b.build()).unwrap();
        let fp: unsafe extern "sysv64" fn(i64) -> i64 = unsafe { m.entry() };
        let bad = unsafe { check_callee_saved_preserved(fp, 0x0123_4567_89AB_CDEF) };
        assert_eq!(bad, 0, "callee-saved bits clobbered under spill: 0x{bad:x}");
    }

    #[test]
    fn jit_preserves_callee_saved_regs_across_outgoing_call_with_stack_args() {
        // A JIT function that itself recurses through a 7-arg call
        // (exercises our outgoing-stack-arg emission) must still
        // preserve the caller's callee-saved registers end-to-end.
        //
        // Entry signature is `fn(i64) -> i64`: the helper passes the
        // depth in RDI (our first arg). The remaining 6 arg slots on
        // the JIT side default to whatever the helper leaves there —
        // we don't care about their values, only that the prologue /
        // epilogue + recursive call round-trip preserves the caller's
        // saveds.
        use crate::codegen::isa::x64::inst::Cond;
        let mut b = FuncBuilder::new("rec7_cs");
        let n = b.arg();
        let a1 = b.arg();
        let a2 = b.arg();
        let a3 = b.arg();
        let a4 = b.arg();
        let a5 = b.arg();
        let a6 = b.arg();
        let zero = b.iconst64(0);
        let base = b.new_block();
        let rec = b.new_block();
        b.branch_icmp(Cond::LE, n, zero, base, rec);
        b.switch_to_block(base);
        let s1 = b.add(a1, a2);
        let s2 = b.add(s1, a3);
        let s3 = b.add(s2, a4);
        let s4 = b.add(s3, a5);
        let s5 = b.add(s4, a6);
        b.ret(s5);
        b.switch_to_block(rec);
        let one = b.iconst64(1);
        let np = b.sub(n, one);
        let r = b.call_sym("rec7_cs", &[np, a1, a2, a3, a4, a5, a6]);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        // Coerce to the 1-arg signature the helper drives. The JIT
        // entry sequence is the same: it just reads arg0 from RDI.
        let fp: unsafe extern "sysv64" fn(i64) -> i64 = unsafe { m.entry() };
        let bad = unsafe { check_callee_saved_preserved(fp, 0xDEAD_BEEF_BAAD_F00D) };
        assert_eq!(
            bad, 0,
            "callee-saved bits clobbered across recursive stack-arg call: 0x{bad:x}"
        );
    }

    #[test]
    fn default_ra_config_includes_callee_saved_regs_before_caller_saved() {
        // Callee-saved come first so `max_by_key`'s last-on-tie semantics
        // prefers caller-saved when both are equally free.
        use crate::codegen::isa::x64::regs::*;
        let cfg = default_ra_config(std::collections::HashMap::new());
        assert_eq!(
            cfg.allocatable_regs,
            vec![R14, R15, RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11]
        );
        for s in &cfg.scratch_regs {
            assert!(!cfg.allocatable_regs.contains(s), "scratch {s} overlaps allocatable");
        }
    }

    #[test]
    fn jit_sysv_rdi_not_confused_with_rsi_in_2_arg_fn() {
        // fn(a, b) -> a - b tests directional correctness of the first
        // two arg regs (RDI/RSI): swapping them would invert the sign.
        let mut b = FuncBuilder::new("sub2");
        let x = b.arg();
        let y = b.arg();
        let r = b.sub(x, y);
        b.ret(r);
        let m = jit(b.build()).unwrap();
        let f: FnI64I64_I64 = unsafe { m.entry() };
        assert_eq!(unsafe { f(100, 40) }, 60);
        assert_eq!(unsafe { f(40, 100) }, -60);
    }
}
