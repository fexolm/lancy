//! End-to-end x64 compilation pipeline.
//!
//! `compile(func)` runs every required pass in order and returns the emitted
//! machine-code bytes. `jit(func)` additionally loads the bytes into an
//! executable mapping.

use crate::codegen::analysis::cfg::CFG;
use crate::codegen::isa::x64::inst::X64Inst;
use crate::codegen::isa::x64::mc::emit_mc::FnMCWriter;
use crate::codegen::isa::x64::passes::abi_lower::SysVAmd64Lowering;
use crate::codegen::isa::x64::regs::{R10, R11, R12, R13, R8, R9, RAX, RBX, RCX, RDI, RDX, RSI};
use crate::codegen::jit::{Module, Relocation};
use crate::codegen::passes::{AbiLowering, destroy_ssa};
use crate::codegen::regalloc::{LinearScan, RegAllocConfig, RegAllocator};
use crate::codegen::tir::{Func, Reg};
use std::collections::HashMap;

/// Build the default `SysV`-flavored `RegAllocConfig`. The allocatable pool is
/// the nine caller-saved integer registers (`RAX/RCX/RDX/RSI/RDI/R8..R11`).
/// `RBX`, `R12`, `R13` are reserved as spill scratches — they are
/// callee-saved under `SysV`, so the MC emitter's prologue preserves them
/// explicitly. Three scratches is the worst case needed by
/// `Mov64rm { src: Mem { base, index: Some, .. }, dst }` when `base`,
/// `index`, and `dst` are all spilled simultaneously.
#[must_use]
pub fn default_ra_config(reg_bind: HashMap<Reg, Reg>) -> RegAllocConfig {
    RegAllocConfig {
        preg_count: 32,
        allocatable_regs: vec![RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11],
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
}
