//! End-to-end x64 compilation pipeline.
//!
//! `compile(func)` runs every required pass in order and returns the emitted
//! machine-code bytes. `jit(func)` additionally loads the bytes into an
//! executable mapping.

use crate::codegen::analysis::cfg::CFG;
use crate::codegen::isa::x64::inst::X64Inst;
use crate::codegen::isa::x64::mc::emit_mc::FnMCWriter;
use crate::codegen::isa::x64::passes::{abi_lower, pseudo_cleanup};
use crate::codegen::isa::x64::regs::{R10, R11, R12, R13, R8, R9, RAX, RBX, RCX, RDI, RDX, RSI};
use crate::codegen::isa::x64::sysv::SysVAmd64;
use crate::codegen::jit::Module;
use crate::codegen::regalloc::{RegAlloc, RegAllocConfig};
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

/// Compile a function end-to-end. Returns the emitted bytes.
#[must_use]
pub fn compile(mut func: Func<X64Inst>) -> Vec<u8> {
    let abi = abi_lower::lower(&mut func, SysVAmd64);
    let cfg = CFG::compute(&func).expect("CFG compute on valid function");
    let ra_cfg = default_ra_config(abi.reg_bind);
    let mut ra = RegAlloc::new(&func, &cfg, &ra_cfg);
    let ra_res = ra.run();
    pseudo_cleanup::run(&mut func, &ra_res);
    let mut w = FnMCWriter::new(&func, &ra_cfg, &ra_res);
    w.emit_fn()
}

/// Compile a function and load the resulting bytes into an executable mapping.
/// Returns the `Module` (which must outlive any derived function pointers).
///
/// # Errors
/// Propagates `io::Error` from `mmap` / `mprotect` in the JIT runtime.
pub fn jit(func: Func<X64Inst>) -> std::io::Result<Module> {
    let bytes = compile(func);
    Module::load(&bytes)
}

#[cfg(test)]
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
}
