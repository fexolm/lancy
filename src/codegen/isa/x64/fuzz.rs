//! Differential fuzzing of the end-to-end pipeline.
//!
//! Inspired by regalloc2's fuzz harness: instead of asserting on internal
//! invariants, generate random programs, compile them through the full
//! pipeline (ABI lower → regalloc → pseudo cleanup → MC emit → JIT load),
//! execute the resulting machine code, and compare the result against an
//! in-process Rust oracle evaluating the same op sequence.
//!
//! Any mismatch means something in the pipeline produced wrong code — most
//! commonly a regalloc bug (wrong register, forgotten spill reload, coalesce
//! that clobbered a still-live value). The generator leans on register
//! pressure: programs with many operations over a fixed set of source vregs
//! exercise spilling and coalescing paths.

use crate::codegen::isa::x64::builder::FuncBuilder;
use crate::codegen::isa::x64::pipeline::jit;
use crate::codegen::tir::Reg;

#[allow(non_camel_case_types)]
type Fn2 = unsafe extern "sysv64" fn(i64, i64) -> i64;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        // Decorrelate adjacent seeds so `seed` / `seed+1` don't produce
        // near-identical sequences.
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xCAFE_BABE_DEAD_BEEF)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn pick(&mut self, modulo: usize) -> usize {
        (self.next() as usize) % modulo
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Const(i64),
}

fn eval(ops: &[Op], a: i64, c: i64) -> i64 {
    let mut vals = vec![a, c];
    for &op in ops {
        let v = match op {
            Op::Add(i, j) => vals[i].wrapping_add(vals[j]),
            Op::Sub(i, j) => vals[i].wrapping_sub(vals[j]),
            Op::Mul(i, j) => vals[i].wrapping_mul(vals[j]),
            Op::Const(k) => k,
        };
        vals.push(v);
    }
    *vals.last().unwrap()
}

fn gen_and_build(seed: u64, n_ops: usize) -> (crate::codegen::tir::Func<crate::codegen::isa::x64::inst::X64Inst>, Vec<Op>) {
    let mut rng = Lcg::new(seed);
    let mut b = FuncBuilder::new(format!("fuzz_{seed}"));
    let a = b.arg();
    let c = b.arg();
    let mut vals: Vec<Reg> = vec![a, c];
    let mut ops_log: Vec<Op> = Vec::with_capacity(n_ops);

    for _ in 0..n_ops {
        let kind = rng.pick(4);
        let op = match kind {
            0 => {
                let i = rng.pick(vals.len());
                let j = rng.pick(vals.len());
                let v = b.add(vals[i], vals[j]);
                vals.push(v);
                Op::Add(i, j)
            }
            1 => {
                let i = rng.pick(vals.len());
                let j = rng.pick(vals.len());
                let v = b.sub(vals[i], vals[j]);
                vals.push(v);
                Op::Sub(i, j)
            }
            2 => {
                let i = rng.pick(vals.len());
                let j = rng.pick(vals.len());
                let v = b.imul(vals[i], vals[j]);
                vals.push(v);
                Op::Mul(i, j)
            }
            _ => {
                // Deliberately small-ish immediate so overflow behavior stays
                // predictable between oracle and JIT (both use wrapping).
                let imm = (rng.next() as i64) >> 48;
                let v = b.iconst64(imm);
                vals.push(v);
                Op::Const(imm)
            }
        };
        ops_log.push(op);
    }

    let last = *vals.last().unwrap();
    b.ret(last);
    (b.build(), ops_log)
}

fn run_one(seed: u64, n_ops: usize, sample_inputs: &[(i64, i64)]) {
    let (func, ops) = gen_and_build(seed, n_ops);
    let module = jit(func).expect("jit load");
    let f: Fn2 = unsafe { module.entry() };
    for &(x, y) in sample_inputs {
        let got = unsafe { f(x, y) };
        let want = eval(&ops, x, y);
        assert_eq!(
            got, want,
            "seed={seed}, n_ops={n_ops}, inputs=({x},{y}): JIT returned {got}, oracle says {want}.\nOps: {ops:?}"
        );
    }
}

const SAMPLE_INPUTS: &[(i64, i64)] = &[
    (0, 0),
    (1, 1),
    (-1, 1),
    (7, -11),
    (123_456, -78_910),
    (i64::MIN, 1),
    (i64::MAX, -1),
];

#[test]
fn fuzz_short_programs() {
    // Short programs (<=10 ops) rarely spill; they exercise coalescing,
    // pre-binds, and straightforward allocation paths.
    for seed in 1..=100 {
        let n_ops = 1 + (seed as usize % 10);
        run_one(seed, n_ops, SAMPLE_INPUTS);
    }
}

#[test]
fn fuzz_pressure_programs() {
    // Longer programs reliably exceed the 9-preg SysV allocatable pool and
    // force spill/reload paths. Keep the count modest so the test runs in
    // well under a second.
    for seed in 1..=30 {
        let n_ops = 20 + (seed as usize % 30);
        run_one(seed, n_ops, SAMPLE_INPUTS);
    }
}

#[test]
fn fuzz_deep_programs() {
    // 80+ ops guarantees heavy spilling and many coalesce candidates as
    // copies pile up. A handful of seeds keeps wall time down.
    for seed in 1..=8 {
        run_one(seed, 80, SAMPLE_INPUTS);
    }
}

#[cfg(test)]
mod sanity {
    use super::*;

    #[test]
    fn oracle_matches_direct_computation_on_a_known_program() {
        // Build "a + c" by hand and confirm `eval` agrees.
        let ops = vec![Op::Add(0, 1)];
        assert_eq!(eval(&ops, 3, 4), 7);
        assert_eq!(eval(&ops, -1, 1), 0);
    }

    #[test]
    fn generator_emits_at_least_one_op_of_each_kind_over_many_seeds() {
        // Coverage sanity: the generator shouldn't degenerate into one op
        // type across a reasonable seed sweep.
        let mut seen_add = false;
        let mut seen_sub = false;
        let mut seen_mul = false;
        let mut seen_const = false;
        for seed in 1..=50 {
            let (_f, ops) = gen_and_build(seed, 20);
            for op in ops {
                match op {
                    Op::Add(..) => seen_add = true,
                    Op::Sub(..) => seen_sub = true,
                    Op::Mul(..) => seen_mul = true,
                    Op::Const(_) => seen_const = true,
                }
            }
        }
        assert!(seen_add && seen_sub && seen_mul && seen_const);
    }
}
