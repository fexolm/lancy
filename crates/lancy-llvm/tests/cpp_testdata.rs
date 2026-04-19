//! Integration tests driven by C++ sources in `tests/testdata/*.cpp`.
//!
//! Each `#[test]` embeds a single `.cpp` via `include_str!`, pipes it
//! through `clang++ -O1 -S -emit-llvm`, feeds the textual LLVM IR to
//! `lancy_llvm::compile_ir_to_jit`, and calls the resulting SysV64
//! function pointer against a small oracle. Tests either pass
//! outright or fail; `#[ignore]` is used for programs the frontend
//! genuinely can't yet handle (typically pulling in large chunks of
//! libstdc++, or LLVM optimizer tricks like i65 intermediates).

use std::io::Write;
use std::process::{Command, Stdio};

use lancy_llvm::compile_ir_to_jit;

#[allow(non_camel_case_types)]
type Fn1 = unsafe extern "sysv64" fn(i64) -> i64;
#[allow(non_camel_case_types)]
type Fn2 = unsafe extern "sysv64" fn(i64, i64) -> i64;
#[allow(non_camel_case_types)]
type Fn3 = unsafe extern "sysv64" fn(i64, i64, i64) -> i64;
#[allow(non_camel_case_types)]
type Fn4 = unsafe extern "sysv64" fn(i64, i64, i64, i64) -> i64;
#[allow(non_camel_case_types)]
type Fn6 = unsafe extern "sysv64" fn(i64, i64, i64, i64, i64, i64) -> i64;

/// Feed `src` to `clang++` and return textual LLVM IR bytes.
fn clang_to_ir(src: &str) -> Vec<u8> {
    let mut child = Command::new("clang++")
        .args([
            "-O1",
            "-std=c++17",
            "-S",
            "-emit-llvm",
            "-fno-exceptions",
            "-fno-rtti",
            "-x",
            "c++",
            "-",
            "-o",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn clang++");
    child
        .stdin
        .as_mut()
        .expect("clang stdin")
        .write_all(src.as_bytes())
        .expect("feed source into clang");
    let out = child.wait_with_output().expect("wait on clang");
    assert!(
        out.status.success(),
        "clang++ failed: {}",
        String::from_utf8_lossy(&out.stderr),
    );
    out.stdout
}

/// Run the end-to-end pipeline; panics on any convert or JIT error.
/// Each integration test either JITs successfully or fails outright —
/// there's no silent skipping.
fn jit_or_panic(src: &str, symbol: &str) -> lancy::codegen::jit::Module {
    let ir = clang_to_ir(src);
    compile_ir_to_jit(&ir, symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"))
}

// -------------------------------------------------------------------
// Straight-line arithmetic — expected to pass on the current frontend.
// -------------------------------------------------------------------

#[test]
fn cpp_add() {
    let m = jit_or_panic(include_str!("testdata/add.cpp"), "add");
    let f: Fn2 = unsafe { m.entry() };
    for (a, b) in [(0, 0), (1, 2), (-5, 5), (i64::MAX, 0), (100, -50)] {
        assert_eq!(unsafe { f(a, b) }, a.wrapping_add(b), "({a},{b})");
    }
}

#[test]
fn cpp_sub() {
    let m = jit_or_panic(include_str!("testdata/sub.cpp"), "sub");
    let f: Fn2 = unsafe { m.entry() };
    for (a, b) in [(10, 3), (0, 1), (i64::MIN, 0), (-7, -3)] {
        assert_eq!(unsafe { f(a, b) }, a.wrapping_sub(b), "({a},{b})");
    }
}

#[test]
fn cpp_mul() {
    let m = jit_or_panic(include_str!("testdata/mul.cpp"), "mul");
    let f: Fn2 = unsafe { m.entry() };
    for (a, b) in [(6, 7), (-3, 4), (0, 100), (123_456, 789), (-1, i64::MIN)] {
        assert_eq!(unsafe { f(a, b) }, a.wrapping_mul(b), "({a},{b})");
    }
}

#[test]
fn cpp_mad() {
    let m = jit_or_panic(include_str!("testdata/mad.cpp"), "mad");
    let f: Fn3 = unsafe { m.entry() };
    for (a, b, c) in [(2, 3, 4), (-1, 10, 5), (0, 999, -7), (i64::MAX, 0, 1)] {
        let want = a.wrapping_mul(b).wrapping_add(c);
        assert_eq!(unsafe { f(a, b, c) }, want, "({a},{b},{c})");
    }
}

#[test]
fn cpp_cube_sum() {
    let m = jit_or_panic(include_str!("testdata/cube_sum.cpp"), "cube_sum");
    let f: Fn2 = unsafe { m.entry() };
    for (x, y) in [(1_i64, 2), (3, 5), (-4, 7), (0, 9), (-10, -3)] {
        let sum = x.wrapping_add(y);
        let want = sum.wrapping_mul(sum).wrapping_mul(sum);
        assert_eq!(unsafe { f(x, y) }, want, "({x},{y})");
    }
}

#[test]
fn cpp_diff_squares() {
    let m = jit_or_panic(
        include_str!("testdata/diff_squares.cpp"),
        "diff_squares",
    );
    let f: Fn2 = unsafe { m.entry() };
    for (a, b) in [(3_i64, 2_i64), (10, 1), (-5, 4), (7, -7), (100, 99)] {
        let want = a.wrapping_mul(a).wrapping_sub(b.wrapping_mul(b));
        assert_eq!(unsafe { f(a, b) }, want, "({a},{b})");
    }
}

#[test]
fn cpp_horner() {
    let m = jit_or_panic(include_str!("testdata/horner.cpp"), "horner");
    let f: Fn1 = unsafe { m.entry() };
    let reference = |x: i64| -> i64 {
        let mut p: i64 = 3;
        p = p.wrapping_mul(x).wrapping_add(5);
        p = p.wrapping_mul(x).wrapping_sub(2);
        p = p.wrapping_mul(x).wrapping_add(7);
        p = p.wrapping_mul(x).wrapping_sub(1);
        p
    };
    for x in [-3, -1, 0, 1, 2, 5, 10] {
        assert_eq!(unsafe { f(x) }, reference(x), "x={x}");
    }
}

#[test]
fn cpp_weighted_sum() {
    let m = jit_or_panic(
        include_str!("testdata/weighted_sum.cpp"),
        "weighted_sum",
    );
    let f: Fn4 = unsafe { m.entry() };
    for (a, b, c, d) in [
        (1, 1, 1, 1),
        (10, -2, 3, 4),
        (-5, -5, -5, -5),
        (0, 7, 11, -3),
    ] {
        let want = 2 * a + 3 * b - 5 * c + 7 * d;
        assert_eq!(unsafe { f(a, b, c, d) }, want, "({a},{b},{c},{d})");
    }
}

#[test]
fn cpp_six_args() {
    let m = jit_or_panic(include_str!("testdata/six_args.cpp"), "six_args");
    let f: Fn6 = unsafe { m.entry() };
    for (a, b, c, d, e, g) in [
        (1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64),
        (-1, -2, -3, -4, -5, -6),
        (10, 20, 5, 3, 100, -50),
        (0, 0, 0, 1, 0, 1),
    ] {
        let want = a
            .wrapping_add(b)
            .wrapping_mul(c.wrapping_sub(d))
            .wrapping_add(e.wrapping_mul(g))
            .wrapping_sub(a);
        assert_eq!(
            unsafe { f(a, b, c, d, e, g) },
            want,
            "({a},{b},{c},{d},{e},{g})"
        );
    }
}

#[test]
fn cpp_stress_arith() {
    let m = jit_or_panic(
        include_str!("testdata/stress_arith.cpp"),
        "stress_arith",
    );
    let f: Fn2 = unsafe { m.entry() };
    let reference = |a: i64, b: i64| -> i64 {
        let t1 = a.wrapping_add(b);
        let t2 = a.wrapping_sub(b);
        let t3 = a.wrapping_mul(b);
        let t4 = t1.wrapping_mul(t2);
        let t5 = t3.wrapping_add(t4);
        let t6 = t5.wrapping_mul(a);
        let t7 = t6.wrapping_add(t5);
        let t8 = t7.wrapping_sub(t1);
        let t9 = t8.wrapping_mul(t2);
        let t10 = t9.wrapping_add(t3);
        let t11 = t10.wrapping_mul(t4);
        let t12 = t11.wrapping_sub(t6);
        t12.wrapping_add(t10)
    };
    for (a, b) in [(1, 2), (3, -5), (-7, 11), (100, 101), (-1, 0)] {
        assert_eq!(unsafe { f(a, b) }, reference(a, b), "({a},{b})");
    }
}

#[test]
fn cpp_fib_unrolled() {
    let m = jit_or_panic(
        include_str!("testdata/fib_unrolled.cpp"),
        "fib_unrolled",
    );
    let f: Fn1 = unsafe { m.entry() };
    let reference = |seed: i64| -> i64 {
        let mut a = seed;
        let mut b = seed.wrapping_add(1);
        for _ in 0..10 {
            let c = a.wrapping_add(b);
            a = b;
            b = c;
        }
        b
    };
    for seed in [0, 1, 2, 5, 10, -3] {
        assert_eq!(unsafe { f(seed) }, reference(seed), "seed={seed}");
    }
}

// -------------------------------------------------------------------
// Harder tests — expected to skip today. Inputs/oracles are still
// written out so the case goes green the moment the frontend catches
// up.
// -------------------------------------------------------------------

#[test]
fn cpp_abs_val() {
    let m = jit_or_panic(include_str!("testdata/abs_val.cpp"), "abs_val");
    let f: Fn1 = unsafe { m.entry() };
    for x in [0_i64, 1, -1, 42, -42, i64::MAX] {
        assert_eq!(unsafe { f(x) }, x.wrapping_abs(), "x={x}");
    }
}

#[test]
fn cpp_my_max() {
    let m = jit_or_panic(include_str!("testdata/my_max.cpp"), "my_max");
    let f: Fn2 = unsafe { m.entry() };
    for (a, b) in [(1, 2), (5, 3), (-1, -1), (i64::MIN, i64::MAX)] {
        assert_eq!(unsafe { f(a, b) }, a.max(b), "({a},{b})");
    }
}

#[test]
fn cpp_clamp() {
    let m = jit_or_panic(include_str!("testdata/clamp.cpp"), "clamp");
    let f: Fn3 = unsafe { m.entry() };
    for (x, lo, hi) in [(5, 0, 10), (-3, 0, 10), (100, 0, 10), (7, 7, 7)] {
        assert_eq!(unsafe { f(x, lo, hi) }, x.clamp(lo, hi), "({x},{lo},{hi})");
    }
}

#[test]
fn cpp_sign() {
    let m = jit_or_panic(include_str!("testdata/sign.cpp"), "sign");
    let f: Fn1 = unsafe { m.entry() };
    for x in [-5_i64, -1, 0, 1, 5, i64::MAX, i64::MIN] {
        assert_eq!(unsafe { f(x) }, x.signum(), "x={x}");
    }
}

#[test]
fn cpp_fib_iter() {
    let m = jit_or_panic(include_str!("testdata/fib_iter.cpp"), "fib_iter");
    let f: Fn1 = unsafe { m.entry() };
    let fib = |n: i64| -> i64 {
        let (mut a, mut b) = (0i64, 1i64);
        for _ in 0..n {
            let c = a.wrapping_add(b);
            a = b;
            b = c;
        }
        a
    };
    for n in [0, 1, 2, 5, 10, 20, 50] {
        assert_eq!(unsafe { f(n) }, fib(n), "n={n}");
    }
}

#[test]
fn cpp_fib_recursive() {
    let m = jit_or_panic(
        include_str!("testdata/fib_recursive.cpp"),
        "fib_recursive",
    );
    let f: Fn1 = unsafe { m.entry() };
    fn rfib(n: i64) -> i64 {
        if n < 2 { n } else { rfib(n - 1) + rfib(n - 2) }
    }
    for n in [0, 1, 2, 5, 10, 15] {
        assert_eq!(unsafe { f(n) }, rfib(n), "n={n}");
    }
}

#[test]
fn cpp_factorial_iter() {
    let m = jit_or_panic(
        include_str!("testdata/factorial_iter.cpp"),
        "factorial_iter",
    );
    let f: Fn1 = unsafe { m.entry() };
    let fact = |n: i64| -> i64 {
        let mut r = 1i64;
        let mut i = 2i64;
        while i <= n {
            r = r.wrapping_mul(i);
            i += 1;
        }
        r
    };
    for n in [0, 1, 5, 10, 15, 20] {
        assert_eq!(unsafe { f(n) }, fact(n), "n={n}");
    }
}

#[test]
fn cpp_gcd_euclid() {
    let m = jit_or_panic(include_str!("testdata/gcd_euclid.cpp"), "gcd_euclid");
    let f: Fn2 = unsafe { m.entry() };
    fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }
    for (a, b) in [(12, 8), (100, 75), (17, 5), (0, 9), (270, 192)] {
        assert_eq!(unsafe { f(a, b) }, gcd(a, b), "({a},{b})");
    }
}

#[test]
fn cpp_pow_int() {
    let m = jit_or_panic(include_str!("testdata/pow_int.cpp"), "pow_int");
    let f: Fn2 = unsafe { m.entry() };
    fn pw(mut base: i64, mut exp: i64) -> i64 {
        let mut r: i64 = 1;
        while exp > 0 {
            if exp & 1 != 0 {
                r = r.wrapping_mul(base);
            }
            base = base.wrapping_mul(base);
            exp >>= 1;
        }
        r
    }
    for (b, e) in [(2, 10), (3, 5), (7, 0), (-2, 7), (1, 100)] {
        assert_eq!(unsafe { f(b, e) }, pw(b, e), "({b},{e})");
    }
}

#[test]
fn cpp_sum_to_n() {
    let m = jit_or_panic(include_str!("testdata/sum_to_n.cpp"), "sum_to_n");
    let f: Fn1 = unsafe { m.entry() };
    for n in [0_i64, 1, 10, 100, 1000] {
        let want = (1..=n).sum::<i64>();
        assert_eq!(unsafe { f(n) }, want, "n={n}");
    }
}

// TODO: `std::vector<long>::push_back` emits an out-of-line
// `_M_realloc_insert` template instantiation that only lives in the
// .o file produced by clang — not in `libstdc++.so`. Our JIT reads
// only the LLVM IR, so `dlsym` can't find that symbol at load time.
// Proper support needs either an .o-file loader alongside the IR or
// an IR-level re-translation of the template body.
#[test]
#[ignore = "needs .o-file loading: vector template instantiations aren't in libstdc++.so"]
fn cpp_vector_sum() {
    let m = jit_or_panic(include_str!("testdata/vector_sum.cpp"), "vector_sum");
    let f: Fn1 = unsafe { m.entry() };
    for n in [0_i64, 1, 5, 10, 100] {
        let want = (1..=n).sum::<i64>();
        assert_eq!(unsafe { f(n) }, want, "n={n}");
    }
}

// TODO: std::vector + std::sort require full libstdc++ cooperation
// (new/delete, comparator instantiation, iterator traits). Same
// caveat as `cpp_string_build`.
#[test]
#[ignore = "depends on libstdc++ ABI quirks beyond the JIT's current reach"]
fn cpp_vector_sort_median() {
    let m = jit_or_panic(
        include_str!("testdata/vector_sort_median.cpp"),
        "vector_sort_median",
    );
    let f: Fn3 = unsafe { m.entry() };
    for (a, b, c) in [(1, 2, 3), (3, 1, 2), (5, 5, 5), (-1, -2, -3), (10, 0, 5)] {
        let mut v = [a, b, c];
        v.sort();
        assert_eq!(unsafe { f(a, b, c) }, v[1], "({a},{b},{c})");
    }
}

// TODO: `std::map<long,long>::operator[]` is a template instantiation
// emitted inside the user's .o file; it isn't part of `libstdc++.so`,
// so `dlsym` can't resolve it from the IR alone. Same infrastructure
// need as `cpp_vector_sum` — a real C++ runtime story for the JIT.
#[test]
#[ignore = "needs .o-file loading: map template instantiations aren't in libstdc++.so"]
fn cpp_map_square_lookup() {
    let m = jit_or_panic(
        include_str!("testdata/map_square_lookup.cpp"),
        "map_square_lookup",
    );
    let f: Fn1 = unsafe { m.entry() };
    for key in [0_i64, 1, 5, 9, 10, -1] {
        let want = if (0..10).contains(&key) { key * key } else { -1 };
        assert_eq!(unsafe { f(key) }, want, "key={key}");
    }
}

// TODO: std::string construction requires calling into libstdc++'s
// `_M_construct` helpers; the JIT's direct `dlsym`-resolved call
// works but the SSO + heap-pointer interplay and mangled-symbol
// contract produces a garbage read — investigate once we have a
// C++ runtime story.
#[test]
#[ignore = "depends on libstdc++ ABI quirks beyond the JIT's current reach"]
fn cpp_string_build() {
    let m = jit_or_panic(include_str!("testdata/string_build.cpp"), "string_build");
    let f: Fn1 = unsafe { m.entry() };
    for n in [0_i64, 1, 16, 32, 64, 1024] {
        assert_eq!(unsafe { f(n) }, n, "n={n}");
    }
}
