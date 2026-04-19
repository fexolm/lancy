//! Tests against textual LLVM IR that exercises multi-block control
//! flow. Clang at `-O1` aggressively lowers `if/else/return` into
//! `select` or `llvm.smax`-style intrinsics that the lancy frontend
//! doesn't support, so we hand-write the shapes we do support (each
//! arm owns its own `ret`, no phi required) instead.

use lancy_llvm::compile_ir_to_jit;

#[allow(non_camel_case_types)]
type Fn1 = unsafe extern "sysv64" fn(i64) -> i64;
#[allow(non_camel_case_types)]
type Fn2 = unsafe extern "sysv64" fn(i64, i64) -> i64;

const BRANCH_MAX_IR: &str = r#"
define i64 @max(i64 %a, i64 %b) {
entry:
  %cmp = icmp sgt i64 %a, %b
  br i1 %cmp, label %then, label %else

then:
  ret i64 %a

else:
  ret i64 %b
}
"#;

#[test]
fn branch_max_picks_larger_value() {
    let m = compile_ir_to_jit(BRANCH_MAX_IR.as_bytes(), "max").expect("convert + jit");
    let f: Fn2 = unsafe { m.entry() };
    assert_eq!(unsafe { f(5, 3) }, 5);
    assert_eq!(unsafe { f(3, 5) }, 5);
    assert_eq!(unsafe { f(-10, -20) }, -10);
    assert_eq!(unsafe { f(7, 7) }, 7);
    assert_eq!(unsafe { f(i64::MIN, i64::MAX) }, i64::MAX);
}

const BRANCH_SIGN_IR: &str = r#"
define i64 @sgn(i64 %x) {
entry:
  %c0 = icmp sgt i64 %x, 0
  br i1 %c0, label %pos, label %not_pos

not_pos:
  %c1 = icmp slt i64 %x, 0
  br i1 %c1, label %neg, label %zero

pos:
  ret i64 1

neg:
  ret i64 -1

zero:
  ret i64 0
}
"#;

#[test]
fn branch_sign_classifies() {
    let m = compile_ir_to_jit(BRANCH_SIGN_IR.as_bytes(), "sgn").expect("convert + jit");
    let f: Fn1 = unsafe { m.entry() };
    assert_eq!(unsafe { f(0) }, 0);
    assert_eq!(unsafe { f(1) }, 1);
    assert_eq!(unsafe { f(-1) }, -1);
    assert_eq!(unsafe { f(i64::MAX) }, 1);
    assert_eq!(unsafe { f(i64::MIN) }, -1);
    assert_eq!(unsafe { f(100) }, 1);
    assert_eq!(unsafe { f(-100) }, -1);
}

const EQ_BRANCH_IR: &str = r#"
define i64 @eq_or(i64 %a, i64 %b) {
entry:
  %c = icmp eq i64 %a, %b
  br i1 %c, label %equal, label %not_equal

equal:
  ret i64 1

not_equal:
  ret i64 0
}
"#;

#[test]
fn branch_equals_returns_boolean_i64() {
    let m = compile_ir_to_jit(EQ_BRANCH_IR.as_bytes(), "eq_or").expect("convert + jit");
    let f: Fn2 = unsafe { m.entry() };
    assert_eq!(unsafe { f(5, 5) }, 1);
    assert_eq!(unsafe { f(5, 6) }, 0);
    assert_eq!(unsafe { f(0, 0) }, 1);
    assert_eq!(unsafe { f(i64::MIN, i64::MAX) }, 0);
}

const FIB_LOOP_IR: &str = r#"
define i64 @fib_iter(i64 %n) {
entry:
  %cond = icmp sgt i64 %n, 0
  br i1 %cond, label %loop, label %exit

loop:
  %a = phi i64 [ 0, %entry ], [ %b, %loop ]
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %b = phi i64 [ 1, %entry ], [ %next, %loop ]
  %next = add nsw i64 %a, %b
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  %r = phi i64 [ 0, %entry ], [ %b, %loop ]
  ret i64 %r
}
"#;

const ALLOCA_ROUNDTRIP_IR: &str = r#"
define i64 @alloca_roundtrip(i64 %x) {
  %p = alloca i64, align 8
  store i64 %x, ptr %p, align 8
  %r = load i64, ptr %p, align 8
  ret i64 %r
}
"#;

#[test]
fn alloca_store_load_roundtrip_jits() {
    let m = compile_ir_to_jit(ALLOCA_ROUNDTRIP_IR.as_bytes(), "alloca_roundtrip")
        .expect("jit");
    let f: Fn1 = unsafe { m.entry() };
    for x in [0_i64, 1, -1, i64::MIN, i64::MAX] {
        assert_eq!(unsafe { f(x) }, x, "x={x}");
    }
}

const GEP_ARRAY_IR: &str = r#"
define i64 @array_read(i64 %i) {
  %arr = alloca [4 x i64], align 16
  %p0 = getelementptr inbounds [4 x i64], ptr %arr, i64 0, i64 0
  store i64 10, ptr %p0, align 8
  %p1 = getelementptr inbounds [4 x i64], ptr %arr, i64 0, i64 1
  store i64 20, ptr %p1, align 8
  %p2 = getelementptr inbounds [4 x i64], ptr %arr, i64 0, i64 2
  store i64 30, ptr %p2, align 8
  %p3 = getelementptr inbounds [4 x i64], ptr %arr, i64 0, i64 3
  store i64 40, ptr %p3, align 8
  %pi = getelementptr inbounds [4 x i64], ptr %arr, i64 0, i64 %i
  %r = load i64, ptr %pi, align 8
  ret i64 %r
}
"#;

#[test]
fn gep_into_alloca_array_reads_correct_element() {
    let m = compile_ir_to_jit(GEP_ARRAY_IR.as_bytes(), "array_read").expect("jit");
    let f: Fn1 = unsafe { m.entry() };
    assert_eq!(unsafe { f(0) }, 10);
    assert_eq!(unsafe { f(1) }, 20);
    assert_eq!(unsafe { f(2) }, 30);
    assert_eq!(unsafe { f(3) }, 40);
}

const SWITCH_IR: &str = r#"
define i64 @select(i64 %x) {
entry:
  switch i64 %x, label %default [
    i64 1, label %c1
    i64 2, label %c2
    i64 3, label %c3
  ]

c1:
  ret i64 100

c2:
  ret i64 200

c3:
  ret i64 300

default:
  ret i64 -1
}
"#;

const FENCE_FREEZE_IR: &str = r#"
define i64 @fence_freeze(i64 %x) {
  fence seq_cst
  %y = freeze i64 %x
  %z = add i64 %y, 1
  ret i64 %z
}
"#;

#[test]
fn fence_and_freeze_lower_correctly() {
    let m = compile_ir_to_jit(FENCE_FREEZE_IR.as_bytes(), "fence_freeze").expect("jit");
    let f: Fn1 = unsafe { m.entry() };
    for x in [0_i64, 1, -1, 41, i64::MAX - 1] {
        assert_eq!(unsafe { f(x) }, x.wrapping_add(1), "x={x}");
    }
}

#[test]
fn switch_dispatches_each_case() {
    let m = compile_ir_to_jit(SWITCH_IR.as_bytes(), "select").expect("jit");
    let f: Fn1 = unsafe { m.entry() };
    assert_eq!(unsafe { f(1) }, 100);
    assert_eq!(unsafe { f(2) }, 200);
    assert_eq!(unsafe { f(3) }, 300);
    assert_eq!(unsafe { f(0) }, -1);
    assert_eq!(unsafe { f(42) }, -1);
}

#[test]
fn fib_iter_hand_written_via_phi_jits_correctly() {
    let m = compile_ir_to_jit(FIB_LOOP_IR.as_bytes(), "fib_iter").expect("convert + jit");
    let f: Fn1 = unsafe { m.entry() };
    let fib = |n: i64| -> i64 {
        let (mut a, mut b) = (0_i64, 1_i64);
        for _ in 0..n {
            let c = a.wrapping_add(b);
            a = b;
            b = c;
        }
        a
    };
    for n in [0_i64, 1, 2, 5, 10] {
        assert_eq!(unsafe { f(n) }, fib(n), "n={n}");
    }
}
