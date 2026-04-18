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
