# Lancy

Lightweight JIT compiler backend. Intended niche: database query execution and similar hot-path scenarios where LLVM JIT's compile time is prohibitive. Long-term goal — production-quality JIT competitive with Cranelift in its niche, with richer SIMD and flexible calling conventions.

## Scope

**In scope:**
- Target-level non-SSA IR (MIR-analog) with target-specific instructions plus `PseudoInstruction` for cross-target shared logic.
- Typed IR: I8/I16/I32/I64, F32/F64, Ptr, V128/V256/V512 over scalar types.
- Fast register allocation — linear scan with move coalescing, spill heuristics, reg hints.
- Multiple calling conventions: SysV AMD64 + custom fastcall for hot paths.
- JIT runtime: mmap PROT_EXEC, calling wrapper, debug symbols.
- SIMD first-class: SSE → AVX → AVX-512.
- Multiple frontends over time. v0: Rust builder API. Planned: LLVM IR converter.
- Targets: x86_64 first, AArch64 later.

**Non-goals:**
- High-level optimizations (GVN, LICM, loop transforms, inlining) — frontend's responsibility.
- Autovectorization — input is already vectorized.
- Alias analysis — frontend provides well-formed memory ops.
- Instruction selection as a separate pass — frontends emit target-level IR directly (LLVM-IR frontend does its own selection during conversion).

## Key invariants

- **IR is formally non-SSA**, but early pipeline preserves SSA by convention. Input is SSA; passes before register allocation maintain it; after regalloc and spill rewriting SSA no longer holds. **No runtime flag — this is a contract between passes, not a checked property.**
- **`PseudoInstruction` is the mechanism for cross-target shared logic**, not a separate IR tier. One IR; instructions are either target-specific (`X64Inst`) or target-neutral (pseudo). See `docs/ARCHITECTURE.md` for the closed pseudo-inst set.
- **Every `Reg` carries a `Type`.** Instruction operand positions imply types; mismatches are invariant violations (debug-asserted).
- **No `todo!()` in merged code.** Every instruction fully implements `get_uses` / `get_defs` / `Display` / machine-code emission.
- **`cargo clippy --all-targets -- -D warnings` must pass** on every commit.

## Pipeline (high level)

1. Frontend emits target IR in SSA shape, using `PseudoInstruction::{Arg, Phi, CallPseudo, StackAlloc, …}` for target-neutral concerns.
2. Analyses: CFG, liveness, dominance — all generic over `Inst`.
3. SSA destruction pass: `Phi` → parallel `Copy`s in predecessors.
4. ABI lowering pass: `Arg`/`CallPseudo` → concrete register/stack moves per calling convention.
5. Register allocation: linear scan, `Copy` as coalescing candidate, `RegDef` as pre-binding constraint.
6. Pseudo cleanup: `Kill`/`ImplicitDef` erased; surviving `Copy` → MOV or elided.
7. Prologue/epilogue insertion: `FrameSetup`/`FrameDestroy` → real sequences using CC's callee-saved set.
8. Machine code emission via `iced-x86`.
9. JIT load: mmap → mprotect PROT_EXEC → function pointer.

See `docs/ARCHITECTURE.md` for the full design, `docs/ROADMAP.md` for the phased plan.

## File layout

- `src/codegen/tir/` — generic target IR: `Func`, `Block`, `Inst` trait, `PseudoInstruction`.
- `src/codegen/isa/x64/` — x86_64 ISA: instructions, registers, MC emission.
- `src/codegen/analysis/` — CFG, liveness, dominance.
- `src/codegen/regalloc/` — register allocator.
- `src/support/` — infra: slotmap, bitset.

## Commands

- `cargo build` — compile.
- `cargo test` — run all tests.
- `cargo clippy --all-targets -- -D warnings` — lint.

## Specialized agents

This project ships subagents in `.claude/agents/`. Delegate to them for concentrated domain work:

- **regalloc-expert** — register allocation, liveness, coalescing, spill.
- **isa-x64-engineer** — x86_64 instruction encoding, iced-x86, SysV ABI, SIMD.
- **pass-engineer** — analysis/transform passes, pipeline ordering, invariants.
- **test-validator** — runs `cargo test` after every Rust edit, reports failures as blocking, writes regression tests for uncovered features.

Spawn them proactively when a task centers on their domain — they carry concentrated context and conventions.

## Automated validation

A Stop hook (`.claude/settings.json`) runs the **test-validator** subagent once at the end of every Claude turn. Behavior:

- Gates on `git status`: if no `.rs` files changed this turn, the validator exits silently.
- Runs `cargo test --no-fail-fast`. If anything fails, the hook blocks with the failure excerpt — fix tests before continuing.
- On green, diffs each changed file to identify new or materially modified **features** (branching logic, new `impl`s carrying behavior, new instruction variants, new passes). Writes the minimum regression test per uncovered feature. Does **not** test constants, trivial accessors, derives, or renames.
- Only edits `.rs` files already touched in the same turn — never spreads work into unrelated files.

The granularity is **per feature, not per edit**: multiple edits within a turn are validated once at the end.

Disable temporarily via `/hooks` if the loop is noisy during a refactor.

## Known bugs / technical debt

Start here when looking for a short first task:
- `src/support/bitset.rs:85` — `has` bounds check uses magic `32` instead of `Self::bits_in_bucket()` (64). Two tests fail.
- `src/codegen/analysis/dom_tree.rs:66` — `compute_idom` panics on unreachable blocks via `unwrap()`. Four tests fail on loop CFGs.
- `src/codegen/regalloc/fast.rs:84` — debug `println!` in hot path.
- `src/codegen/isa/x64/inst.rs` — six `todo!()` in `get_defs` / `Display` (CondJmp, Mov64mi64, CMP64rr, Mov64rm, Mov64mr, Mov64ri64).
- `src/codegen/isa/x64/mc/emit_mc.rs` — all variants except `Mov64rr`/`Jmp` emit as `todo!()`.
- `src/bin/main.rs` — regalloc invocation commented out; replace with end-to-end demo once JIT runtime lands.

## Style

- No emojis in code or docs.
- No multi-paragraph docstrings. One-line comments only when the WHY is non-obvious.
- Prefer editing existing files over creating new ones.
- No `unwrap()` on fallible operations outside tests — use `expect("msg")` or proper error propagation.
