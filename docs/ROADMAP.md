# Roadmap

Phased plan. Each phase has a concrete deliverable and exit criteria. Earlier phases unblock later ones; skipping is not recommended.

## Phase 0 — Fix what's broken

**Goal:** all tests pass, no `todo!()` in code paths reachable from tests.

- [ ] `support/bitset.rs:85` — replace magic `* 32` with `* Self::bits_in_bucket()`. Verify 2 failing bitset tests pass.
- [ ] `analysis/dom_tree.rs::compute_idom` — handle unreachable blocks without panicking. Verify 4 failing domtree tests pass.
- [ ] `regalloc/fast.rs:84` — remove debug `println!`.
- [ ] `isa/x64/mc/emit_mc.rs` — document each `todo!()` with the Phase that will close it. Don't implement yet; they belong to Phase 2.
- [ ] Add CI check: `cargo clippy --all-targets -- -D warnings` enforced on PRs.

**Exit:** `cargo test` green. `cargo clippy --all-targets -- -D warnings` green.

## Phase 1 — Typed IR

**Goal:** every `Reg` has a `Type`. Instruction variants are type-aware.

- [ ] Introduce `Type` enum: `I8`, `I16`, `I32`, `I64`, `F32`, `F64`, `Ptr`, `V128(ScalarTy)`, `V256(ScalarTy)`, `V512(ScalarTy)`.
- [ ] Extend `Reg` / `Func::new_vreg` to take `Type`.
- [ ] Replace `X64Inst::Mov64rr` et al. with a typed form. Options:
  - `Mov { ty: Type, dst: Reg, src: Reg }` — one variant, carries type.
  - Or separate variants per width if the encoder path differs significantly.
  - Decide during Phase 2; for now just pick one and migrate.
- [ ] Debug-assert type compatibility in `Instruction::new` / builder helpers.

**Exit:** all existing functionality preserved; every vreg typed; type mismatches panic in debug.

## Phase 2 — Minimal x86_64 instruction set

**Goal:** enough x64 to compile non-trivial scalar integer functions.

- [ ] Integer arithmetic: `ADD`, `SUB`, `IMUL`, `IDIV`, `AND`, `OR`, `XOR`, `SHL`, `SHR`, `SAR`, `NOT`, `NEG`. Forms: `rr`, `rm`, `mr`, `ri`.
- [ ] Memory: `MOV` (all sizes × all forms), `LEA`.
- [ ] Control flow: `JMP`, `CondJmp` (all `Cond`), `CALL`, `RET`.
- [ ] Compare/test: `CMP`, `TEST`.
- [ ] Each instruction: full `get_uses`, `get_defs`, `Display`, MC emission via iced-x86.
- [ ] No `todo!()` in these code paths.

**Exit:** can construct and MC-emit an IR for `fn(a: i64, b: i64) -> i64 { if a < b { a + b } else { a - b } }` (no regalloc yet).

## Phase 3 — Pseudo-instruction set

**Goal:** all target-neutral pseudos from `ARCHITECTURE.md §3` are implemented.

- [ ] Add: `Copy`, `Phi`, `StackAlloc`, `CallPseudo`, `FrameSetup`, `FrameDestroy`, `ImplicitDef`, `Kill`, `RegDef`.
- [ ] `get_uses`/`get_defs` for each.
- [ ] SSA destruction pass: `Phi` → parallel `Copy`s in predecessors. Handles swap/cycle case (via temporary vreg).
- [ ] Regalloc: consume `Copy` as coalescing candidate, `RegDef` as pre-binding constraint, `Kill` as explicit range end.
- [ ] Pseudo cleanup pass: erase `Kill`/`ImplicitDef`; surviving `Copy` → target MOV or elided.

**Exit:** an SSA-form input with phis survives the pipeline unchanged at the end.

## Phase 4 — Calling convention abstraction

**Goal:** frontends can emit abstract `Arg`/`CallPseudo`; backend lowers correctly for SysV AMD64.

- [ ] `CallingConvention` trait + `ArgLoc`/`RetLoc` types.
- [ ] `SysV` impl: arg/return locations, callee-saved set, stack alignment, shadow space.
- [ ] ABI lowering pass: `Arg` → physical reg move or stack load. `CallPseudo` → argument placement + target `CALL` + return value extraction.
- [ ] Prologue/epilogue insertion: compute frame size (from `StackAlloc` + spill slots) + used callee-saved set; emit `PUSH` + `SUB RSP` + `MOV` sequence for prologue, inverse for epilogue.

**Exit:** can compile and execute (via inline manual call) a function taking 6+ int args and returning a value.

## Phase 5 — Regalloc upgrades

**Goal:** produce code quality competitive with `gcc -O0` / `clang -O0`.

- [ ] Move coalescing via `Copy` hints. Measure copy elimination rate on a corpus.
- [ ] Reg hints propagated from ABI (`Arg` should prefer arg-reg, return preceding value should prefer return-reg).
- [ ] Spill weight: `uses-count / live-range-length`. Spill lowest-weight interval when pressure forces eviction.
- [ ] Rematerialization: values defined by `Mov ri` or loads from read-only memory can be re-emitted at use sites instead of spilled.

**Exit:** on a hand-crafted benchmark of 10 small functions, coalesced copies ≥ 80%, no spurious spills on pressure ≤ available regs.

## Phase 6 — JIT runtime + end-to-end tests

**Goal:** demonstrably working JIT. This is where the project stops being a toy.

- [ ] `jit::Module` with `load` / `entry<F>` / `Drop`.
- [ ] `mmap` + `mprotect` via `libc` crate (no unsafe surprises).
- [ ] End-to-end differential test harness: define a set of reference Rust functions (arithmetic, branches, loops, calls), build equivalent TIR, JIT-compile, call, diff results on randomized inputs. Run under `miri` to catch UB in the runtime.
- [ ] First e2e tests: `add`, `sub`, `mul`, `if/else`, `while`-loop, function that calls another JIT function.

**Exit:** the e2e suite is the gatekeeper — no PR lands that breaks it.

## Phase 7 — SIMD

**Goal:** SSE → AVX → AVX-512 vector operations.

- [ ] SSE baseline: `MOVAPS/MOVAPD`, `PADDB/W/D/Q`, `PMULLD`, `PAND/POR/PXOR`, `PCMPEQ*`, `PSHUFD`, `PSLLDQ/PSRLDQ`.
- [ ] F-vector: `ADDPS/ADDPD`, `MULPS/MULPD`, `SUBPS/SUBPD`, `DIVPS/DIVPD`, `SQRTPS/PD`, compare+select.
- [ ] AVX expansion via VEX: 256-bit versions of all above, 3-operand forms.
- [ ] AVX-512 via EVEX: 512-bit, mask registers (`K0..K7`), per-element masking, embedded broadcast.
- [ ] E2E tests: vector add, dot product, filter kernel.

**Exit:** can compile a column-scan filter kernel (vectorized comparison + masked store) and match native SSE/AVX/AVX-512 output bit-for-bit.

## Phase 8 — LLVM IR frontend

**Goal:** first non-builder frontend. Enables real DB integration experiments.

- [ ] Choose ingestion: `llvm-ir` crate (pure Rust, parses bitcode) or `inkwell` (LLVM C++ bindings). Prefer pure Rust if it covers what's needed.
- [ ] Translation: LLVM instructions → TIR with ISel-shaped logic baked into the converter.
- [ ] Handle typed values, SSA preservation (LLVM IR is already SSA), phi conversion (TIR already has `Phi`).
- [ ] Calls to external C functions via `CallPseudo` + runtime symbol resolution.

**Exit:** can compile a non-trivial LLVM IR module (say, a small ClickHouse-style column kernel) and produce matching results.

## Phase 9 — AArch64

**Goal:** second target. Validates the ISA abstraction.

- [ ] `src/codegen/isa/aarch64/` mirroring x64 layout.
- [ ] AArch64 AAPCS calling convention implementation.
- [ ] NEON vector support (128-bit only; no SVE yet).
- [ ] Same pass pipeline, same regalloc, same JIT runtime.

**Exit:** the full e2e suite from Phase 6 passes on AArch64.

## Open questions deferred

- Debug info format (DWARF) — not in scope until a consumer asks.
- Profile-guided spill heuristics — speculative, defer.
- Hot-path `ret`-patching for re-JIT — speculative, defer.
- Cross-function inlining — explicitly out of scope (frontend's job).
