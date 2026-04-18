# Lancy Architecture

## 1. IR model

**One IR, target-level, formally non-SSA.** Modeled after LLVM MIR.

- `Func<I>` — a function parameterized by target ISA `I: Inst`. Owns basic blocks and virtual registers.
- `Block` — slotmap key into `BlockData<I>`. A `BlockData` is a `Vec<Instruction<I>>` with a terminator at the end.
- `Reg` — `u32` virtual register id. Every `Reg` carries a `Type` (see §2). Pre-assigned physical registers are handled via the `RegDef` pseudo plus regalloc's binding config.
- `Instruction<I>` — enum of either target-specific `I` (e.g. `X64Inst`) or `PseudoInstruction` (§3).
- `Inst` trait — the interface all target instruction sets implement: `is_branch`, `is_ret`, `is_term`, `get_uses`, `get_defs`, `get_branch_targets`. Passes are generic over `I: Inst`.

**SSA as convention.** Frontend emits SSA. Passes up to and including SSA destruction preserve SSA. Regalloc and everything after it do not. This is documented contract; no flags, no checks. Each pass's doc comment states what it requires and what it preserves.

## 2. Type system

Closed set:

| Type | Description |
|---|---|
| `I8`, `I16`, `I32`, `I64` | Signed/unsigned integers (sign is a per-instruction concern, not the type). |
| `F32`, `F64` | IEEE-754 floats. |
| `Ptr` | 64-bit pointer. |
| `V128<T>`, `V256<T>`, `V512<T>` | SIMD vectors. `T` is any scalar type above. |

Every `Reg` carries a `Type`. Every instruction operand position implies a required type. Debug builds assert type compatibility on construction. Release builds trust the frontend.

Vector width is target-determined. The frontend is responsible for committing to a target vector width; `V512<T>` is illegal on targets without AVX-512.

## 3. PseudoInstruction set

Target-neutral ops. Closed list — adding a new one requires a design change, not just a commit.

| Pseudo | Purpose | Lowered by | Timing |
|---|---|---|---|
| `Arg(dst: Reg, idx: u32)` | Incoming argument `idx`. | ABI lowering pass. | After CC binding, before regalloc. |
| `Copy(dst: Reg, src: Reg)` | Typed value copy. Primary coalescing candidate. | Regalloc (erased on coalesce) or pseudo cleanup (→ target MOV). | During/after regalloc. |
| `Phi(dst: Reg, incoming: [(Block, Reg)])` | SSA merge. | SSA destruction pass (→ parallel `Copy`s in predecessors). | Before regalloc. |
| `StackAlloc(slot: StackSlot, size: u32, align: u32)` | Reserve a stack slot. | Prologue insertion. | After regalloc (frame is known). |
| `CallPseudo(callee, args: [Reg], rets: [Reg], cc: CallingConv)` | Abstract call. | ABI lowering pass (→ moves + target CALL). | Before regalloc. |
| `FrameSetup` / `FrameDestroy` | Markers for prologue/epilogue placement. | Prologue insertion pass (→ PUSH/SUB/MOV or POP/ADD/RET sequence). | Late. |
| `ImplicitDef(dst: Reg)` | Undef definition. Regalloc sees it as a def with no cost. | Pseudo cleanup (erased). | Pre-emit. |
| `Kill(src: Reg)` | Explicit end-of-live-range marker. | Pseudo cleanup (erased). | Always. |
| `RegDef(vreg: Reg, preg: PReg)` | Pre-bind vreg to a specific physical register (e.g. for calling conventions, intrinsic constraints). | Regalloc honors as constraint. | Erased after regalloc. |

Rationale: everything that every target needs — argument passing, calls, frames, spilling — lives here, so targets only define their real machine instructions.

## 4. Pass pipeline

```
Frontend (SSA-form TIR)
    │
    ▼
CFG ─┐
Liveness ─┤── analyses, all generic over Inst
DomTree ─┘
    │
    ▼
SSA destruction          (Phi → parallel Copy)
    │
    ▼
ABI lowering             (Arg, CallPseudo → physical moves + target insts)
    │
    ▼
Legalization             (ensure every inst is encodable on target)
    │
    ▼
Register allocation      (linear scan, Copy coalescing, RegDef constraints)
    │
    ▼
Pseudo cleanup           (Kill/ImplicitDef erased; surviving Copy → MOV or elided)
    │
    ▼
Prologue/epilogue insertion   (FrameSetup/FrameDestroy → real sequences, using CC's callee-saved)
    │
    ▼
MC emission              (iced-x86 for x64)
    │
    ▼
JIT load                 (mmap + mprotect + function ptr)
```

Each pass documents: **requires** (invariants it assumes), **preserves** (invariants it maintains), **invalidates** (invariants it may break).

## 5. Calling convention abstraction

A target-neutral trait; targets (x64 SysV, x64 Win64, AArch64 AAPCS, custom fastcall) implement it. ABI lowering and prologue passes consume it; regalloc consumes caller/callee-saved information and pre-binding hints.

Sketch:

```rust
pub trait CallingConvention {
    fn arg_locations(&self, sig: &FnSig) -> Vec<ArgLoc>;
    fn return_locations(&self, sig: &FnSig) -> Vec<RetLoc>;
    fn callee_saved(&self) -> &[PReg];
    fn caller_saved(&self) -> &[PReg];
    fn stack_alignment(&self) -> u32;
    fn shadow_space(&self) -> u32;   // Win64 = 32, SysV = 0
}

pub enum ArgLoc {
    Reg(PReg),
    Stack { offset: i32 },
    SplitRegStack { regs: SmallVec<[PReg; 2]>, stack_offset: Option<i32> },
}
```

SysV AMD64: int args `RDI/RSI/RDX/RCX/R8/R9`, float args `XMM0..XMM7`, return `RAX` (+ `RDX` for 128-bit), callee-saved `RBX/RBP/R12..R15`, 16-byte stack alignment at call, no shadow space.

Custom fastcall (for hot DB kernels): everything in registers, no callee-saved, minimal frame. Defined per consumer.

## 6. Register allocation

Linear scan over live intervals. SSA property of the early pipeline makes liveness cheap (one pass instead of iterative dataflow).

Extensions over the current naive implementation:
- **Move coalescing** via `Copy` hints. When src and dst of a `Copy` have non-overlapping live ranges, merge them and erase the copy.
- **Reg hints** from `RegDef` and ABI pre-binding.
- **Spill weights** based on use density (uses-per-unit-length of live range).
- **Rematerialization** for cheap-to-recompute values (immediates, simple loads from read-only memory) — spill regenerates instead of reloading.

The algorithm produces:
- Coloring: `Reg → AllocatedSlot` where `AllocatedSlot` is either `Reg(PReg)` or `Stack(StackSlot)`.
- Frame layout: `StackSlot → offset`.
- Total frame size.

## 7. Machine code emission

Delegate to `iced-x86` for x64. The emitter walks blocks in layout order, maps each `X64Inst` to `iced-x86` `CodeAssembler` calls, resolves labels for branches, and emits a flat `Vec<u8>`.

Produces:
- Code bytes.
- Relocations for external symbols (for future CallPseudo → external C function).
- Debug map: instruction index → byte range → (vreg → preg) for disassembly correlation.

## 8. JIT runtime

```rust
pub struct Module {
    code: *mut u8,
    size: usize,
}

impl Module {
    pub fn load(bytes: &[u8]) -> io::Result<Self> { ... }
    pub fn entry<F>(&self, name: &str) -> Option<F> { ... }  // typed function pointer
}

impl Drop for Module {
    fn drop(&mut self) { unsafe { munmap(self.code, self.size); } }
}
```

`load`: `mmap(NULL, size, RW, PRIVATE | ANON)` → `memcpy` → `mprotect(RX)` → store. `entry` does a typed cast to `fn(...) -> ...`.

## 9. Testing strategy

Three layers:

1. **Unit tests per pass.** Existing coverage for CFG, liveness, domtree, regalloc, bitset, slotmap, emission. Keep growing.
2. **IR-builder integration tests.** Construct known IR, run selected passes, assert IR shape.
3. **End-to-end differential tests.** Define a reference Rust function; build equivalent TIR; compile through the full pipeline; JIT-load; call; compare results over randomized inputs. This is the non-negotiable correctness bar — without it we're shipping a compiler with no proof it computes anything.

## 10. Non-goals and deliberate cuts

- No SelectionDAG / tablegen-style pattern matcher. Frontends emit target IR directly.
- No high-level IR layer above TIR. Frontends work in their own representation (LLVM IR, AST, whatever) and translate in one step.
- No alias analysis. Frontend is expected to have resolved this upstream.
- No autovectorization. Frontend emits vectors directly using the typed vector operations.
- No register coalescing across functions. Each function is compiled in isolation.
- No debug info format (DWARF/PDB) — just in-memory source maps for disassembly.
