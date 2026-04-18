---
description: Scaffold a new x86_64 instruction across all required locations (enum variant, Inst trait, Display, MC emission, tests).
---

Add a new x86_64 instruction to `src/codegen/isa/x64/`. The user will specify the mnemonic and form (e.g. "ADD64rr", "SHL64ri", "VADDPS YYY").

Follow the checklist in `.claude/agents/isa-x64-engineer.md` exactly. Delegate to the `isa-x64-engineer` subagent for the actual implementation — it carries the concentrated context.

Do not merge with any `todo!()` in the new variant's code paths. Do not skip tests.

Argument: $ARGUMENTS
