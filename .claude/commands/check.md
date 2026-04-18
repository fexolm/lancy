---
description: Run full local check — build, clippy, tests. Fail loudly on any red.
---

Run, in order, surfacing the first failure:

1. `cargo build --all-targets`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo test`

If any step fails, report the failure verbatim and stop. Do not attempt to fix unless the user asks.
