# crates/

Rust crates vendored from [oh-my-pi](https://github.com/can1357/oh-my-pi)
(`crates/pi-ast`, `crates/pi-walker`, `crates/pi-edit`, `crates/pi-builtins`),
forked at commit `4999b98bd51d5e9cd68be16af9b3a6dec1111865` (2026-09-13).
A few extras came along because the requested crates depend on them:

- `pi-diff` — a direct dependency of `pi-edit`.
- `crates/vendor/brush-core` — a locally-patched fork of the upstream
  [brush](https://github.com/reubeno/brush) shell's `brush-core`, which
  `pi-builtins` builds against. oh-my-pi redirects `brush-core` to this
  fork workspace-wide via `[patch.crates-io]` (mirrored in the root
  `Cargo.toml` here); without it, `pi-builtins` would resolve against the
  plain crates.io release, which the code isn't necessarily written against.
- `pi-natives` — the N-API bindings crate wiring `pi-ast` and `pi-edit` up
  to TypeScript, ported from oh-my-pi's own `pi-natives` (its `ast.rs` and
  `edit.rs`, plus the small internal helpers those two depend on: cancel,
  crash_handler, glob_util, iofs, js, prof, task). oh-my-pi's much larger
  binding surface (clipboard, audio, PDF, shell, VCS, ...) is not included,
  and neither is `pi-builtins` (it has no direct N-API binding upstream —
  it's only reachable through `pi-shell`, which isn't forked here). The
  JS-facing package is `packages/log-natives`; see its README for build/use.

These are source-only vendored copies (no shared git history with upstream).
`cargo check --workspace --all-targets` passes, and `pi-natives` builds into
a working native addon via `bun run build` in `packages/log-natives` —
verified end to end by calling `astMatch` and `editDiffString` from Bun.
See `LICENSE` (MIT) for the upstream copyright notice, which must be
preserved per its terms; `crates/vendor/brush-core/LICENSE` carries the
notice for that fork specifically.

This workspace requires the nightly toolchain pinned in `rust-toolchain.toml`
(matching oh-my-pi's own pin) because `pi-edit`'s `xutf` dependency uses
`#![feature(portable_simd)]` starting at 1.4.0 — stable Rust can't build it.

## Known quirks from extraction

- `pi-ast`'s `pruned_walk_matches_unpruned_on_repo_corpus_sample` test
  scans `CARGO_MANIFEST_DIR/../..` (this repo's root) for `.ts`/`.py`/`.rs`
  files as a fuzz corpus. It guards on a `packages/` directory existing,
  which is also true here, so it will sweep logician's own sources instead
  of oh-my-pi's — harmless for the test's purpose (any large real-world
  corpus works) but worth knowing if it's ever slow or its failure is
  confusing.
