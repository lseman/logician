# Logician Improvement Plan — 2026-09-19

**Goal:** make logician better, faster, more SOTA. Evidence base: full read of `log-core`/`log-tui`/`log-rag`/`crates` + ecosystem integrations, three-way diff of the vendored `pi-*` crates against oh-my-pi `origin/main` (fork point `4999b98`, 2026-09-13; 319 upstream PRs since), and a survey of `repos/{headroom, deepseek-harness, gsd-core, claude-mem, agentmemory}`.

Full supporting reports: `agent://OmpDeltaScout` (upstream delta), `agent://LogicianCoreScout` (internal audit).

## 0. What is already strong (don't touch)

- Native BPE counting (`pi-natives/tokens.rs`): UTF-16 direct, rayon batches, 9 encodings embedded.
- Parallel tool batches with model-ordered results; `ToolResultCache` (LRU + mtime invalidation).
- Three-mode compaction (LLM / remote / snapcompact local PNG frames) with turn-boundary cuts.
- Layered guards: `LoopDetector`, `TextLoopDetector`, `OutputGuard`, `RunBudgetController`, verified-stop policy.
- TUI: 60fps row-diff renderer with per-turn caches — measured, not a bottleneck.
- RAG: local ONNX MiniLM, hybrid dense+BM25 + reranker, on-demand tools only.
- Memoriam/legroom: persistent Python JSONL workers (right pattern; the flaw is *when* they're invoked, not the transport).

## 1. Done in this session — upstream crate sync (verified)

| Crate | Change | Status |
|---|---|---|
| `pi-edit` | Full sync: merged upstream `sloppy/apply.rs` (header payload syntax, seen-line guard, stall/panic fixes) into the split `pattern/locate/recovery` layout + 7 other files + 2 new test fixtures | 199 tests pass, workspace green |
| `pi-walker` | Ported ranked-collection fast path (`RankedCollectVisitor`, O(n log k) binary-heap instead of collect-all-then-sort for ranked+limited walks) + `examples/ranked-collection.rs` | Example verified against `find -printf` ordering |
| `pi-natives` | New N-API `hashline_is_read_truncation_notice`; doc sync | compiles |
| `snapcompact` | Extracted to standalone crate (prior session) | linked into addon |
| `pi-diff`, `pi-ast` | upstream delta is Cargo-header/comment re-wrap only | skipped (cosmetic) |
| `pi-walker` `cache.rs` | upstream ScanCache redesign (dashmap, byte budget, +400 lines) conflicts with logician's `ParallelWalkControl` stop-early addition | **deferred** — needs a dedicated port of the whole cache module |

## 2. P0 — fast wins (all five done, verified)

| # | Item | Result |
|---|---|---|
| 1 | Skip redundant full-history token estimation | `agent-harness.ts`: prefer provider-reported `usage.totalTokens`; estimate only as fallback. Removes 1–2 full-history serialization+BPE passes per turn. |
| 2 | Model-aware token counting | `resolveTokenEncoding(model)` family table in `messages.ts` (verified against `pi-natives/tokens.rs` docs); threaded through 5 call sites (harness context_update, session context controller, session-compactor gate, builtin-hooks threshold path, `shouldAutoCompact`). Smoke: o200k 29 vs Qwen3 30 vs DeepSeekV3 30 on identical text. |
| 3 | Default request/stream-idle timeouts | `backend.ts`: 30 s initial-response guard + 300 s stream-idle guard (per-request, configurable via `createLLMBackend` options, `0` disables). Both surface as retryable `BackendError`. Bun quirk handled: `AbortSignal.timeout` rejects with `TimeoutError`, not `AbortError` — attribution by signal, not error name. `classifyNetworkError` now treats "timed out" as transient. 3 regression tests (`provider-timeouts.test.ts`) with real local servers. |
| 4 | Resurrect or delete `StablePrefix` | **Deleted** (287 lines): `toContext()` returned empty tools (unwirable without a redesign = P1.1-scale), zero consumers repo-wide, dead build site in `agent-harness.ts` removed. |
| 5 | Structured snapcompact summary | `log-snapcompact`: deterministic zero-LLM digest — `Goal:` (first user message), `Files:` (path args from tool calls, cap 12, free-form bash commands skipped), `Activity:` (per-tool counts), `Last before archive:` (final assistant statement) — prepended to the existing archive prose. Verified against a realistic session + empty/bare edge cases. |

**Verification:** log-core typecheck clean, 362/362 tests; log-snapcompact 31/31; timeout guards exercised against real hanging local servers. (Pre-existing red in log-runtime/log-tui typecheck is unrelated — those files have no local changes.)

## 3. P1 — medium (each ~1 week or less)

1. **Fix prompt-cache invalidation** (biggest $/latency lever on hosted endpoints): memoriam **prepends** a memory system message at position 0 every turn (busts the whole prefix cache); legroom rewrites the entire payload through Python every provider call (and every retry). Fixes: inject memory in the stable system-prompt dynamic tail refreshed only on memory-revision change; prefix-memoize legroom (history is append-only — compress only the new tail, splice); emit `cache_control` breakpoints where the endpoint supports them. Upstream is actively investing here (#12431, #12332) — worth matching their approach.
2. **Async batched session journal** — `session-store.ts:729-764` does ~3 sync fsyncs + 3 open/close pairs per persisted message, on the event loop; a 20-tool-call turn ≈ 120 fsyncs blocking streaming/TUI/hooks. One open, buffered appends, coalesced fsync at turn end/timer, async meta, drop per-append dir fsync.
3. **Persistent graphician worker** — `graphician.ts` spawns a fresh Python process per query (re-imports stack, opens the 571 MB DB, 1–10 s) and can stall a tool slot 5–60 s on rebuild-on-query. Reuse the `JsonlWorker` pattern with a background debounced incremental rebuild decoupled from query latency.
4. **Predictive compaction** — while a turn streams, project token growth from `usage` and pre-compact the tail (or pre-render snapcompact frames) in the background, so `context_full` mid-loop doesn't stall on a synchronous LLM summary.
5. **Wire memoriam auto-observation** (batched async `afterToolCall`/`message_end` → `observe`) or drop the doc claim — today memories are only written when the model calls `retain`; recall quality degrades silently. Plus a shared read-only tool cache across the subagent spawn tree (subagents re-execute identical read-only work today).

## 4. P2 — feature ports from oh-my-pi (ranked value/effort)

| # | Port | Source | Effort | Value |
|---|---|---|---|---|
| 1 | **Bash output minimizer** | `crates/pi-shell/src/minimizer/` (~150 KB Rust: detect/plan/pipeline) | M | **high** — post-process logician's bash output before it hits the model; direct context savings on every shell call. Skip the embedded brush shell itself. |
| 2 | **Secrets obfuscation** | `coding-agent/src/secrets/` (obfuscator 60 KB + placeholder-scan) | S–M | med — redaction logician has no analog for; pure-TS, self-contained |
| 3 | **In-process VCS** | `crates/pi-vcs` (gitoxide + jj-lib) + NAPI | M | med — fast status/diff/numstat without spawning git (TUI git-status, compaction); bonus Jujutsu |
| 4 | **Model catalog + compat rules** | `packages/catalog` (models.json + rules.json + resolver) | M | med — logician's provider backend is 24.5 KB; port resolver + rules format, not the whole discovery zoo |
| 5 | **CoW task isolation** | `crates/pi-iso` + `isolation-runner.ts` | M | med — complements sandbox profiles for subagent workspaces |
| 6 | **Streaming / provider-specific compaction** | `packages/agent/src/compaction/` (v2-streaming, anthropic/openai, shake, pruning, branch-summarization) | M | med — logician has the engine; the delta is streaming + per-provider strategies |
| 7 | **Subagent bounding** (incremental-yield liveness guard) | upstream #12351 | S | med — small, directly applicable to the delegation hub |
| 8 | Plan-mode file machinery (autosave/handoff/approved-plan) | `coding-agent/src/plan-mode/` | S | low–med — logician has the permission mode; add the file workflow |
| 9 | Persisted named-agent registry | `coding-agent/src/registry/` | S–M | low–med — reusable agent definitions over ad-hoc `spawn_agent` |
| 10 | internal:// issue-PR + history protocols | `coding-agent/src/internal-urls/` | M | low–med — structured GitHub/cross-session tool I/O |
| — | Provider wire breadth, turn-recovery, advisor, DAP, SSH, collab, voice | various | L | low — only on demand; not core to logician's positioning |

**Not ports** (logician already has equivalents): hashline/AST edit, LSP, browser automation, memory (memoriam), RAG, skills/plugins/MCP/hooks, subagents, permission/sandbox modes, loop guards, JSONL sessions with branch/rewind, autoresearch, EOH, doctor, headless, web search, eval kernels, TTSR, legroom, goal tracking.

## 5. Inspiration-repo ideas

- **headroom** (context compression layer, 55.9k → 24.3k tokens, 60–95% savings, important lines preserved byte-for-byte): its killer idea is **CCR — Compress-Cache-Retrieve**: compress tool outputs losslessly, keep originals in a store, let the model retrieve the original on demand. Logician's legroom is lossy pre-provider compression; a reversible CCR mode (or a legroom upgrade to headroom's pipeline: per-content-type compressors + statistical JSON/array compression 70–90% + AST code compression) is the clearest cross-repo win.
- **deepseek-harness**: everything-is-a-plugin architecture; its benchmark harness (logician analog: `log-eval`) is worth a look for eval methodology, not for porting.
- **claude-mem / agentmemory**: persistent-memory patterns overlap with memoriam; nothing to port beyond confirming memoriam's direction.
- **gsd-core**: logician already integrates GSD for planning (`.planning/`); no action.

## 6. Upstream-sync process

Fork is 2026-09-15; upstream moves at ~319 PRs/2 weeks. Recommendation:

- Keep a `repos/oh-my-pi` checkout + a `crates/README.md` fork-point record (exists).
- Re-sync `pi-edit` on a fixed cadence (it's the active robustness surface upstream); the split-file layout makes this a 3-way merge — script the mechanical parts (the `merge_sloppy.py` approach), review the semantic hunks.
- Cherry-pick, don't chase: the TS side (16-package monorepo) is too far apart for wholesale sync; take the P2 table items deliberately.
- Track the upstream themes that matter to logician: prompt-cache-head stability, pi-edit robustness, subagent bounding, catalog.

## 7. Suggested order

1. ~~P0 items 1–5~~ — **done** (see §2).
2. P1.1 (prompt cache) + P1.2 (journal) — the two biggest remaining per-turn costs.
3. P2.1 (bash output minimizer) — the single highest-value context-economy feature port.
4. P1.3 (graphician worker) + P2.2 (secrets) in parallel.
5. pi-walker `cache.rs` ScanCache port as a standalone task when the cache module gets a dedicated pass.
