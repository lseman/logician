---
title: "Concerns"
analysis_date: "2026-08-08"
---

# Concerns

**Analysis Date:** 2026-08-08

*Logician TUI — technical debt, known issues, security considerations, performance concerns, and fragile areas.*

---

## Security Concerns

### Shell Command Execution

**Location:** `tui/packages/coding-agent/src/tools/bash.ts`, `shell.ts`

Shell commands are executed via `child_process.spawn/execFile`. While there is a denylist-based guard for destructive commands, the approach relies on pattern matching rather than sandboxing by default.

| Concern | Detail |
|---|---|
| **Denylist approach** | `bash.ts:118` — "Not a sandbox — a denylist for a short list of unambiguously destructive commands" |
| **Sandbox option** | Available via `sandbox.ts` using Bubblewrap (`bwrap`), but not the default |
| **Environment injection** | `mcp/client.ts:508` — MCP server env vars are merged from `process.env` |
| **Path traversal** | Permission system uses glob patterns (`src/tools/shared/permissions.ts`) — glob-to-regex conversion could have edge cases |

### Process Environment

**Location:** `tui/packages/tui/src/index.ts`, `mcp/client.ts`

The TUI entry point loads `~/.logician/.env` and injects all values into `process.env`. This means:
- Any key=value in the env file affects the entire process
- No validation of key names or values
- MCP servers inherit all injected env vars

### Python Subprocess

**Location:** `tui/packages/rag/src/ingestion.ts`

RAG document extraction spawns Python processes (Docling). The subprocess command and arguments are constructed from tool parameters — input validation is critical.

---

## Performance Concerns

### Large File Handling

**Location:** Multiple tool files

The codebase has explicit handling for large files, indicating awareness of performance boundaries:

| File | Concern |
|---|---|
| `tools/read-file.ts` | 2000 lines / 50KB truncation limit |
| `tools/write-file.ts` | Large files should use `write_file_append` in chunks |
| `tools/diff-utils.ts` | LCS algorithm falls back to whole-block when too expensive |
| `tools/write-file-append.ts` | Designed for streaming large files across multiple calls |

### Transcript Rendering

**Location:** `tui/packages/tui/src/rendering/transcript/`

The transcript rendering system is performance-critical. Benchmarks exist (`benchmark.ts`, `benchmark-keystroke.ts`, `benchmark-latency.ts`) measuring:
- Cold/hot render performance
- Streaming (1-2 new lines per frame)
- Large screen (500/1000/2000 lines)
- Frame timing consistency (p50/p95/p99)

**Concern:** The rendering system uses a complex Flex + ScrollView composition (`rendering/flex.ts`, `rendering/layout-node.ts`, `rendering/scroll-view.ts`) that could degrade with very long transcripts.

### Vector Search

**Location:** `tui/packages/rag/src/store/sqlite-store.ts`

SQLite-backed vector store uses `BigInt(row.rowid)` for key conversion. With very large vector stores, SQLite query performance could become a bottleneck.

### Token Estimation

**Location:** `tui/packages/agent-core/src/agent/messages.ts`

Token estimation uses heuristics (`estimateChatPayloadTokens`, `estimateTokens`). Inaccurate estimates could lead to:
- Premature compaction (wasting context)
- Context overflow (losing messages)

---

## Fragile Areas

### MCP Configuration Discovery

**Location:** `tui/packages/coding-agent/src/mcp/manager.ts`

The MCP config discovery walks up the directory tree looking for `.logician.json`. The debug flag `LOGICIAN_MCP_DEBUG=1` was added for investigation (line 18-25):

> "Temporary, opt-in tracing for the 'MCP loaded but /mcp shows nothing' investigation... Remove once the bug is found — this is not meant to stay."

This suggests an unresolved bug in MCP config loading.

### Render Debug Logging

**Location:** `tui/packages/tui/src/terminal/core.ts:94-107`

```typescript
// Opt-in, off by default: LOGICIAN_TUI_DEBUG_RENDER=1 appends one JSON line per
// ...
const RENDER_DEBUG_ENABLED = process.env.LOGICIAN_TUI_DEBUG_RENDER === "1";
```

Similar debug logging is scattered across the codebase (`MCP_DEBUG`, `RENDER_DEBUG`), suggesting ongoing investigation of performance or correctness issues.

### Session Branching

**Location:** `tui/packages/agent-core/src/agent/harness/branching.ts`

Branching (fork, navigate, summarize) is a complex feature with its own summarization engine. The codebase has dedicated types (`BranchInfo`, `BranchSummaryData`) and error types (`BranchSummaryError`), indicating this is a non-trivial area.

### Compaction

**Location:** `tui/packages/agent-core/src/compaction/`

Context compaction is critical for long sessions. The system uses:
- `DEFAULT_COMPACTION_FRACTION = 0.8` — fires at 80% of window
- `COMPACTION_COOLDOWN_TURNS = 3` — cooldown between compactions
- Token estimation heuristics

**Risk:** Incorrect compaction could lose important context or create gaps in conversation history.

### Extension System

**Location:** `tui/packages/agent-core/src/extensions/`

The extension system includes a Pi adapter (`pi-adapter.ts`) for cross-compatibility with Pi extensions. This adds complexity:
- Event type translation
- Tool call format conversion
- State mapping between systems

---

## Technical Debt

### Test Coverage Gaps

| Area | Status |
|---|---|
| RAG pipeline | No tests — Python subprocess integration is untested |
| Memory hooks | No dedicated tests — relies on agent-core integration |
| Extension system | Limited — only demo extension, no unit tests |
| Hook system | Limited — loop detector tested, other hooks less covered |
| MCP HTTP client | Limited — stdio client more tested |

### TypeScript `any` Usage

Files with notable `any`/`unknown` usage (potential type safety concerns):

| File | Concern |
|---|---|
| `autoresearch/src/hooks.ts` | Generic hook handling |
| `autoresearch/src/shortcuts.ts` | Shortcut data structures |
| `rag/src/store/sqlite-store.ts` | SQLite row data |
| `rag/src/embedder.ts` | Embedder configuration |
| `coding-agent/src/tools/read-skill.ts` | Skill metadata parsing |
| `coding-agent/src/tools/search.ts` | Search result types |

### Legacy Code Patterns

| Location | Pattern |
|---|---|
| `agent-bridge.ts` | 1000+ lines — single point of complexity |
| `builtin-hooks.ts` | Complex hook composition with many interacting hooks |
| `agent-loop-runner.ts` | Long functional loop with many guard checks |

### Date Stamping

All codebase map documents use hardcoded date `2026-08-08`. The map-codebase workflow requires date stamps to be set to the current date — this is a procedural concern, not a code concern, but worth noting.

---

## Known Issues

### MCP Debug Investigation

The `LOGICIAN_MCP_DEBUG=1` flag in `mcp/manager.ts` was added to investigate "MCP loaded but /mcp shows nothing". This suggests an unresolved issue with MCP server discovery or loading.

### Render Debug Investigation

The `LOGICIAN_TUI_DEBUG_RENDER=1` flag in `terminal/core.ts` suggests ongoing investigation of rendering performance or correctness.

### Permission System Edge Cases

The glob-to-regex conversion in `permissions.ts` could have edge cases:
- Special characters in file paths
- Unicode characters
- Very long patterns

### Subagent Concurrency

The subagent concurrency limiter (`delegation/runtime.ts`) uses a fixed max parallel agents setting. No dynamic adjustment based on system resources.

---

## Risk Assessment

| Area | Risk Level | Impact | Mitigation |
|---|---|---|---|
| Shell execution | Medium | System damage | Sandbox available, denylist in place |
| MCP config loading | Medium | Broken integrations | Debug flag added, investigation ongoing |
| Context compaction | Medium | Lost context | Proactive triggers, cooldowns |
| Large file handling | Low | Performance degradation | Chunking, truncation limits |
| Extension compatibility | Low | Broken Pi extensions | Adapter layer, type guards |
| Vector search scale | Low | Slow RAG queries | SQLite indexing, usearch optimization |

---

*Concerns analysis: 2026-08-08*
