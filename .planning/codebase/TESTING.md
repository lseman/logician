---
title: "Testing"
analysis_date: "2026-08-08"
---

# Testing

**Analysis Date:** 2026-08-08

*Logician TUI — test framework, structure, mocking, and coverage practices.*

---

## Test Framework

| Aspect | Detail |
|---|---|
| **Runner** | Node.js native test runner via `tsx --test` |
| **Assertion** | `node:assert/strict` |
| **Temp files** | `node:os.tmpdir()` + `node:fs.mkdtempSync()` |
| **Benchmarks** | `node:perf_hooks.performance` |
| **Test command** | `bun test <glob>` or `tsx --test <glob>` |

### Test Discovery

Tests are discovered by the `--test` flag with glob patterns:

```bash
# Root package.json test script:
bun test tui/packages/agent-core/src/__tests__/*.test.ts \
  tui/packages/agent-capabilities/src/__tests__/*.test.ts \
  tui/packages/coding-agent/src/__tests__/*.test.ts \
  tui/packages/tui/src/__tests__/*.test.ts \
  tui/packages/tui/__tests__/*.test.ts

# Per-package test script:
tsx --test $(find src -name '*.test.ts' -type f -print)
```

### Test Count

- **99 test files** across all packages
- Tests are colocated with source in `src/__tests__/` directories

---

## Test File Structure

### Conventions

| Convention | Detail |
|---|---|
| **Location** | `src/__tests__/<name>.test.ts` — colocated with source |
| **Naming** | `<feature>.test.ts` — matches the feature being tested |
| **Import** | `import assert from "node:assert/strict"` |
| **Test registration** | `void test("description", async () => { ... })` |
| **Async tests** | `async` functions with explicit `await` |
| **Temp directories** | `mkdtempSync(join(tmpdir(), "prefix-"))` |
| **Cleanup** | `rmSync(dir, { recursive: true, force: true })` in `afterEach` |

### Test File Pattern

```typescript
import assert from "node:assert/strict";
import { test, afterEach } from "node:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmpDir = mkdtempSync(join(tmpdir(), "test-prefix-"));

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

void test("feature does X", async () => {
  // Arrange
  // Act
  // Assert
  assert.strictEqual(actual, expected);
});

void test("feature handles error case", async () => {
  // ...
});
```

---

## Test Categories

### Unit Tests

Tests individual functions, classes, and modules in isolation.

| Package | Test Files | Focus |
|---|---|---|
| `agent-core` | ~15 files | Tool execution, hooks, compaction, session, loop detection |
| `agent-capabilities` | ~5 files | Skills, reasoning, RAG tools |
| `coding-agent` | ~35 files | Config, skills, diff-utils, MCP, trust, tools, sandbox, web-search |
| `tui` | ~20 files | Benchmark, terminal, config, headless-exec, composite rendering |

### Integration Tests

Tests cross-module interactions.

| Test | File | Purpose |
|---|---|---|
| MCP integration | `coding-agent/src/__tests__/mcp-integrations.test.ts` | MCP server config loading |
| Session persistence | `coding-agent/src/__tests__/session_store.test.ts` | JSONL session read/write |
| Config loading | `coding-agent/src/__tests__/config.test.ts` | Multi-source config resolution |
| Trust system | `coding-agent/src/__tests__/trust.test.ts` | Trust decision persistence |

### Performance Benchmarks

| Benchmark | File | Measures |
|---|---|---|
| Main benchmark | `tui/src/__tests__/benchmark.ts` | Cold/hot render, streaming, typing, scroll, large screen, frame timing, layout |
| Keystroke benchmark | `tui/src/__tests__/benchmark-keystroke.ts` | Keystroke latency |
| Latency benchmark | `tui/src/__tests__/benchmark-latency.ts` | End-to-end latency |

### Test Harnesses

| Harness | File | Purpose |
|---|---|---|
| FakeBackend | `agent-core/src/__tests__/fake-backend.ts` | Scriptable LLM backend for tests |
| Sample Pi extension | `agent-core/src/__tests__/sample-pi-extension.ts` | Pi extension demo for adapter tests |
| PTY harness | `tui/src/testing/pty-harness.ts` | PTY-based terminal testing |
| Terminal screen | `tui/src/testing/terminal-screen.ts` | Terminal screen mock |
| PTY app home | `tui/src/testing/pty-app-home.ts` | PTY app for home directory tests |

---

## Mocking Strategy

### FakeBackend Pattern

For LLM backend testing, a `FakeBackend` class implements `LLMBackend` with a responder queue:

```typescript
// fake-backend.ts
export class FakeBackend implements LLMBackend {
  private responders: Responder[] = [];

  addResponder(responder: Responder): void {
    this.responders.push(responder);
  }

  async generate(messages, options): Promise<LLMResponse> {
    const responder = this.responders.shift();
    return responder ? responder(messages, options) : textResponse("");
  }
}
```

Responders can return sync or async responses, enabling precise control over LLM behavior in tests.

### PTY Harness Pattern

For terminal UI testing, a PTY harness provides a real PTY with controlled input/output:

```typescript
// pty-harness.ts
export class PtyHarness {
  // Spawns a PTY process
  // Controls input/output
  // Provides terminal state inspection
}
```

### Config Mocking

Tests create temporary workspace directories with controlled config files:

```typescript
function mkWorkspace(): string {
  const dir = mkdtempSync(join(tmpdir(), "workspace-"));
  writeFileSync(join(dir, ".logician.json"), JSON.stringify(config));
  return dir;
}
```

---

## Test Coverage by Module

### agent-core

| Module | Tests | Coverage |
|---|---|---|
| Bash tool | `bash-tool.test.ts` | Command execution, timeout, output limits |
| Edit tool | `edit-tool.test.ts` | File editing, atomic writes |
| Diff utils | `diff-utils.test.ts` | Disjoint edits, context lines |
| Sandbox | `sandbox.test.ts` | Bubblewrap isolation |
| Sandbox runtime | `sandbox-runtime.test.ts` | Runtime configuration |
| Loop manager | `loop-manager.test.ts` | Loop detection, guards |
| MCP client | `mcp-client.test.ts` | Stdio/HTTP clients |
| MCP manager | `mcp-manager.test.ts` | Server loading, tool creation |
| Config | `config.test.ts` | Config validation, loading |
| Runtime config | `runtime-config.test.ts` | Runtime configuration resolution |
| Trust | `trust.test.ts` | Trust decisions, store |
| Skills | `skills.test.ts` | Skill loading, activation |
| Session title | `session-title.test.ts` | Title generation |
| Slash commands | `slash-commands.test.ts` | Command parsing, dispatch |
| Search tool | `search-tool.test.ts` | File search |
| Event mapping | `event-mapping.test.ts` | Agent event → bridge event |
| Transcript updates | `transcript-message-update.test.ts` | Message rendering |
| Subagent activity | `transcript-subagent-activity.test.ts` | Subagent output rendering |
| Subagent direct mode | `transcript-subagent-direct-mode.test.ts` | Direct mode rendering |
| Tool ordering | `transcript-tool-ordering.test.ts` | Tool call display order |
| EOS file | `eoh-file.test.ts` | End-of-head file handling |
| Memory tools | `memory-tools.test.ts` | Memory hook integration |
| Path policy | `path-policy.test.ts` | Path allow/deny rules |
| System prompt MCP | `system-prompt-mcp.test.ts` | MCP in system prompt |
| Web search | `web-search.test.ts` | Web search tool |
| Bridge message delivery | `bridge-message-delivery.test.ts` | Message delivery to bridge |
| Doctor | `doctor.test.ts` | Health diagnostics |
| Config (coding-agent) | `config.test.ts` | Config loading/saving |

### tui

| Module | Tests | Coverage |
|---|---|---|
| Benchmark | `benchmark.ts`, `benchmark-keystroke.ts`, `benchmark-latency.ts` | Performance metrics |
| Terminal sanitize | `terminal-sanitize.test.ts` | Output sanitization |
| Config (tui) | `config.test.ts` | TUI config |
| Headless exec | `headless-exec.test.ts` | Non-interactive execution |
| Composite TUI | `composite-tui-line-fast-path.test.ts` | Rendering performance |

---

## Test Patterns

### Arrange-Act-Assert

All tests follow the AAA pattern:

```typescript
void test("feature name", async () => {
  // Arrange — set up test data, mocks, temp files
  const backend = new FakeBackend();
  backend.addResponder(textResponse("test output"));

  // Act — call the function under test
  const result = await someFunction(args);

  // Assert — verify expected behavior
  assert.strictEqual(result, expected);
  assert.ok(result.includes("expected substring"));
});
```

### Temp Directory Isolation

Each test that needs filesystem access creates its own temp directory:

```typescript
const tmpDir = mkdtempSync(join(tmpdir(), "test-prefix-"));
// ... use tmpDir for test files ...
rmSync(tmpDir, { recursive: true, force: true });
```

### Async Test Handling

Tests use `async` functions with explicit `await`:

```typescript
void test("async operation", async () => {
  const result = await someAsyncFunction();
  assert.strictEqual(result, expected);
});
```

The `no-floating-promises` ESLint rule is disabled for test files since the test runner manages async lifecycle.

### Benchmark Pattern

Benchmarks use `performance.now()` for timing:

```typescript
const start = performance.now();
// ... operation to measure ...
const elapsed = performance.now() - start;
```

---

## Test Configuration

### Per-Package Test Scripts

| Package | Test Command |
|---|---|
| `agent-core` | `tsx --test $(find src -name '*.test.ts' -type f -print)` |
| `agent-capabilities` | `tsx --test $(find src -name '*.test.ts' -type f -print)` |
| `coding-agent` | `tsx --test $(find src -name '*.test.ts' -type f -print)` |
| `tui` | `tsx --test $(find src -name '*.test.ts' -type f -print)` |
| `rag` | (no test script — typecheck only) |
| `memory` | `tsx --test $(find src -name '*.test.ts' -type f)` |
| `autoresearch` | `node --test tests/*.test.mjs` |

### CI Pipeline

```bash
bun run typecheck && bun run lint && bun run format:check && bun run test
```

---

## Testing Gaps

| Area | Status | Notes |
|---|---|---|
| RAG pipeline | No tests | Python subprocess integration |
| Memory hooks | No dedicated tests | Relies on integration with agent-core |
| Autoresearch | Some tests | `tests/*.test.mjs` (Node.js format) |
| Extension system | Limited | `sample-pi-extension.ts` is demo, not test |
| Hook system | Limited | Loop detector tested, other hooks less covered |
| MCP HTTP client | Limited | Stdio client more tested |

---

*Testing analysis: 2026-08-08*
