---
title: "Conventions"
analysis_date: "2026-08-08"
---

# Conventions

**Analysis Date:** 2026-08-08

*Logician TUI — code style, naming, patterns, and error handling conventions.*

---

## Code Style

### Formatter (Biome)

Root config: `biome.json`

| Setting | Value |
|---|---|
| **Indent** | Tabs |
| **Quotes** | Double (`"`) |
| **Semicolons** | Always |
| **Trailing commas** | All (JS), none (JSON) |
| **Arrow parentheses** | `asNeeded` |
| **Organize imports** | On (via `assist`) |

### Linter (Biome + ESLint)

| Tool | Scope | Config |
|---|---|---|
| **Biome** | Root + all packages | `biome.json` |
| **ESLint** | `tui/packages/tui/` only | `tui/packages/tui/eslint.config.js` |

### TypeScript Rules

| Rule | Level | Notes |
|---|---|---|
| `no-floating-promises` | **error** | All promises must be handled |
| `no-misused-promises` | **error** | |
| `no-unused-vars` | warn | `_` prefix ignores allowed |
| `no-explicit-any` | warn | Prefer `unknown` |
| `no-non-null-assertion` | warn | Avoid `!` |
| `max-params` | warn (7 max) | |
| `max-lines-per-function` | warn (300 max) | Blank lines and comments excluded |
| `eqeqeq` | **error** | Always use `===` |
| `prefer-const` | **error** | |
| `no-duplicate-imports` | **error** | |

### Test File Exceptions

- `no-floating-promises` disabled in `*.test.ts` files (test runner manages async lifecycle)

---

## Naming Conventions

| Element | Convention | Examples |
|---|---|---|
| **Files** | `snake_case.ts` | `agent-loop-runner.ts`, `loop-detector.ts` |
| **Classes** | `PascalCase` | `AgentHarness`, `McpManager`, `LogicianTUI` |
| **Functions** | `camelCase` | `runAgentLoop()`, `createMcpClient()` |
| **Interfaces** | `PascalCase` | `McpClient`, `Tool`, `AgentConfig` |
| **Types** | `PascalCase` | `BackendErrorCategory`, `PermissionMode` |
| **Constants** | `UPPER_SNAKE_CASE` | `MCP_PROTOCOL_VERSION`, `DEFAULT_COMPACTION_FRACTION` |
| **Private fields** | `camelCase` with `private` keyword | `private proc: ChildProcess` |
| **Test files** | `*.test.ts` | `skills.test.ts`, `diff-utils.test.ts` |
| **Barrel files** | `index.ts` | Re-exports from sub-modules |
| **Package names** | `@logician/<name>` | `@logician/agent-core`, `@logician/coding-agent` |

---

## Code Patterns

### Result Type Pattern

Used extensively for fallible operations instead of throwing:

```typescript
export type Result<TValue, TError> =
  | { ok: true; value: TValue }
  | { ok: false; error: TError };

export function ok<TValue, TError>(value: TValue): Result<TValue, TError>
export function err<TValue, TError>(error: TError): Result<TValue, TError>
export function getOrThrow<TValue, TError>(result: Result<TValue, TError>): TValue
```

**Used in:** `src/agent/types/types-errors.ts` — file errors, execution errors, session errors.

### Barrel Export Pattern

Every package uses `index.ts` as a barrel that re-exports from sub-modules:

```typescript
// src/index.ts
export * from "./agent/agent-loop-runner.ts";
export * from "./agent/backend.ts";
// ... etc
```

Sub-paths are exposed via `package.json` exports field:

```json
"exports": {
  ".": "./src/index.ts",
  "./agent/*": "./src/agent/*",
  "./tools/*": "./src/tools/*"
}
```

### Hook Composition Pattern

Hooks are composed in layers:

```typescript
// builtin-hooks.ts
const defaultHooks = buildBuiltinHooks(config, eventBus);
const composed = composeHooks(defaultHooks, userHooks);
```

Hook types are strongly typed via `AgentHooks` interface with context/result types per event.

### Event Bus Pattern

Typed event system for extensions:

```typescript
// extensions/event-bus.ts
export interface ExtensionEventBus {
  on(event: ExtensionEventName, handler: ExtensionEventHandler): void;
  emit(event: ExtensionEvent): void;
}
```

Events are strongly typed: `TurnStartEvent`, `ToolExecutionStartEvent`, `SessionCompactEvent`, etc.

### Tool Definition Pattern

Tools follow a consistent interface:

```typescript
export interface Tool {
  name: string;
  label?: string;
  description: string;
  promptSnippet?: string;
  promptGuidelines?: string[];
  parameters: Record<string, unknown>;
  prepareArguments?: (args: unknown) => Record<string, unknown>;
  executionMode?: ToolExecutionMode;
  cacheable?: boolean;
  timeoutMs?: number;
  resolveTimeoutMs?: (args: Record<string, unknown>) => number | undefined;
  hookAliases?: string[];
  readOnly?: boolean;
  execute: (args: Record<string, unknown>, ctx: ToolContext) => Promise<string | ToolResult>;
}
```

### MCP Client Pattern

Both stdio and HTTP clients implement the same interface:

```typescript
export interface McpClient {
  name: string;
  initialize(): Promise<void>;
  listTools(): Promise<McpToolDefinition[]>;
  callTool(name: string, args: Record<string, unknown>): Promise<unknown>;
  close(): void;
}
```

---

## Error Handling

### Typed Error Classes

| Error Class | Code Type | Purpose |
|---|---|---|
| `FileError` | `FileErrorCode` | File operations (not_found, permission_denied, etc.) |
| `ExecutionError` | `ExecutionErrorCode` | Execution failures (timeout, spawn_error, etc.) |
| `SessionError` | `SessionErrorCode` | Session failures |
| `CompactionError` | `CompactionErrorCode` | Compaction failures |
| `BranchSummaryError` | `BranchSummaryErrorCode` | Branch summarization failures |
| `BackendError` | `BackendErrorCategory` | LLM backend failures (context_full, rate_limit, transient, etc.) |
| `AgentError` | `AgentErrorType` | General agent errors |

### Error Classification

Backend errors are classified at the HTTP boundary:

```typescript
export type BackendErrorCategory =
  | "context_full"    // Prompt exceeds context window
  | "rate_limit"      // HTTP 429 — retryable
  | "transient"       // HTTP 5xx — retryable
  | "client"          // HTTP 4xx — not retryable
  | "poisoned_history" // Unparseable tool call args
  | "unknown"
```

Retryable errors (`rate_limit`, `transient`) include `retryAfterMs` from `Retry-After` header.

### Error Wrapping

```typescript
export function toError(error: unknown): Error
export function wrapError(message: string, cause?: Error): AgentError
```

Unknown thrown values are normalized to Error instances.

---

## Permission System

### Permission Modes

| Mode | Behavior |
|---|---|
| `acceptAll` | Allow everything (default, legacy) |
| `acceptEdits` | Read-only + file-edit tools allowed |
| `ask` | Read-only tools allowed; everything else asks user |
| `plan` | Read-only tools allowed; everything else denied |

### Rule Syntax

```
"bash"              — every call of the bash tool
"bash(git *)"       — bash calls matching glob "git *"
"edit_file(src/*)"  — edit_file calls matching "src/*"
```

Evaluation order: deny rules → allow rules → mode policy.

---

## Configuration Conventions

### Config Validation

Uses `@sinclair/typebox` for schema-first validation:

```typescript
// agent-core/src/agent/configuration/config-validator.ts
throwOnValidationErrors(schema, config);
```

### Config Loading

- Walks up directory tree for `.logician.json`
- Respects environment variable overrides (`LOGICIAN_CONFIG`, `MCP_CONFIG`)
- Loads `~/.logician/.env` for MCP server env vars

---

## Documentation Conventions

### File Headers

All source files start with a comment block describing purpose:

```typescript
// ── Agent Core Entry Point ─────────────────────────────────────────
// Thin barrel re-exporting the three sub-modules plus the tools barrel.
```

### ADR/PRD/SPEC Documents

Planning documents use YAML frontmatter:

```yaml
---
title: "Doc Title"
analysis_date: "2026-08-08"
---
```

### Test Descriptions

Test names are descriptive strings:

```typescript
void test("frontmatter extensions are parsed (allowed-tools, argument-hint, model)", async () => {
```

---

*Conventions analysis: 2026-08-08*
