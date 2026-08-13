# Hook architecture

Logician has four intentionally separate interception systems:

1. `native/` — the typed `AgentHooks` contract and deterministic composition bus.
2. `builtin/` — internal runtime policies such as compaction, budgets, checkpoints,
   and loop protection.
3. `extensions/` — lifecycle events exposed to TypeScript extensions.
4. `plugins/claude-code/` — an adapter for Claude Code plugin manifests,
   event names, stdin payloads, and hook responses.

New runtime behavior should use native hooks. Built-in policy belongs in
`builtin/`; extension-facing lifecycle APIs belong in `extensions/`. Claude
protocol details must not leak outside the compatibility boundary.

## Native runtime guarantees

- Every handler has a stable identity and optional source metadata.
- Higher-priority handlers execute first; equal priorities preserve registration order.
- Duplicate explicit handler IDs are rejected.
- Failures are isolated by default and recorded per handler.
- Diagnostics include executions, errors, timeouts, and elapsed time.
- Owned resources register cleanup callbacks; disposal runs them once in LIFO order.
- A disposed runtime rejects new registrations.
- Every native handler and observer receives an `AbortSignal`.
- Per-handler timeouts abort that signal before the runtime continues.
- Parent cancellation is linked to the handler-scoped signal.
