# Coding agent architecture

The coding agent is the product orchestration layer above `agent-core` and
`agent-capabilities`.

- `runtime/` owns the bridge, event translation, and scheduled loops.
- `sessions/` owns transcript projection and durable session storage.
- `configuration/` owns user-facing configuration loading and persistence.
- `commands/` owns slash-command discovery and dispatch contracts.
- `tools/`, `skills`, `prompts/`, `context-files/`, `mcp/`, and `trust/` are
  feature modules consumed by the runtime.

Implementation files live under their responsibility-oriented directory
(`runtime/bridge.ts`, `sessions/session-store.ts`, etc). Legacy flat subpath
exports (`@logician/coding-agent/bridge`, `/config`, `/slash-commands`, ...)
remain in package.json for external back-compat, but now resolve into these
directories rather than root-level files.
