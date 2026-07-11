# Coding agent architecture

The coding agent is the product orchestration layer above `agent-core` and
`agent-capabilities`.

- `runtime/` owns the bridge, event translation, and scheduled loops.
- `sessions/` owns transcript projection and durable session storage.
- `configuration/` owns user-facing configuration loading and persistence.
- `commands/` owns slash-command discovery and dispatch contracts.
- `tools/`, `skills`, `prompts/`, `context-files/`, `mcp/`, and `trust/` are
  feature modules consumed by the runtime.

Legacy root import paths remain supported. New code should import through these
responsibility-oriented entry points so implementation files can be split
without another public API migration.
