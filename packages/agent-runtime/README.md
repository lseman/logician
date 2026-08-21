# `@logician/agent-runtime`

The composition layer for complete Logician applications.

It combines `@logician/agent-core` with tools, skills, commands, MCP, memory,
plugins, Claude Code adapters, EoH, configuration, transcript state, and
TUI-facing application modules.

Use `AgentRuntime` as the application facade. `AgentCoreBridge` remains only as a
deprecated compatibility alias.

`agent-runtime` may depend on feature packages; feature packages and `agent-core`
must never depend on `agent-runtime`. Architecture tests enforce this package
graph and verify that production workspace imports are declared.
