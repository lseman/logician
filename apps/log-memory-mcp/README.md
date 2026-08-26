# @logician/log-memory-mcp

Stdio MCP adapter for `@logician/memoriam`. It exposes a deliberately small
interface while keeping persistence, retrieval, and memory evolution inside the
memory module.

## Run

The server requires an explicit workspace. Its default database is
`<workspace>/.logician/memory.db`.

```bash
bun run apps/log-memory-mcp/src/index.ts --workspace /absolute/project/path
```

Use `--db /absolute/path/memory.db` to select a different database. The same
values can be supplied through `LOGICIAN_MEMORY_WORKSPACE` and
`LOGICIAN_MEMORY_DB`.

## MCP configuration

```json
{
  "mcpServers": {
    "logician-memory": {
      "command": "bun",
      "args": [
        "run",
        "/absolute/path/to/logician/apps/log-memory-mcp/src/index.ts",
        "--workspace",
        "/absolute/project/path"
      ]
    }
  }
}
```

The server exposes:

- `memory_search`
- `memory_get`
- `memory_save`
- `memory_observe`
- `memory_feedback`

Write tools require an idempotency key. Use a stable event or tool-call ID,
qualified by the originating agent when necessary. Reusing a key within one
workspace returns the original record instead of writing a duplicate.

The adapter writes protocol messages only to stdout. Operational diagnostics
belong on stderr.
