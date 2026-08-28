---
title: Configuration
description: .logician.json, environment variables, and runtime settings.
---

# Configuration

Logician resolves configuration once at startup from user settings, trusted
project settings, and environment overrides. The resolved snapshot is passed to
the runtime; the harness and subagents do not reread files independently.

## Configuration file

Put user-wide settings in `~/.logician/settings.json` and trusted project
overrides in `.logician.json`:

```json
{
  "baseUrl": "http://127.0.0.1:8080",
  "model": "model-id",
  "maxTokens": 8192,
  "temperature": 0.7,
  "permissionMode": "ask",
  "permissions": {
    "allow": ["read_file", "grep"],
    "deny": []
  },
  "inferenceMode": "none",
  "thinkingLevel": "off",
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000
  },
  "maxParallelAgents": 4,
  "mcpServers": {
    "docs": {
      "command": "docs-mcp",
      "args": []
    }
  },
  "plugins": {}
}
```

Object-valued sections are merged recursively across user and project layers,
including `permissions`, `compaction`, `lsp.serverOverrides`, MCP servers, and
truncation limits. A project override therefore does not erase unrelated user
settings in the same section. Project configuration is ignored until the
workspace is trusted.

Writes made by selectors and `/settings` update only the selected field in the
user file via an atomic replacement. Starting Logician, switching sessions, or
applying a resolved setting never writes configuration.

## Legroom SDK mode

Legroom can compress the outbound provider payload through a persistent Python
worker, without enabling its HTTP proxy. Install `legroom` in the Python
environment used to launch Logician, then configure:

```json
{
  "legroom": {
    "mode": "sdk",
    "python": "python3",
    "args": ["-m", "legroom.sdk_worker"],
    "failOpen": true,
    "timeoutMs": 30000,
    "config": {
      "protect_recent": 2,
      "use_model_profile": true,
      "ccr_enabled": false
    }
  }
}
```

The worker starts lazily on the first provider request and exits with Logician.
Only outbound `messages` are transformed; stored session history remains
unchanged. `failOpen` defaults to `true`, returning the original messages if
the worker is unavailable, rejects a request, or exceeds `timeoutMs`.

## Environment variables

Logician also loads `~/.logician/.env` at startup. Use it for secrets referenced by MCP header or process-environment placeholders; do not commit secrets to project configuration.

| Variable | Description | Default |
|---|---|---|
| `LOGICIAN_LLM_URL` | LLM API endpoint | `http://127.0.0.1:8080` |
| `LOGICIAN_MODEL` | Model name | configured model |
| `LOGICIAN_SYSTEM_PROMPT` | System prompt override | configured prompt |
| `LOGICIAN_CONTEXT_WINDOW` | Context-window token count | provider/config value |
| `LOGICIAN_HOOKS` | Enable runtime hooks (`0` disables) | config value |
| `LOGICIAN_REASONER` | Structured pre-reasoner ID | `none` |

## Structured reasoners

Reasoners are opt-in and disabled by default. When enabled, the selected reasoner produces advisory analysis before the normal agent loop; the agent still verifies that analysis and can use tools normally.

```json
{
  "reasoner": "reflexion",
  "reasonerConfig": {
    "maxTrials": 2
  }
}
```

Available IDs are `ssr`, `tot`, `got`, `reflexion`, `self_consistency`, `best_of_n`, `auto_cot`, `in_context_cot`, and `cover`. Use `none` to disable them. The `/reasoner` command changes and persists the active mode.

## Persistent memory (memoriam)

Persistent memory is provided by the standalone
[memoriam](https://github.com/lseman/memoriam) engine, embedded as an
out-of-process JSON-lines SDK worker (the same pattern as Legroom above).
Enable it with a `memoriam` block:

```json
{
  "memoriam": {
    "mode": "sdk",
    "python": "/path/to/memoriam/.venv/bin/python",
    "args": ["-m", "memoriam.integration.sdk_worker"],
    "failOpen": true,
    "timeoutMs": 30000,
    "config": { "db_path": "~/.logician/memories.db" }
  }
}
```

Omit the block or set `"mode": "off"` to run without persistent memory.
`failOpen` (default `true`) keeps the agent turn alive if the worker is
unreachable or slow; `config.db_path` is the SQLite database path.

Each turn, the worker retrieves a compact memory context and the runtime
prepends it to the provider payload. `/memory` and `/obs` in the TUI drive the
worker directly — listing, searching, consolidating, and clearing memories and
observations. Memory retrieval searches both durable memories and the episodic
observation store; current-session observations are never reinjected because
they already exist in the active transcript, so retrieval surfaces prior
sessions' knowledge instead.

See [Evolving memory](../architecture/evolving-memory.md) for lifecycle,
validity, and consolidation details.

## Runtime settings

View and modify settings during a session:

```text
/settings
/permissions acceptEdits
/thinking off
/mode
/reasoner reflexion
```

## Priority

Configuration is resolved in this order (highest priority first):
1. Environment variables
2. Explicit runtime selections (which persist only when the user changes them)
3. Trusted project `.logician.json`
4. User `~/.logician/settings.json`
5. Built-in defaults

Run `logician doctor --json` to inspect the selected config path, validation
warnings, and effective backend, permission, MCP, and diagnostics settings. The
doctor is read-only and does not connect to the model or MCP servers.
