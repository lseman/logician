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
  "inferenceMode": "thinking-coding",
  "thinkingLevel": "medium",
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

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `LOGICIAN_LLM_URL` | LLM API endpoint | `http://127.0.0.1:8080` |
| `LOGICIAN_MODEL` | Model name | configured model |
| `LOGICIAN_SYSTEM_PROMPT` | System prompt override | configured prompt |
| `LOGICIAN_CONTEXT_WINDOW` | Context-window token count | provider/config value |
| `LOGICIAN_HOOKS` | Enable runtime hooks (`0` disables) | config value |
| `LOGICIAN_MEMORY_EXTRACTOR_URL` | Dedicated semantic-memory model endpoint | Primary LLM endpoint |
| `LOGICIAN_MEMORY_EXTRACTOR_MODEL` | Model used for semantic-memory extraction | Active model |
| `LOGICIAN_MEMORY_EMBEDDINGS` | Enable local semantic memory retrieval | `false` |
| `LOGICIAN_MEMORY_EMBEDDING_MODEL` | Local embedding model | `Xenova/all-MiniLM-L6-v2` |
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

## Dedicated memory extractor

Memory extraction can run against a smaller model on a separate OpenAI-compatible endpoint:

```json
{
  "memory": true,
  "memoryExtractor": {
    "baseUrl": "http://127.0.0.1:8081",
    "model": "mistralai/Ministral-3-3B-Instruct-2512"
  }
}
```

The primary agent continues using `baseUrl` and `model`. Extraction runs through a durable SQLite-backed background queue, so it does not delay turn completion and unfinished jobs recover after restart. Evidence payloads are bounded and redacted before persistence. If the extractor endpoint is unavailable or returns invalid or ungrounded output, memory creation falls back to deterministic synthesis.

## Local semantic memory retrieval

Semantic retrieval is optional and disabled by default. Enable it alongside memory with:

```json
{
  "memory": true,
  "memoryEmbeddings": true,
  "memoryEmbeddingModel": "Xenova/all-MiniLM-L6-v2"
}
```

The quantized embedding model loads lazily and is cached by the local Transformers runtime. Its first enablement may download model files. Logician continues using fast SQLite FTS retrieval while the model warms, then fuses lexical and semantic ranks without requiring a separate vector service. New episodes and consolidated memories are embedded in the background.

Retrieved context injects compact summaries of relevant durable memories. The agent can call the read-only `memory_get` tool with up to 20 displayed memory or observation IDs when it needs complete rationale or evidence, keeping routine prompts smaller.

Memory retrieval searches both durable memories and the episodic observation store, using reciprocal-rank fusion with task, file, phase, recency, and semantic signals. Durable memories are injected by default. Current-session observations are never reinjected because they already exist in the active transcript; up to three strongly relevant prior observations are used only when no durable memory answers the retrieval query. Every injected entry is a compact stable-ID card that can be expanded with `memory_get`.

Model-extracted claims are probationary until corroborated by independent
evidence. Retrieval outcome feedback updates a versioned contextual policy in
shadow mode only, so learned recommendations cannot change prompts without a
separate repeated evaluation gate. See
[Evolving memory](../architecture/evolving-memory.md) for lifecycle, validity,
security, and rollout details.

## Runtime settings

View and modify settings during a session:

```
/settings list
/settings set permissions acceptEdits
/settings set reasoning.mode tot
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
