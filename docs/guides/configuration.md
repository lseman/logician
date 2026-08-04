---
title: Configuration
description: .logician.json, environment variables, and runtime settings.
---

# Configuration

Logician can be configured via `.logician.json`, environment variables, or runtime commands.

## Configuration file

Create `.logician.json` in the project root:

```json
{
  "llm": {
    "url": "http://127.0.0.1:8080",
    "model": "gpt-4o",
    "apiKey": "sk-...",
    "maxTokens": 8192,
    "temperature": 0.7
  },
  "permissions": "ask",
  "thinkingLevel": "medium",
  "reasoning": {
    "mode": "auto_cot",
    "maxIterations": 10
  },
  "compaction": {
    "enabled": true,
    "triggerFraction": 0.75
  },
  "mcp": {
    "servers": {}
  },
  "plugins": [],
  "tools": {
    "allowed": [],
    "denied": []
  }
}
```

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `LOGICIAN_LLM_URL` | LLM API endpoint | `http://127.0.0.1:8080` |
| `LOGICIAN_LLM_MODEL` | Model name | `gpt-4o` |
| `LOGICIAN_LLM_API_KEY` | API key | (none) |
| `LOGICIAN_PERMISSIONS` | Permission mode | `ask` |
| `LOGICIAN_THINKING` | Thinking level | `medium` |
| `LOGICIAN_MAX_TOKENS` | Max tokens | `8192` |
| `LOGICIAN_TEMPERATURE` | Temperature | `0.7` |
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
2. Runtime `/settings` commands
3. `.logician.json`
4. Built-in defaults
