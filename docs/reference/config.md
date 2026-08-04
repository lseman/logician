---
title: Config Schema
description: Full configuration schema with all options.
---

# Config Schema

Complete configuration reference for `.logician.json`.

The current flat runtime keys include:

```json
{
  "baseUrl": "http://127.0.0.1:8080",
  "model": "model-id",
  "memory": false,
  "memoryEmbeddings": false,
  "memoryEmbeddingModel": "Xenova/all-MiniLM-L6-v2",
  "memoryExtractor": {
    "baseUrl": "http://127.0.0.1:8081",
    "model": "small-model-id"
  },
  "memoryViewer": true,
  "memoryViewerPort": 3200,
  "reasoner": "none",
  "reasonerConfig": {}
}
```

`reasoner` defaults to `none`. A selected reasoner runs before the ordinary tool-capable agent loop and its `reasonerConfig` values override that reasoner's registry defaults.

`memoryEmbeddings` defaults to `false`. When enabled, Logician lazily loads the configured local embedding model and fuses its semantic results with SQLite FTS memory retrieval. Lexical retrieval remains available while the model warms.

## Full schema

```json
{
  "llm": {
    "url": "string",           // Required: LLM API endpoint
    "model": "string",         // Required: Model name
    "apiKey": "string",        // Optional: API key
    "maxTokens": "number",     // Optional: Default 8192
    "temperature": "number",   // Optional: Default 0.7
    "topP": "number",          // Optional: Default 1.0
    "stream": "boolean"        // Optional: Default true
  },
  "permissions": "plan|ask|acceptEdits|acceptAll",
  "thinkingLevel": "low|medium|high|full",
  "reasoning": {
    "mode": "string",          // Reasoner: cot, reflexion, tot, etc.
    "maxIterations": "number"  // Default 10
  },
  "compaction": {
    "enabled": "boolean",      // Default true
    "triggerFraction": "number", // Default 0.75
    "strategy": "summarize"    // Summarization strategy
  },
  "mcp": {
    "servers": {
      "name": {
        "command": "string",
        "args": "string[]",
        "env": "Record<string, string>",
        "url": "string",       // For HTTP transport
        "transport": "stdio|http"
      }
    }
  },
  "plugins": ["string"],       // Plugin file paths
  "tools": {
    "allowed": ["string"],     // Tool name allowlist (empty = all)
    "denied": ["string"]      // Tool name denylist
  },
  "access": {
    "allowedPaths": ["string"], // Directories the agent can access
    "deniedPaths": ["string"]   // Directories the agent cannot access
  },
  "headless": {
    "timeout": "number",       // Default 300
    "maxIterations": "number", // Default 20
    "streamOutput": "boolean"  // Default true
  }
}
```

## Environment variable overrides

All config options can be overridden via environment variables:

| Config path | Environment variable |
|---|---|
| `llm.url` | `LOGICIAN_LLM_URL` |
| `llm.model` | `LOGICIAN_LLM_MODEL` |
| `llm.apiKey` | `LOGICIAN_LLM_API_KEY` |
| `llm.maxTokens` | `LOGICIAN_MAX_TOKENS` |
| `llm.temperature` | `LOGICIAN_TEMPERATURE` |
| `permissions` | `LOGICIAN_PERMISSIONS` |
| `thinkingLevel` | `LOGICIAN_THINKING` |
