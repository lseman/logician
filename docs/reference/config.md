---
title: Config Schema
description: Full configuration schema with all options.
---

# Config Schema

Configuration uses the same validated keys in user
`~/.logician/settings.json` and trusted project `.logician.json` files. Unknown
keys produce diagnostics rather than silently changing runtime behavior.

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
  "reasonerConfig": {},
  "thinkingLevel": "off",
  "inferenceMode": "none"
}
```

`reasoner` defaults to `none`. A selected reasoner runs before the ordinary tool-capable agent loop and its `reasonerConfig` values override that reasoner's registry defaults.

`thinkingLevel` defaults to `off`. `inferenceMode` defaults to `none`, labeled **Provider** in the UI, which omits Logician's sampling presets and lets the provider use its defaults.

`memoryEmbeddings` defaults to `false`. When enabled, Logician lazily loads the configured local embedding model and fuses its semantic results with SQLite FTS memory retrieval. Lexical retrieval remains available while the model warms.

## Common settings

```json
{
  "baseUrl": "string",
  "model": "string",
  "maxTokens": "positive number",
  "temperature": "number from 0 to 2",
  "permissionMode": "plan|ask|acceptEdits|acceptAll",
  "permissions": { "allow": ["string"], "deny": ["string"] },
  "inferenceMode": "auto|none|thinking-general|thinking-coding|instruct-general|instruct-reasoning|instruct-coding|deterministic|creative|analytical",
  "thinkingLevel": "off|minimal|low|medium|high|xhigh",
  "executionProfile": "autonomous|minimal",
  "toolExecution": "parallel|sequential",
  "compaction": {
    "enabled": "boolean",
    "reserveTokens": "positive number",
    "keepRecentTokens": "positive number"
  },
	"legroom": {
		"mode": "off|sdk",
		"python": "python3",
		"args": ["-m", "legroom.sdk_worker"],
		"failOpen": "boolean (default true)",
		"timeoutMs": "positive number (default 30000)",
		"config": {}
	},
  "maxParallelAgents": "positive number",
  "mcpServers": {
    "name": { "command": "string", "args": ["string"], "enabled": true }
  },
  "allowedPaths": ["absolute path"],
  "allowAllPaths": "boolean",
  "lsp": { "enabled": "boolean", "timeoutMs": "positive number" },
  "truncation": {},
  "plugins": {}
}
```

## Environment variable overrides

Environment overrides exist for selected deployment-sensitive settings:

| Config path | Environment variable |
|---|---|
| `baseUrl` | `LOGICIAN_LLM_URL` |
| `model` | `LOGICIAN_MODEL` |
| `systemPrompt` | `LOGICIAN_SYSTEM_PROMPT` |
| `contextWindowTokens` | `LOGICIAN_CONTEXT_WINDOW` |
| `hooks` | `LOGICIAN_HOOKS` |

Logician reads `~/.logician/.env` before resolving MCP servers. It is intended for secrets referenced as `${VARIABLE}` in MCP `headers` or `env` maps.
