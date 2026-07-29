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
