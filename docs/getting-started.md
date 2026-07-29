---
title: Getting Started
description: Install Logician and run your first agent session.
---

# Getting Started

This guide gets you from zero to your first agent session in under 5 minutes.

## Prerequisites

| Requirement | Details |
|---|---|
| Node.js | `>=22.19.0` |
| LLM backend | OpenAI-compatible API (any provider) |
| `rg` (ripgrep) | Optional — speeds up search tools |
| `fd` | Optional — faster file finding |
| SearXNG | Optional — powers `web_search` |

## Installation

```bash
git clone https://github.com/lseman/logician.git
cd logician/tui
npm install
```

## Configuration

Create `.logician.json` in the project root:

```json
{
  "llm": {
    "url": "http://127.0.0.1:8080",
    "model": "your-model-name",
    "apiKey": "your-api-key"
  },
  "permissions": "ask",
  "thinkingLevel": "medium"
}
```

Or use environment variables:

```bash
export LOGICIAN_LLM_URL=http://127.0.0.1:8080
export LOGICIAN_LLM_MODEL=your-model-name
export LOGICIAN_LLM_API_KEY=your-api-key
```

## Running the TUI

```bash
npm start
```

The TUI starts in interactive mode. Type your instruction and press Enter. The agent will stream its reasoning, tool calls, and results in real time.

## Headless mode

For CI/CD pipelines or scripted workflows:

```bash
npm start -- exec --jsonl "fix the failing test in src/utils.ts"
```

This outputs JSONL (one JSON object per line) with the full reasoning trace, tool calls, and final result.

## Next steps

- Read the [Guides](/guides/overview) for in-depth topics
- Try the [First Session tutorial](/tutorials/first-session)
- Explore the [API reference](/reference/api)
