---
title: "Integrations"
analysis_date: "2026-08-08"
---

# Integrations

**Analysis Date:** 2026-08-08

*Logician TUI — external APIs, MCP servers, web services, and inter-process integrations.*

---

## MCP (Model Context Protocol) Servers

The coding-agent layer provides a full MCP client/manager for integrating external tool servers.

| Component | File |
|---|---|
| MCP Client (stdio) | `tui/packages/coding-agent/src/mcp/client.ts` — `StdioMcpClient` |
| MCP Client (HTTP) | `tui/packages/coding-agent/src/mcp/client.ts` — `HttpMcpClient` |
| MCP Manager | `tui/packages/coding-agent/src/mcp/manager.ts` — `McpManager` |
| MCP Barrel | `tui/packages/coding-agent/src/mcp/index.ts` |

### MCP Protocol

- **Protocol version:** `2025-03-26`
- **Transport:** Stdio (spawned child process) and HTTP
- **Message format:** JSON-RPC with Content-Length framing
- **Config discovery:** Walks up directory tree for `.logician.json`; also respects `LOGICIAN_MCP_CONFIG`, `MCP_CONFIG`, `LOGICIAN_CONFIG` env vars

### MCP Server Configuration

```typescript
interface McpServerConfig {
  enabled?: boolean;
  type?: string;
  command?: string;       // stdio server binary
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
  url?: string;           // HTTP server URL
  headers?: Record<string, string>;
  timeout?: number;       // seconds (default: 30)
}
```

---

## Web Tools

Built-in tools for fetching and searching the web.

| Tool | File | Purpose |
|---|---|---|
| `web_fetch` | `tui/packages/coding-agent/src/tools/web-fetch.ts` | Fetch and extract readable content from URLs |
| `web_search` | (coding-agent tools) | Search the web via SearXNG |

### SearXNG Integration

Web search routes through a local SearXNG instance (default: `http://127.0.0.1:8090`).

---

## Python RAG Pipeline

The RAG package (`@logician/rag`) integrates with Python for document extraction.

| Component | File | Purpose |
|---|---|---|
| Docling extraction | `tui/packages/rag/src/ingestion.ts` | Python subprocess calling Docling |
| Embedder | `tui/packages/rag/src/embedder.ts` | HuggingFace Transformers ONNX inference |
| Vector store | `tui/packages/rag/src/store/sqlite-store.ts` | SQLite-backed vector storage |
| Pipeline | `tui/packages/rag/src/pipeline/index.ts` | Orchestration layer |

### Python Dependencies

- **Docling** — document extraction (PDF, DOCX, etc.)
- **@huggingface/transformers** — ONNX runtime for embedding models
- **usearch** — vector similarity search

Python package lives in `rag-python/` with `pyproject.toml` (project name: `rag-extract`).

---

## Persistent Memory

The memory package (`@logician/memory`) provides SQLite-backed persistent memory with hooks integration.

| Component | File | Purpose |
|---|---|---|
| Memory store | `tui/packages/memory/src/store/` | SQLite persistence |
| Embeddings | `tui/packages/memory/src/embeddings/local-embedder.js` | Local ONNX embeddings |
| Hooks | `tui/packages/memory/src/hooks/` | Session hooks for capture/injection |
| Viewer server | `tui/packages/memory/src/viewer/viewer-server.js` | HTTP viewer server for memory browsing |

### Viewer Server

- Provides an HTTP endpoint for browsing memory/observations
- Port resolution via `getBoundViewerPort()`

---

## Trust System

The trust package (`@logician/coding-agent/src/trust/`) manages trust decisions for tool execution.

| Component | File | Purpose |
|---|---|---|
| Trust store | `tui/packages/coding-agent/src/trust/index.ts` | Persistent trust decisions |
| Trust prompt overlay | `tui/packages/tui/src/overlays/trust-prompt-overlay.ts` | Interactive trust prompt in TUI |
| Trust resolution | `tui/packages/coding-agent/src/trust/` | Policy engine for tool execution |

---

## Plugin System

Agent-core provides a plugin backend for extending functionality.

| Component | File | Purpose |
|---|---|---|
| Plugin runner | `tui/packages/agent-core/src/tools/shared/plugins.ts` | `runPluginBackend()` |
| JSON utils | `tui/packages/agent-core/src/tools/shared/json-utils.ts` | JSON parsing with comments |

---

## External Repositories (repos/)

The `repos/` directory contains related projects that may be integrated or referenced:

| Repository | Description |
|---|---|
| `repos/claude-mem/` | Claude Code memory plugin (bun, TypeScript) |
| `repos/pi/` | Pi coding agent (npm, TypeScript) |
| `repos/gsd-core/` | GSD workflow core (npm, TypeScript) |
| `repos/pi-autoresearch/` | Pi autoresearch (pnpm, TypeScript) |
| `repos/agentmemory/` | Agent memory service (Docker, TypeScript) |

---

## Terminal & UI Integrations

| Component | File | Purpose |
|---|---|---|
| Terminal core | `tui/packages/tui/src/terminal/core.ts` | Terminal detection, width calculation |
| Theme system | `tui/packages/tui/src/terminal/theme.ts` | Color themes, truecolor support |
| Goal runner | `tui/packages/tui/src/app/goal-runner.ts` | Multi-turn goal execution |
| Headless exec | `tui/packages/tui/src/app/headless-exec.ts` | Non-interactive execution mode |

### Terminal Features

- **Truecolor:** Forces `FORCE_COLOR=3` for hex color support
- **Width calculation:** Uses `string-width` for accurate terminal width
- **Sanitization:** Terminal output sanitization for safe display

---

## Environment & Configuration

| Mechanism | File/Env | Purpose |
|---|---|---|
| Home env loader | `tui/packages/tui/src/index.ts` | Loads `~/.logician/.env` for MCP env vars |
| MCP config env | `MCP_CONFIG`, `LOGICIAN_MCP_CONFIG` | Override MCP config path |
| MCP debug | `LOGICIAN_MCP_DEBUG=1` | Enable MCP debug logging |
| Logician config | `LOGICIAN_CONFIG` | Override main config path |

---

*Integrations analysis: 2026-08-08*
