---
title: "Tech Stack"
analysis_date: "2026-08-08"
---

# Tech Stack

**Analysis Date:** 2026-08-08

*Logician TUI — technology stack, runtimes, frameworks, and dependencies.*

---

## Languages & Runtimes

| Language / Runtime | Version | Scope |
|---|---|---|
| **TypeScript** | 5.7.2 (root), ^6.0.3 (packages) | All source code |
| **Node.js** | >=22.19.0 | Runtime requirement |
| **Bun** | >=1.3.14 (packageManager: bun@1.3.14) | Package manager, test runner, typecheck executor |
| **Python** | >=3.9 | RAG document extraction (Docling subprocess) |

## Package Manager

- **Bun** (bun@1.3.14) — primary package manager and script runner
- **npm** — present in `repos/` sub-projects (pi, gsd-core, pi-autoresearch)
- **pnpm** — present in `repos/pi-autoresearch`

## Project Structure

Monorepo with Bun workspaces under `tui/packages/`:

```
tui/packages/
├── agent-core/          — Core agent engine (loop, harness, hooks, types, compaction, extensions)
├── agent-capabilities/  — Agent capabilities (todo, ask-user, subagents, reasoners, RAG, EoH)
├── coding-agent/        — Orchestration layer (sessions, config, skills, slash commands, MCP, trust)
├── tui/                 — Terminal UI layer (terminal rendering, overlays, app shell)
├── rag/                 — RAG pipeline (Python Docling extraction, usearch vector store, SQLite)
├── memory/              — Persistent memory (SQLite, embeddings, hooks, viewer server)
└── autoresearch/        — Autonomous experiment loop (run/measure/keep-discard)
```

Root-level `tui/package.json` declares the workspaces; root `package.json` is a thin shim.

## Key Dependencies

### Internal (workspace)

| Package | Depends On |
|---|---|
| `@logician/agent-core` | `emphasize`, `ignore`, `yaml` |
| `@logician/agent-capabilities` | `@logician/agent-core`, `@logician/rag` |
| `@logician/coding-agent` | `@logician/agent-core`, `@logician/agent-capabilities`, `@logician/autoresearch`, `@logician/memory`, `ignore` |
| `@logician/tui` | `@logician/agent-capabilities`, `@logician/agent-core`, `@logician/autoresearch`, `@logician/coding-agent`, `string-width` |
| `@logician/rag` | `@huggingface/transformers`, `usearch` |
| `@logician/memory` | `@huggingface/transformers` |
| `@logician/autoresearch` | `@logician/agent-core` |

### External (user-facing packages)

| Package | Version | Purpose |
|---|---|---|
| `@huggingface/transformers` | ^4.2.0 | ONNX runtime for embedding models (RAG, memory) |
| `usearch` | ^2.26.0 | Vector search (RAG similarity search) |
| `yaml` | ^2.9.0 | YAML parsing (config, hooks) |
| `ignore` | ^7.0.5 | .gitignore-style path filtering |
| `emphasize` | ^7.0.0 | Markdown syntax highlighting |
| `string-width` | ^8.2.2 | Terminal string width calculation |
| `@sinclair/typebox` | ^0.34.41 | TypeScript-first JSON Schema validator |

### Dev Dependencies

| Package | Version | Purpose |
|---|---|---|
| `@biomejs/biome` | ^2.5.7 | Linting & formatting (root) |
| `@typescript-eslint/*` | ^8.65.0 | ESLint plugin (tui package) |
| `eslint` | ^10.8.0 | Linting (tui package) |
| `tsx` | ^4.23.1 | TypeScript execution (dev/test) |
| `typescript` | 5.7.2 / ^6.0.3 | Type checking |
| `bun-types` | ^1.0.0 / ^1.3.14 | Bun type definitions |
| `@types/bun` | ^1.3.14 | Bun type definitions |
| `@types/node` | ^22.0.0 / ^26.1.2 | Node.js type definitions |

## Python Stack (rag-python/)

- **Python** >=3.9
- **setuptools** >=68.0
- **Docling** — document extraction (PDF, DOCX, etc.) via subprocess from the `@logician/rag` package
- Project name: `rag-extract`

## Configuration

| File | Purpose |
|---|---|
| `tui/tsconfig.json` | Project references (6 packages) |
| `tui/packages/*/tsconfig.json` | Per-package TypeScript config |
| `biome.json` | Root Biome lint/format rules |
| `tui/packages/tui/eslint.config.js` | ESLint config for TUI package |

## Entry Points

| Entry Point | File |
|---|---|
| Main TUI | `tui/packages/tui/src/index.ts` |
| Headless exec | `tui/packages/tui/src/app/headless-exec.ts` |
| AgentCoreBridge | `@logician/coding-agent/application` |
| Doctor report | `@logician/coding-agent/developer-tools` |

## Build & Test Commands

```bash
bun run dev          # Start TUI (tsx)
bun run start        # Alias for dev
bun test             # Run all package tests (tsx --test)
bun run typecheck    # TypeScript check across workspaces
bun run lint         # Biome lint
bun run format       # Biome format
bun run ci           # typecheck + lint + format:check + test
```

## Test Infrastructure

- **Test runner:** `tsx --test` (Node.js native test runner via tsx)
- **Test files:** `src/__tests__/*.test.ts` in each package
- **Test count:** 294 TypeScript source files across all packages

---

*Tech stack analysis: 2026-08-08*
