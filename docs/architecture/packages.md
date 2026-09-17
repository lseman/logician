# Packages

Nine workspace packages arranged in three architectural layers.

```
Layer 1 — Foundation
├── log-natives      N-API bindings for pi-ast / pi-edit (forked from oh-my-pi)
├── log-snapcompact  Bitmap PNG frame context compaction
├── log-autoresearch Autonomous experiment loop (run → measure → keep/discard)
├── log-eoh          Evolution of Heuristics — session logic, persistence, dashboard
├── log-rag          Hybrid search: dense + BM25, cross-encoder reranking, query rewriting
└── log-eval         Outcome-grounded evaluation runner and reports

Layer 2 — Core
└── log-core → log-snapcompact
    Provider loop, execution harness, hooks, policies, tool contracts,
    conversation state, durable session storage, compaction.
    Must not import feature or application packages.

Layer 3 — Application
├── log-runtime → log-autoresearch, log-core, log-eoh, log-natives,
│                 log-rag, log-snapcompact
│   Composition layer — tools, skills, commands, MCP, memory, plugins,
│   Claude Code adapters, EoH, configuration, transcript state.
│   Facade: AgentRuntime.
│
└── log-tui → log-autoresearch, log-core, log-runtime
    Terminal UI with differential rendering, overlays, input control,
    flex layout, image protocol, session tree view.

         ┌───────────────┐
         │   log-tui     │  ← presentation layer
         ├───────────────┤
         │ log-runtime   │  ← composition / application layer
         ├───────────────┤
         │   log-core    │  ← agent engine
         ├───────────────┤
         │ log-snapcompact│  ← foundation (compression)
         └───────────────┘
```

## Dependency rules

1. **Foundation → Core → Application.** Packages in a lower layer must never
   import packages from a higher layer. This is enforced by the architecture
   contract test in `packages/log-tui/src/__tests__/architecture-contracts.test.ts`.

2. **log-core is product-independent.** It must not import Logician feature
   or application packages. Runtime integrations belong in `log-runtime`.

3. **log-runtime may depend on feature packages.** Feature packages and
   `log-core` must never depend on `log-runtime`.

4. **All workspace imports must be declared.** The architecture test verifies
   that production imports use declared `dependencies`/`devDependencies` and
   reference exported package subpaths.

## Package descriptions

### Foundation

| Package | Scope |
|---|---|
| `@logician/log-natives` | N-API bindings wrapping the `pi-natives` Rust crate (`pi-ast` + `pi-edit`) for structural search/rewrite and streaming edit operations. Platform-specific `*.node` binary built via napi-rs. |
| `@logician/log-snapcompact` | Local, deterministic context compaction via bitmap PNG frames. Zero external runtime dependencies. |
| `@logician/log-autoresearch` | Autonomous experiment loop — run, measure, keep or discard. Ported from `pi-autoresearch`. Provides hooks, paths, JSONL helpers, compaction, and shortcuts for the research agent workflow. |
| `@logician/log-eoh` | Evolution of Heuristics (EoH, arXiv 2401.02051). Session logic, persistence, compaction, hooks, engine, evaluator, LLM integration, population management, and prompts. |
| `@logician/log-rag` | SOTA retrieval-augmented generation: hybrid search (dense + BM25), smart chunking, cross-encoder reranking, query rewriting, context management. |
| `@logician/log-eval` | Outcome-grounded evaluation runner and reports for Logician agent trials. CLI entry point `logician-eval`. |

### Core

| Package | Scope |
|---|---|
| `@logician/log-core` | Lean agent engine: provider loop, `AgentHarness` (functional execution kernel), `AgentSession` (interactive coordination), `SessionStore` (durable JSONL conversation tree), `EventJournal`, `CancellationScope`, compaction. Exports focused subpaths (`./harness`, `./session`, `./event-journal`, `./runtime`, etc.). Depends only on `log-snapcompact`. |

### Application

| Package | Scope |
|---|---|
| `@logician/log-runtime` | Logician runtime composition — the full application facade (`AgentRuntime`). Combines `log-core` with tools, skills, commands, MCP, memory, plugins, Claude Code adapters, EoH, configuration, and transcript state. Exports 16+ subpaths (`./application`, `./commands`, `./skills`, `./tools`, `./trust`, `./sessions`, etc.). |
| `@logician/log-tui` | Terminal UI layer. Differential rendering engine, flex layout, input controller (vi-style), overlays (choice popup, settings, session tree, research dashboard), image protocol (kitty graphics), status/tail bars, work surface management. Depends on `log-autoresearch`, `log-core`, `log-runtime`. |

## Subpath exports

Each package declares explicit subpath exports in `package.json` `exports`.
Importing via a path not listed in `exports` is considered a violation and will
be caught by the architecture contract test.

## Why flat layout

Nine packages with a 3-layer dependency graph and no circular dependencies
fit comfortably in a flat `packages/` directory. Reorganization into
subdirectories (`foundation/`, `core/`, `app/`) would add workspace config
churn without solving a real pain point. This arrangement is revisable if
the package count grows beyond ~12 or the dependency depth exceeds 4 layers.
