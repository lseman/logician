# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest test/

# Run single test file
pytest test/test_mcp_context7.py -v

# Run specific test
pytest test/test_example.py::TestExample::test_something -v

# Run tests with coverage
pytest test/ --cov=src --cov-report=term-missing

# Type check
mypy src/

# Lint / format
ruff check src/ skills/
ruff format src/ skills/

# Build Rust CLI
cd rust-cli && cargo build --release

# Run Rust CLI tests
cd rust-cli && cargo test

# Check Rust code
cd rust-cli && cargo clippy
```

## Architecture Overview

### High-Level Structure

```
agent/
├── src/                  # Python agent core
│   ├── agent/
│   │   └── core.py       # Main Agent class with tool execution loop
│   ├── backends/         # LLM backends (llama_cpp, vllm)
│   ├── db/               # SQLite + Chroma vector store layer
│   ├── eoh/              # Evolution of Heuristics module
│   ├── mcp/              # MCP client integration
│   ├── memory.py         # Conversation memory management
│   ├── reasoners/        # Reasoning strategies (CoT, Reflexion, SSR, ToT)
│   ├── thinking.py       # Pre/post-tool thinking logic
│   ├── tools/            # ToolRegistry and tool execution
│   └── config.py         # Configuration dataclasses
├── skills/               # Dynamic skills loaded at runtime
│   ├── 10_superpowers/   # 14 core agent meta-skills (debugging, planning, TDD, etc.)
│   ├── 20_ralph/         # Ralph PRD + autonomous execution format
│   ├── 30_anthropics/    # Anthropic-sourced skill patterns (frontend, MCP builder, etc.)
│   ├── global/           # Session utilities: think, scratch, todo, orchestrator
│   ├── coding/           # Coding skills: explore, edit_block, multi_edit, search_replace,
│   │                     #   patch, quality, shell, git, repl, web
│   ├── timeseries/       # Data loading, preprocessing, analysis, forecasting, plotting
│   ├── academic/         # Literature review + systematic search (S2, OpenAlex, IEEE)
│   ├── svg/              # SVG diagram and visual asset generation
│   ├── rag/              # RAG ingestion, retrieval, and tuning
│   └── qol/              # Context ingestion: firecrawl, docling
├── rust-cli/             # Rust TUI binary
│   └── src/
│       ├── main.rs
│       ├── ui.rs         # Ratatui-based TUI
│       └── bridge.rs     # Python bridge (logician_bridge.py)
├── logician_bridge.py    # Python bridge: HTTP ↔ LLM backend translator
├── logician_messaging.py # Message formatting and history management
├── agent_config.json     # Agent runtime configuration
├── SOUL.md               # Agent operating charter (identity, routing, workflows)
├── SKILLS.md             # Skills index and conventions
├── skills_health.md      # Skill routing diagnostic checklist
├── common_mistakes.md    # Common agent error patterns to avoid
├── Makefile              # Quick dev shortcuts
└── pyproject.toml        # Project metadata and dependencies
```

### Core Flow

1. **Initialization**: Agent loads config from `agent_config.json` and `src/config.py`
2. **Skill Loading**: Skills dynamically loaded from `skills/` via `src/tools/__init__.py`
3. **Skill Routing**: Hybrid BM25 + dense embedding + fuzzy routing selects skills (see `env.AGENT_SKILL_ROUTING_WEIGHTS` in config)
4. **Tool Execution**: Tools routed through `ToolRegistry`
5. **Reasoning**: CoT / Reflexion / SSR / ToT applied per `ThinkingConfig`
6. **Memory**: History persisted in SQLite + vector embeddings in ChromaDB
7. **RAG**: Context7 MCP server provides remote doc retrieval; local `rag/` skill for local ingestion

### Key Components

#### Agent Core (`src/agent/core.py`)
- Main `Agent` class with `PLAN → ACT → OBSERVE → VERIFY → REPORT` loop
- Manages context window, tracing, memory, verification gate
- Applies thinking strategies at tool boundaries and turn ends

#### Tool Registry (`src/tools/__init__.py`)
- Central tool dispatch
- `load_tools_from_skills()` dynamically loads from skill modules
- Integrates MCP clients for external tool access
- SkillCatalog: hybrid routing with usage-recency bias

#### Configuration (`src/config.py`)
- `Config`: LLM endpoint, temperature, timeout, context budget
- `ThinkingConfig`: reasoner pipeline ordering and token budgets
- Verification patterns, skill routing weights, auto-compact behavior

#### Reasoners (`src/reasoners/`)
- `auto_cot.py`: Automatic chain-of-thought
- `reflexion.py`: Self-critique loop
- `ssr.py`: Self-supervised refinement
- `tot.py`: Tree-of-thoughts exploration

#### Skills System
- Each skill: `skills/<group>/<skill_name>/SKILL.md` + optional `scripts/` dir
- `SKILL.md` YAML frontmatter: `name`, `description`, `aliases`, `triggers`, `preferred_tools`, `when_not_to_use`, `next_skills`
- Skills export tools via `__tools__ = [...]` in their script modules

#### Rust TUI (`rust-cli/`)
- Ratatui-based terminal UI
- Communicates with Python core via `logician_bridge.py`
- Config loaded from `agent_config.json` in the run directory

## Testing

```bash
# Full test suite
pytest test/ --cov=src --cov-report=term-missing

# MCP integration tests
pytest test/test_mcp_context7.py -v

# Rust unit tests
cd rust-cli && cargo test
```

## Environment Variables

```bash
AGENT_LOG_LEVEL                    # Logging level (default: ERROR)
AGENT_LOG_JSON                     # Enable JSON logging (0/1)
AGENT_SKILL_DENSE_ENABLED          # Enable dense skill routing (0/1)
AGENT_SKILL_DENSE_MODEL            # Embedding model for dense routing
AGENT_SKILL_ROUTING_RECALL_K       # Top-K candidates in dense retrieval
AGENT_SKILL_ROUTING_MIN_SCORE      # Minimum routing score threshold
AGENT_SKILL_USAGE_HALF_LIFE_HOURS  # Recency bias decay for usage scoring
AGENT_SKILL_ROUTING_WEIGHTS        # Comma-sep routing signal weights
```

## MCP Servers

Context7 configured in `agent_config.json`:
- URL: `https://mcp.context7.com/mcp`
- Provides remote documentation retrieval

## External API Keys (in `agent_config.json` → `env`)

- `S2_API_KEY` — Semantic Scholar
- `OPENALEX_MAILTO` — OpenAlex
- `IEEE_API_KEY` — IEEE Xplore
- `UNPAYWALL_EMAIL` — Unpaywall OA resolver

## Coding Guidelines

- Python 3.10+ with type hints throughout
- Clean, idiomatic code; explicit over implicit
- No unnecessary abstractions
- Prefer clarity and reproducibility for research code
- Run `ruff check` + `mypy` before committing Python changes
- Run `cargo clippy` before committing Rust changes
