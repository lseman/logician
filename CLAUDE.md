# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build CLI binary
bun run build

# Run CLI in dev mode with hot-reload
bun run dev

# Run all tests
pytest tests/

# Run single test
pytest tests/test_example.py::TestExample::test_something -v

# Type check TypeScript
bun run typecheck

# Install dependencies
bun run install

# Clean built artifacts
bun run clean
```

## Architecture Overview

### High-Level Structure

```
logician/
├── cli/                 # TypeScript/React/Ink CLI application
│   └── src/
│       └── index.tsx   # Main CLI entry point
├── src/                 # Python agent core
│   ├── agent/
│   │   └── core.py     # Main Agent class with tool execution loop
│   ├── backends/       # LLM backends (llama_cpp, vllm)
│   ├── db/             # SQLite + Chroma vector store layer
│   ├── reasoners/      # Reasoning strategies (CoT, Reflexion, SSR, ToT)
│   ├── tools/          # ToolRegistry and tool execution
│   └── config.py       # Configuration dataclasses
├── skills/             # Dynamic skills loaded at runtime
│   ├── 00_global/      # Bootstrap, think, scratchpad, todo
│   ├── 01_coding/      # File operations, shell, git, quality checks
│   └── 02_timeseries/  # Time series analysis skills
├── test/               # pytest test suite
├── agent_config.json   # Agent configuration (MCP servers, etc.)
├── Makefile            # Build and dev scripts
└── pyproject.toml      # Project metadata and dependencies
```

### Core Flow

1. **Initialization**: Agent loads configuration from `agent_config.json` and `src/config.py`
2. **Skill Loading**: Dynamic skills loaded from `skills/` directory via `src/tools/__init__.py`
3. **Tool Execution**: Tools routed through `ToolRegistry` with skill-based prioritization
4. **Reasoning**: Reasoning strategies (CoT, Reflexion, SSR, ToT) applied based on `ThinkingConfig`
5. **Memory**: Conversation history persisted in SQLite + vector embeddings in Chroma
6. **RAG**: Context7 MCP server provides remote document retrieval

### Key Components

#### Agent Core (`src/agent/core.py`)
- Main `Agent` class orchestrating tool execution loop
- Manages context window, tracing, and memory integration
- Applies thinking strategies after tool calls and at turn boundaries

#### Tool Registry (`src/tools/__init__.py`)
- Central tool dispatch system
- `load_tools_from_skills()` dynamically loads tools from skill modules
- Integrates MCP clients for external tool access
- SkillCatalog routing for intelligent tool selection

#### Configuration (`src/config.py`)
- `Config`: Core LLM settings (llama_cpp_url, temperature, timeout)
- `ThinkingConfig`: Prompt+reasoner combinations and pipeline ordering
- Tool configuration: `use_toon_for_tools`, `enable_skill_routing`, `auto_compact`
- Verification patterns for write operations

#### Reasoners (`src/reasoners/`)
- `auto_cot.py`: Automatic chain-of-thought
- `reflexion.py`: Self-critique loop
- `ssr.py`: Self-supervised refinement
- `tot.py`: Tree-of-thoughts exploration

#### Skills System
Skills are Python modules in `skills/` directory that extend agent capabilities:
- Each skill defines tools and/or prompts
- Loaded dynamically based on routing configuration
- Organized by domain (global, coding, timeseries)

## Testing

```bash
# Run all tests with coverage
pytest test/ --cov=src --cov-report=term-missing

# Run single test file
pytest tests/test_mcp_context7.py -v

# Run specific test
pytest tests/test_example.py::TestExample::test_something -v
```

## Environment Variables

```bash
AGENT_LOG_LEVEL      # Logging level (default: ERROR)
AGENT_LOG_JSON       # Enable JSON logging (0/1)
```

## MCP Servers

Context7 MCP server configured in `agent_config.json`:
- URL: https://api.context7.com/mcp
- Provides remote document retrieval for RAG

## Coding Guidelines

- Python 3.10+ with type hints
- Clean, idiomatic code; explicit over implicit
- No unnecessary abstractions
- Prefer clarity and reproducibility for research code
