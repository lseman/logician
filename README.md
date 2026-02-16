# Logician

![Logician Logo](logo/logician-banner-light.svg#gh-light-mode-only)
![Logician Logo](logo/logician-banner.svg#gh-dark-mode-only)


*A LLM agent framework for llama.cpp with tools, RAG, and rich execution traces.*

Logician is a small, batteries-included framework for building **tool-using agents** on top of `llama.cpp`. It focuses on:

- **Deterministic tool calling** (JSON or TOON – Token-Oriented Object Notation)
- **First-class logging & traces** (structured logs + markdown timelines)
- **Persistent memory & semantic search** over conversations
- **Optional RAG** over external documents
- **Self-reflection loops** (optional second-pass critique/improvement)

---

## Features

- 🔧 **Tool Registry**
  - Simple `add_tool(...)` API with typed `ToolParameter`s
  - Model sees a clear, generated tools prompt (JSON or TOON format)
  - Central `ToolRegistry` that logs registration and execution

- 📒 **Persistent Memory (MessageDB)**
  - Conversation history stored in SQLite
  - Semantic search over past messages via Chroma + SentenceTransformers
  - Automatic summarization of long histories into a synthetic SYSTEM message

- 📚 **RAG Document Store (DocumentDB)**
  - Separate Chroma collection for external documents
  - Chunking, metadata, and top-k retrieval injected into prompt as SYSTEM context

- 🧠 **Self-Reflection (optional)**
  - Agent can critique its own final answer and optionally refine or call more tools
  - Configurable reflection prompt and temperature / token budget

- ⚙️ **llama.cpp Client**
  - Supports both `/v1/chat/completions` and `/completion` APIs
  - Streaming (`stream=True`) with token callback
  - Retry with backoff, configurable stop tokens, temperature, and max tokens

- 📈 **Logging & Tracing**
  - Central logging configuration via `AGENT_LOG_LEVEL` and `AGENT_LOG_JSON`
  - Per-module loggers (`agent`, `agent.tools`, `agent.db`, `agent.rag`, `agent.llama`)
  - Every `Agent.run(...)` returns an `AgentResponse` with:
    - `debug`: structured trace (events, timings, config)
    - `trace_md`: markdown timeline for quick inspection or UI rendering

- 🧾 **TOON Tool Calls (Optional)**
  - If `toon_format` is installed, agent can prefer **TOON** over JSON
  - Strict parsing for both TOON and JSON tool call payloads
  - Automatic fallback to JSON when TOON is not available

---

## Installation

Logician is currently intended to be used **from source** inside your project.

```bash
git clone https://github.com/lseman/logician.git
cd logician
pip install -e .

## Project Layout

- `src/`: core framework (agent, memory, DB, tool registry, backends)
- `skills/`: tool definitions loaded by `ToolRegistry`
- `apps/plotting/`: plotting and visualization helpers used by notebooks/scripts
- `apps/runners/`: executable demo/diagnostic runners
- Top-level files like `run.py`, `repl_demo.py`, and `test_clean_session.py` are compatibility entrypoints that forward to `apps/runners/*`
