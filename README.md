# Logician

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/logician?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

*A lightweight, deterministic LLM agent framework built on llama.cpp with tool calling, RAG, persistent memory, and structured tracing."

---

## 🎯 What Is Logician?

Logician is a **production-grade agent framework** for building **tool-using AI agents** on top of `llama.cpp`. It provides:

- **Deterministic tool execution** (JSON or TOON format)
- **Persistent memory** with semantic search over conversations
- **RAG integration** for external knowledge
- **Structured logging & tracing** for debugging and monitoring
- **Self-reflection** loops for iterative improvement
- **llama.cpp backend** for local, privacy-preserving inference

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/lseman/logician.git
cd logician
pip install -e .

# Run a demo
python apps/runners/demo.py

# Or start a REPL
python apps/runners/repl_demo.py
```

### Basic Usage

```python
from agent import Agent
from agent.tools import ToolRegistry

# Register your tools
tool_registry = ToolRegistry()

# Create agent
agent = Agent(
    model="llama3:8b",
    tools=[tool_registry],
    memory_path=".memory",
)

# Run with reflection
result = agent.run(
    prompt="Analyze this time series and forecast next 30 days",
    max_turns=5,
    use_reflection=True,
)

print(result.answer)
print(result.trace_md)  # Markdown timeline
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent Core                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Planning    │→│   Acting      │→│   Observing       │  │
│  │  (think)     │  │   (tools)    │  │   (verify)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                         Backends                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ llama.cpp    │  │  vLLM        │  │  MCP Client      │  │
│  │ (local)      │  │ (GPU)        │  │ (model context)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Memory & Storage                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  MessageDB   │  │  DocumentDB  │  │  RAG Vector Store │  │
│  │  (SQLite)    │  │  (Chroma)    │  │  (Chroma)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🛠️ Tool Registry

- **Typed parameters** for safe tool calling
- **Dynamic tool loading** via `add_tool(...)` API
- **Dual format support**: JSON or TOON (Token-Oriented Object Notation)

```python
from agent.tools import ToolParameter, ToolRegistry

tool_registry.add_tool(
    name="fetch_data",
    description="Fetch time series data from CSV",
    parameters={
        "filepath": ToolParameter(type="string", required=True),
        "date_column": ToolParameter(type="string", required=True),
    },
)
```

### 🧠 Persistent Memory (MessageDB)

- **SQLite storage** for conversation history
- **Semantic search** via Chroma + SentenceTransformers
- **Auto-summarization** of long histories into SYSTEM prompts

### 📚 RAG Document Store

- **Separate Chroma collection** for external documents
- **Chunking & metadata** with top-k retrieval
- **Injected as SYSTEM context** in prompts

```python
from agent import DocumentDB
db = DocumentDB(collection_name="docs")
db.add_directory("./docs/")

# Documents appear as SYSTEM context in agent prompts
```

### 🔄 Self-Reflection

- **Second-pass critique** of final answers
- **Optional refinement** or additional tool calls
- **Configurable** prompt and token budget

```python
result = agent.run(
    prompt="Debug this code",
    use_reflection=True,  # Enable self-reflection
    reflection_prompt="Review your answer for accuracy and completeness",
)
```

### 📊 Logging & Tracing

- **Structured logging** via `AGENT_LOG_LEVEL` and `AGENT_LOG_JSON`
- **Per-module loggers**: `agent`, `agent.tools`, `agent.db`, `agent.rag`
- **Trace output**:
  - `debug`: structured JSON trace (events, timings, config)
  - `trace_md`: markdown timeline for UI rendering

### ⚙️ llama.cpp Backend

- **Dual API support**: `/v1/chat/completions` and `/completion`
- **Streaming** with token callbacks
- **Retry with backoff**, configurable stop tokens
- **Temperature, max tokens**, and other inference parameters

---

## 📦 Project Structure

```
agent/
├── src/
│   ├── agent/          # Core agent logic
│   │   ├── core.py     # Main agent loop
│   │   ├── trace.py    # Structured tracing
│   │   ├── memory.py   # Memory management
│   │   └── tools/      # Tool registry
│   ├── backends/       # Model backends
│   ├── db/             # Database layer
│   ├── eoh/            # Evolution of Heuristics (meta-learning)
│   ├── mcp/            # Model Context Protocol client
│   └── reasoners/      # Reasoning strategies (CoT, ToT, etc.)
├── apps/
│   ├── runners/        # Demo and diagnostic runners
│   └── plotting/       # Visualization helpers
├── skills/             # Tool definitions
├── tests/              # Unit and integration tests
├── docs/               # Documentation
├── pyproject.toml      # Dependencies and build config
└── README.md
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `AGENT_LOG_JSON` | Emit JSON logs | `false` |
| `AGENT_MODEL_PATH` | Path to llama.cpp model | `./models/` |
| `AGENT_MEMORY_PATH` | Memory storage path | `.memory` |
| `AGENT_TOON_FORMAT` | Use TOON format | `false` |

### Agent Parameters

```python
agent = Agent(
    model="llama3:8b",
    tools=[tool_registry],
    memory_path=".memory",
    max_turns=10,
    temperature=0.7,
    use_reflection=True,
    toon_format=False,
)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_agent.py

# Run with coverage
pytest tests/ --cov=src/agent
```

---

## 🚧 Roadmap

- [ ] **Python SDK** for easier integration
- [ ] **UI dashboard** for tracing and monitoring
- [ ] **Multi-agent collaboration** workflows
- [ ] **Fine-tuning support** for custom models
- [ ] **Event streaming** for real-time tracing

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for the inference backend
- [Chroma](https://www.trychroma.com/) for vector storage
- [SentenceTransformers](https://www.sbert.net/) for embeddings

---

**Need help?** Open an issue or join the discussion.
