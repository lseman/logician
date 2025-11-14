"""
LLM Agent Framework — with built-in logging (default DEBUG to stdout)
Supports TOON (Token-Oriented Object Notation) for tool calls as JSON alternative
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
import uuid
import math
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import httpx
import numpy as np

# ===================== Optional Chroma deps =====================
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    chromadb = None
    HAS_CHROMA = False

# ===================== Optional TOON deps =====================
try:
    from toon_format import encode, decode
    HAS_TOON = True
except ImportError:
    HAS_TOON = False
    encode = decode = None

# =====================================================================
# Logging
# =====================================================================
def _resolve_log_level(value: Optional[str | int]) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip().upper()
        return getattr(logging, value, logging.ERROR)
    return logging.ERROR

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach extras (avoid default record attrs)
        for k, v in record.__dict__.items():
            if k.startswith("_"):
                continue
            if k in ("args","name","msg","levelno","levelname","pathname","filename",
                     "module","exc_info","exc_text","stack_info","lineno","funcName",
                     "created","msecs","relativeCreated","thread","threadName",
                     "processName","process"):
                continue
            try:
                json.dumps({k: v})
                payload[k] = v
            except Exception:
                payload[k] = str(v)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def get_logger(name: str,
               level: Optional[str | int] = None,
               json_logs: Optional[bool] = None) -> logging.Logger:
    """
    Create/reuse a module logger with a single ConsoleHandler.
    Defaults pulled from env or sensible DEBUG defaults.
    """
    logger = logging.getLogger(name)
    # Resolve global defaults once
    root = logging.getLogger()
    if not getattr(root, "_configured_by_agent_logger", False):
        env_level = os.getenv("AGENT_LOG_LEVEL")
        env_json = os.getenv("AGENT_LOG_JSON")
        root.setLevel(_resolve_log_level(env_level or logging.ERROR))
        # Avoid duplicate handlers if user already configured logging
        if not root.handlers:
            ch = logging.StreamHandler()
            if (env_json and env_json.strip().lower() in ("1","true","yes")):
                ch.setFormatter(_JsonFormatter())
            else:
                fmt = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"
                ch.setFormatter(logging.Formatter(fmt=fmt, datefmt="%H:%M:%S"))
            root.addHandler(ch)
        root._configured_by_agent_logger = True # type: ignore[attr-defined]
    # Per-logger tuning (optional)
    if level is not None:
        logger.setLevel(_resolve_log_level(level))
    if json_logs is not None:
        # Replace formatter on root console handler
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler):
                h.setFormatter(_JsonFormatter() if json_logs else logging.Formatter(
                    fmt="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
                    datefmt="%H:%M:%S",
                ))
    return logger

log = get_logger("agent.framework") # module-level logger

# =====================================================================
# Config
# =====================================================================
@dataclass
class Config:
    llama_cpp_url: str = "http://localhost:8080"
    timeout: float = 120.0
    temperature: float = 0.7
    max_tokens: int = 1024
    max_iterations: int = 8
    use_chat_api: bool = True
    chat_template: str = "chatml" # chatml | llama2 | zephyr | simple
    stop: Tuple[str, ...] = ("<|im_end|>", "</s>", "[INST]", "USER:", "<|user|>")
    stream: bool = False # if True, llama.cpp /v1/chat supports "stream": true
    max_consecutive_tool_calls: int = 5 # guardrail
    retry_attempts: int = 2 # http retry attempts with backoff
    # NEW: RAG
    rag_enabled: bool = True
    rag_top_k: int = 3
    chroma_path: str = "message_history.chroma"
    # NEW: Self-Reflection
    enable_reflection: bool = False # Toggle self-reflection loop
    reflection_prompt: str = "You are critiquing your own response. Review the conversation and final output: [FINAL]. If it's incomplete, unclear, or needs tools/refinement, output a tool call or improved text. Otherwise, say 'COMPLETE'."
    # NEW: Logging config
    log_level: str | int = field(default_factory=lambda: os.getenv("AGENT_LOG_LEVEL", "ERROR"))
    log_json: bool = field(default_factory=lambda: os.getenv("AGENT_LOG_JSON", "0").lower() in ("1","true","yes"))
    # NEW: TOON config
    use_toon_for_tools: bool = True  # Use TOON instead of JSON for tool calls

# =====================================================================
# Data Models
# =====================================================================
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True

@dataclass
class Tool:
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable[..., Any]

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class AgentResponse:
    messages: List[Message]
    tool_calls: List[ToolCall]
    iterations: int
    final_response: str
    # NEW:
    debug: Dict[str, Any] = field(default_factory=dict) # raw structured trace
    trace_md: str = "" # pretty printable markdown

# =====================================================================
# Chroma-based MessageDB (replaces FAISS)
# =====================================================================
@dataclass
class MessageDB:
    db_path: str = "agent_sessions.db" # Keep for non-vector data (timestamps, etc.)
    chroma_path: str = "message_history.chroma" # Persistent vector store
    embedding_model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        if not HAS_CHROMA:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        self._log = get_logger("agent.db")
        self._init_sqlite()
        self._init_chroma()
        self.encoder = None

    # --------------- SQLite (metadata only) ---------------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_sqlite(self) -> None:
        self._log.debug("Initializing SQLite at %s", self.db_path)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    tool_call_id TEXT,
                    name TEXT,
                    chroma_id TEXT UNIQUE
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(timestamp);"
            )
            conn.commit()

    # --------------- Chroma ---------------
    def _init_chroma(self) -> None:
        self._log.debug("Initializing Chroma at %s", self.chroma_path)
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name="messages",
            embedding_function=ef,
            metadata={"session_id": "filter_key"},
        )

    def _embed(self, text: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if self.encoder is None:
            self._log.debug("Loading SentenceTransformer(%s)", self.embedding_model_name)
            self.encoder = SentenceTransformer(self.embedding_model_name)
        return self.encoder.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def save_message(self, session_id: str, msg: Message) -> int:
        self._log.debug("Saving message role=%s len=%d session=%s",
                        msg.role.value, len(msg.content), session_id[:8])
        chroma_id = str(uuid.uuid4())
        emb = self._embed(msg.content)
        self.collection.add(
            documents=[msg.content],
            metadatas=[{
                "session_id": session_id,
                "role": msg.role.value,
                "name": msg.name or "",
                "tool_call_id": msg.tool_call_id or "",
            }],
            ids=[chroma_id],
            embeddings=[emb.tolist()],
        )
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id, name, chroma_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, msg.role.value, msg.content, msg.tool_call_id, msg.name, chroma_id),
            )
            rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()
        return int(rowid)

    def load_history(
        self,
        session_id: str,
        limit: int = 20,
        summarize_old: bool = True,
        use_semantic: bool = False,
        semantic_query: Optional[str] = None,
    ) -> List[Message]:
        self._log.debug(
            "Loading history session=%s limit=%d semantic=%s",
            session_id[:8], limit, use_semantic
        )
        if use_semantic and semantic_query:
            results = self.collection.query(
                query_texts=[semantic_query],
                n_results=limit,
                where={"session_id": session_id},
                include=["metadatas", "documents"],
            )
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            msgs = [
                Message(
                    role=MessageRole(meta["role"]),
                    content=doc,
                    name=meta.get("name"),
                    tool_call_id=meta.get("tool_call_id"),
                )
                for doc, meta in zip(docs, metas)
            ]
        else:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT role, content, name, tool_call_id
                       FROM messages
                       WHERE session_id=?
                       ORDER BY id DESC
                       LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
                msgs = [
                    Message(role=MessageRole(r), content=c, name=n, tool_call_id=t)
                    for (r, c, n, t) in reversed(rows)
                ]
        if summarize_old and len(msgs) > 14:
            cut = len(msgs) // 2
            old = msgs[:cut]
            keybits = "; ".join(m.content[:48].replace("\n", " ") for m in old[-4:])
            summary = Message(
                role=MessageRole.SYSTEM,
                content=f"[Summary of {len(old)} prior turns] Key points: {keybits} …",
            )
            msgs = [summary] + msgs[cut:]
        return msgs

    def clear_session(self, session_id: str) -> None:
        self._log.info("Clearing session %s", session_id[:8])
        self.collection.delete(where={"session_id": session_id})
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.commit()

    def list_sessions(self) -> List[Tuple[str, str]]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT session_id, MAX(timestamp) FROM messages GROUP BY session_id ORDER BY MAX(timestamp) DESC"
            ).fetchall()

    def global_semantic_search(
        self, query: str, k: int = 20, session_filter: Optional[str] = None
    ) -> List[Message]:
        self._log.debug("Global semantic search q=%r k=%d filter=%s", query, k, session_filter)
        where = {"session_id": session_filter} if session_filter else {}
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
            include=["metadatas", "documents"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [
            Message(
                role=MessageRole(meta["role"]),
                content=doc,
                name=meta.get("name"),
                tool_call_id=meta.get("tool_call_id"),
            )
            for doc, meta in zip(docs, metas)
        ]

# =====================================================================
# RAG DocumentDB (Chroma-based)
# =====================================================================
@dataclass
class DocumentDB:
    """RAG vector store for external documents (separate from chat history)."""
    chroma_path: str = "rag_docs.chroma" # Persistent dir
    embedding_model_name: str = "all-MiniLM-L6-v2"
    collection_name: str = "default"

    def __post_init__(self) -> None:
        if not HAS_CHROMA:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        self._log = get_logger("agent.rag")
        self._log.debug("Initializing RAG store at %s [%s]", self.chroma_path, self.collection_name)
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=ef
        )

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None,
        chunk_size: int = 512,
    ) -> List[str]:
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        chunks, chunk_metas, chunk_ids = [], [], []
        for i, text in enumerate(texts):
            if len(text) > chunk_size:
                for j, chunk in enumerate([text[k:k+chunk_size] for k in range(0, len(text), chunk_size)]):
                    chunks.append(chunk)
                    chunk_metas.append({**metadatas[i], "chunk": j})
                    chunk_ids.append(f"{ids[i]}_{j}")
            else:
                chunks.append(text)
                chunk_metas.append(metadatas[i])
                chunk_ids.append(ids[i])
        self._log.info("Adding %d chunks to RAG store", len(chunks))
        self.collection.add(documents=chunks, metadatas=chunk_metas, ids=chunk_ids)
        return chunk_ids

    def query(
        self, query: str, n_results: int = 3, where: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        self._log.debug("RAG query %r topk=%d where=%s", query, n_results, where)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return [{"content": doc, "metadata": meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])]

# =====================================================================
# Llama.cpp Client
# =====================================================================
class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        use_chat_api: bool,
        chat_template: str,
        stop: Iterable[str],
        retry_attempts: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.use_chat = use_chat_api
        self.template = chat_template
        self.stop = list(stop)
        self.retry_attempts = max(0, retry_attempts)
        self._log = get_logger("agent.llama")

    def _format_messages(self, messages: List[Message]) -> str:
        if self.template == "chatml":
            out = []
            for m in messages:
                out.append(f"<|im_start|>{m.role.value}\n{m.content}<|im_end|>")
            out.append("<|im_start|>assistant\n")
            return "\n".join(out)
        if self.template == "llama2":
            segs = []
            sys = [m for m in messages if m.role == MessageRole.SYSTEM]
            if sys:
                segs.append(f"<s>[INST] <<SYS>>\n{sys[-1].content}\n<</SYS>>\n")
            for m in messages:
                if m.role == MessageRole.USER:
                    segs.append(f"{m.content} [/INST] ")
                elif m.role == MessageRole.ASSISTANT:
                    segs.append(f"{m.content} </s><s>[INST] ")
            return "".join(segs)
        if self.template == "zephyr":
            out = []
            for m in messages:
                out.append(f"<|{m.role.value}|>\n{m.content}</s>")
            out.append("<|assistant|>\n")
            return "\n".join(out)
        return (
            "".join(f"{m.role.value.upper()}: {m.content}\n" for m in messages)
            + "ASSISTANT: "
        )

    def _request(
        self, client: httpx.Client, method: str, url: str, **kw
    ) -> httpx.Response:
        last_exc = None
        for attempt in range(1, self.retry_attempts + 2):
            try:
                self._log.debug("HTTP %s %s (attempt %d)", method, url, attempt)
                resp = client.request(method, url, **kw)
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                self._log.warning("Request failed (attempt %d/%d): %s",
                                  attempt, self.retry_attempts + 1, e)
                if attempt <= self.retry_attempts:
                    backoff = 0.6 * attempt
                    self._log.debug("Backoff %.2fs", backoff)
                    time.sleep(backoff)
                else:
                    self._log.error("Giving up after %d attempts", attempt)
                    raise
        assert False, last_exc # pragma: no cover

    def generate(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            if self.use_chat:
                payload = {
                    "messages": [{"role": m.role.value, "content": m.content} for m in messages],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": self.stop,
                    "stream": bool(stream),
                }
                url = f"{self.base_url}/v1/chat/completions"
                self._log.info("POST /v1/chat/completions temp=%s max_tokens=%s stream=%s",
                               temperature, max_tokens, stream)
                if stream:
                    r = self._request(client, "POST", url, json=payload)
                    full = []
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            full.append(delta)
                            if on_token:
                                on_token(delta)
                    return "".join(full).strip()
                else:
                    r = self._request(client, "POST", url, json=payload)
                    data = r.json()
                    return data["choices"][0]["message"]["content"].strip()
            else:
                prompt = self._format_messages(messages)
                payload = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "n_predict": max_tokens,
                    "stop": self.stop,
                    "stream": bool(stream),
                }
                url = f"{self.base_url}/completion"
                self._log.info("POST /completion temp=%s n_predict=%s stream=%s",
                               temperature, max_tokens, stream)
                if stream:
                    r = self._request(client, "POST", url, json=payload)
                    full = []
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        tok = data.get("content")
                        if tok:
                            full.append(tok)
                            if on_token:
                                on_token(tok)
                    return "".join(full).strip()
                else:
                    r = self._request(client, "POST", url, json=payload)
                    data = r.json()
                    return data.get("content", "").strip()

# =====================================================================
# Tool Registry
# =====================================================================
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._log = get_logger("agent.tools")

    def register(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        function: Callable[..., Any],
    ) -> "ToolRegistry":
        self._log.info("Registering tool: %s", name)
        self._tools[name] = Tool(name, description, parameters, function)
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def tools_schema_prompt(self, use_toon: bool = True) -> str:
        if not self._tools:
            return ""
        if use_toon:
            lines = [
                "\n\nTOOLS AVAILABLE (use only if needed):",
                "Return EXACT TOON when calling a tool (no extra text):",
                "tool_call:",
                "  name: <tool_name>",
                "  arguments:",
                "    <param1>: <value1>",
                "    ...",
                "",
            ]
        else:
            lines = [
                "\n\nTOOLS AVAILABLE (use only if needed):",
                "Return EXACT JSON when calling a tool:",
                '{"tool_call":{"name":"<tool_name>","arguments":{...}}}',
                "Do NOT add extra text before/after the JSON.",
                "",
            ]
        for t in self._tools.values():
            lines.append(f"Tool: {t.name}")
            lines.append(f" Description: {t.description}")
            if t.parameters:
                lines.append(" Parameters:")
                for p in t.parameters:
                    req = "required" if p.required else "optional"
                    lines.append(f" - {p.name} ({p.type}, {req}): {p.description}")
            lines.append("")
        return "\n".join(lines)

    def execute(self, call: ToolCall, use_toon: bool = True) -> str:
        tool = self.get(call.name)
        if not tool:
            self._log.error("Tool not found: %s", call.name)
            return f"Error: tool '{call.name}' not found."
        try:
            self._log.debug("Executing tool=%s args=%s", call.name, call.arguments)
            out = tool.function(**(call.arguments or {}))
            if isinstance(out, str):
                preview = out[:160].replace("\n", " ")
            else:
                preview = json.dumps(out, ensure_ascii=False)[:160]
            self._log.debug("Tool result preview: %s", preview)
            if use_toon and HAS_TOON and isinstance(out, (dict, list)):
                return encode(out)
            return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        except Exception as e:
            self._log.exception("Error executing tool %s", call.name)
            return f"Error executing '{call.name}': {e}"

# =====================================================================
# Agent (with always-on debug trace + RAG + Self-Reflection)
# =====================================================================
class Agent:
    def __init__(
        self,
        llm_url: str = "http://localhost:8080",
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        use_chat_api: bool = True,
        chat_template: str = "chatml",
        db_path: str = "agent_sessions.db",
        embedding_model: Optional[str] = None,
    ) -> None:
        self.config = config or Config(
            llama_cpp_url=llm_url,
            use_chat_api=use_chat_api,
            chat_template=chat_template,
        )
        # Apply logging config
        get_logger("agent.framework", self.config.log_level, self.config.log_json)
        self._log = get_logger("agent")
        self._log.info("Initializing Agent chat_api=%s template=%s",
                       self.config.use_chat_api, self.config.chat_template)

        # TOON fallback check
        if self.config.use_toon_for_tools and not HAS_TOON:
            self._log.warning("TOON enabled but toon_format not installed. Falling back to JSON.")
            self.config.use_toon_for_tools = False

        self.llm = LlamaCppClient(
            base_url=self.config.llama_cpp_url,
            timeout=self.config.timeout,
            use_chat_api=self.config.use_chat_api,
            chat_template=self.config.chat_template,
            stop=self.config.stop,
            retry_attempts=self.config.retry_attempts,
        )
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.tools = ToolRegistry()
        self.db = MessageDB(
            db_path=db_path,
            chroma_path=self.config.chroma_path,
            embedding_model_name=(embedding_model or "all-MiniLM-L6-v2"),
        )
        self.doc_db: Optional[DocumentDB] = None
        if self.config.rag_enabled and embedding_model:
            self.doc_db = DocumentDB(embedding_model_name=embedding_model)
        self.current_session_id: Optional[str] = None

    def _default_system_prompt(self) -> str:
        if self.config.use_toon_for_tools:
            return (
                "You are a reliable assistant. If a TOOL is needed, output EXACT TOON:\n"
                "tool_call:\n"
                "  name: <tool>\n"
                "  arguments:\n"
                "    ...\n"
                "If no tool is needed, answer normally and clearly."
            )
        else:
            return (
                "You are a reliable assistant. If a TOOL is needed, output EXACT JSON:\n"
                '{"tool_call":{"name":"<tool>","arguments":{...}}}\n'
                "If no tool is needed, answer normally and clearly."
            )

    # ------------- Public API -------------
    def add_tool(
        self,
        name: str,
        description: str,
        function: Callable[..., Any],
        parameters: Optional[List[ToolParameter]] = None,
    ) -> "Agent":
        self.tools.register(name, description, parameters or [], function)
        return self

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        verbose: bool = False,
        use_semantic_retrieval: bool = False,
        stream: Optional[Callable[[str], None]] = None,
    ) -> str:
        resp = self.run(
            message,
            session_id=session_id,
            verbose=verbose,
            use_semantic_retrieval=use_semantic_retrieval,
            stream_callback=stream,
        )
        return resp.final_response

    def reset(self, session_id: Optional[str] = None) -> "Agent":
        sid = session_id or self.current_session_id
        if sid:
            self.db.clear_session(sid)
        self.current_session_id = None
        return self

    def list_sessions(self) -> List[Tuple[str, str]]:
        return self.db.list_sessions()

    def semantic_search(self, query: str, session_id: str, k: int = 8) -> List[Message]:
        return self.db.global_semantic_search(query, k, session_filter=session_id)

    # ------------- Orchestration -------------
    def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        use_semantic_retrieval: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> AgentResponse:
        started_ts = time.perf_counter()
        debug_events: List[Dict[str, Any]] = []

        def _event(kind: str, **data: Any) -> None:
            debug_events.append({"t": round(time.perf_counter() - started_ts, 6), "kind": kind, **data})

        temp = temperature if temperature is not None else self.config.temperature
        n_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        self.current_session_id = session_id or str(uuid.uuid4())
        self._log.info("Run message=%r session=%s", message, self.current_session_id[:8])
        # Build conversation
        convo: List[Message] = [
            Message(
                role=MessageRole.SYSTEM,
                content=self.system_prompt + self.tools.tools_schema_prompt(self.config.use_toon_for_tools),
            )
        ]
        history = self.db.load_history(
            self.current_session_id,
            limit=18,
            summarize_old=True,
            use_semantic=use_semantic_retrieval,
            semantic_query=message if use_semantic_retrieval else None,
        )
        convo.extend(history)
        # NEW: RAG injection
        if self.doc_db and self.config.rag_enabled:
            rag_results = self.doc_db.query(message, n_results=self.config.rag_top_k)
            if rag_results:
                rag_context = "\n\n".join(
                    f"[Doc: {r['metadata'].get('source', 'unknown')}] {r['content']}"
                    for r in rag_results
                )
                convo.append(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=f"Use this relevant document context if helpful: {rag_context}",
                    )
                )
                _event("rag_retrieval", n_results=len(rag_results), preview=rag_context[:200])
                self._log.debug("RAG hits=%d preview=%r", len(rag_results), rag_context[:120])
        user_msg = Message(role=MessageRole.USER, content=message)
        convo.append(user_msg)
        self.db.save_message(self.current_session_id, user_msg)
        _event("user_message", session=self.current_session_id, message=message)
        tool_calls: List[ToolCall] = []
        iterations = 0
        consecutive_tools = 0
        while iterations < self.config.max_iterations:
            iterations += 1
            if verbose:
                print(f"\n--- Iter {iterations} • session={self.current_session_id[:8]} ---")
            _event("llm_request_begin",
                   temperature=temp,
                   max_tokens=n_tok,
                   stream=self.config.stream and (stream_callback is not None),
                   chat_api=self.config.use_chat_api,
                   stop=self.config.stop)
            gen_start = time.perf_counter()
            text = self.llm.generate(
                convo,
                temperature=temp,
                max_tokens=n_tok,
                stream=self.config.stream and stream_callback is not None,
                on_token=stream_callback,
            )
            gen_dur = time.perf_counter() - gen_start
            _event("llm_response_raw", duration_s=round(gen_dur, 6), sample=text[:240])
            self._log.debug("LLM response (%.3fs) sample=%r", gen_dur, text[:160])
            asst = Message(role=MessageRole.ASSISTANT, content=text)
            convo.append(asst)
            self.db.save_message(self.current_session_id, asst)
            call = self._parse_tool_call_strict(text)
            if not call:
                _event("final_answer", content_preview=text[:240])
                break # final answer reached
            consecutive_tools += 1
            if consecutive_tools > self.config.max_consecutive_tool_calls:
                msg = "[Tool-call limit reached] Provide your final answer succinctly."
                convo.append(Message(role=MessageRole.SYSTEM, content=msg))
                _event("guardrail_stop", reason="max_consecutive_tool_calls")
                self._log.warning("Guardrail stop: max_consecutive_tool_calls")
                break
            tool_calls.append(call)
            if verbose:
                print(f"→ Tool call: {call.name}({call.arguments})")
            _event("parsed_tool_call", name=call.name, arguments=call.arguments)
            self._log.info("Tool call: %s args=%s", call.name, call.arguments)
            exec_start = time.perf_counter()
            result = self.tools.execute(call, use_toon=self.config.use_toon_for_tools)
            exec_dur = time.perf_counter() - exec_start
            if verbose:
                print(f"← Tool result (first 240): {result[:240]}")
            _event("tool_result", name=call.name, duration_s=round(exec_dur, 6), result_preview=result[:240])
            self._log.debug("Tool result (%.3fs) preview=%r", exec_dur, result[:160])
            tool_msg = Message(role=MessageRole.TOOL, name=call.name, tool_call_id=call.id, content=result)
            convo.append(tool_msg)
            self.db.save_message(self.current_session_id, tool_msg)
            final = next((m.content for m in reversed(convo) if m.role == MessageRole.ASSISTANT), "")
            # Optional self-reflection
            if self.config.enable_reflection and iterations < self.config.max_iterations:
                _event("reflection_begin", current_final=final[:100])
                reflect_temp = temp * 0.8
                reflect_tokens = int(n_tok * 0.6)
                reflect_sys = self.config.reflection_prompt.replace("[FINAL]", final)
                reflect_convo = [Message(role=MessageRole.SYSTEM, content=reflect_sys)]
                reflect_convo.extend([m for m in convo[-4:] if m.role != MessageRole.SYSTEM])
                reflect_start = time.perf_counter()
                reflect_text = self.llm.generate(reflect_convo, temperature=reflect_temp, max_tokens=reflect_tokens, stream=False)
                reflect_dur = time.perf_counter() - reflect_start
                _event("reflection_response", duration_s=round(reflect_dur, 6), sample=reflect_text[:200])
                self._log.info("Reflection (%.3fs) sample=%r", reflect_dur, reflect_text[:120])
                reflect_msg = Message(role=MessageRole.ASSISTANT, content=reflect_text)
                convo.append(reflect_msg)
                self.db.save_message(self.current_session_id, reflect_msg)
                reflect_call = self._parse_tool_call_strict(reflect_text)
                if reflect_call:
                    tool_calls.append(reflect_call)
                    consecutive_tools += 1
                    if consecutive_tools > self.config.max_consecutive_tool_calls:
                        _event("guardrail_stop", reason="max_consecutive_tool_calls (in reflection)")
                        self._log.warning("Guardrail stop in reflection")
                    else:
                        exec_start = time.perf_counter()
                        result = self.tools.execute(reflect_call, use_toon=self.config.use_toon_for_tools)
                        exec_dur = time.perf_counter() - exec_start
                        _event("tool_result (reflection)", name=reflect_call.name, duration_s=round(exec_dur, 6), result_preview=result[:100])
                        tool_msg = Message(role=MessageRole.TOOL, name=reflect_call.name, tool_call_id=reflect_call.id, content=result)
                        convo.append(tool_msg)
                        self.db.save_message(self.current_session_id, tool_msg)
                        iterations += 1
                elif "COMPLETE" in reflect_text.upper() or len(reflect_text.strip()) < 20:
                    _event("reflection_complete", reason="explicit or short")
                else:
                    final = reflect_text
                    _event("reflection_refine", preview=final[-100:])
            debug = {
                "session_id": self.current_session_id,
                "iterations": iterations,
                "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls],
                "events": debug_events,
                "config": {
                    "temperature": temp,
                    "max_tokens": n_tok,
                    "use_chat_api": self.config.use_chat_api,
                    "chat_template": self.config.chat_template,
                    "stop": self.config.stop,
                    "stream": self.config.stream,
                    "max_consecutive_tool_calls": self.config.max_consecutive_tool_calls,
                },
                "timings": {
                    "total_duration_s": round(time.perf_counter() - started_ts, 6),
                },
            }
            trace_md = self._render_trace_markdown(debug)
            self._log.info("Run complete iterations=%s tool_calls=%s total=%.3fs",
                           iterations, len(tool_calls), debug["timings"]["total_duration_s"])
            return AgentResponse(
                messages=convo,
                tool_calls=tool_calls,
                iterations=iterations,
                final_response=final,
                debug=debug,
                trace_md=trace_md,
            )

    # ------------- Parsing -------------
    def _parse_tool_call_strict(self, text: str) -> Optional[ToolCall]:
        use_toon = self.config.use_toon_for_tools and HAS_TOON

        if use_toon:
            # Extract TOON block (heuristic for 'tool_call:' start)
            lines = text.splitlines()
            in_block = False
            block_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("tool_call:"):
                    in_block = True
                    block_lines = [line]
                elif in_block and stripped and not stripped.startswith(("  ", "    ")):  # End block
                    break
                elif in_block:
                    block_lines.append(line)
            
            if in_block and block_lines:
                toon_block = "\n".join(block_lines).strip()
                try:
                    data = decode(toon_block)
                    if isinstance(data, dict) and "tool_call" in data:
                        td = data["tool_call"]
                        name = td.get("name")
                        args = td.get("arguments", {})
                        if isinstance(name, str) and isinstance(args, dict):
                            return ToolCall(id=f"call_{time.time():.6f}", name=name, arguments=args)
                except Exception as e:
                    self._log.warning("TOON decode failed: %s", e)

        # Fallback to JSON if TOON fails or disabled
        for candidate in self._extract_json_candidates(text):
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(data, dict) and "tool_call" in data and isinstance(data["tool_call"], dict):
                td = data["tool_call"]
                name = td.get("name")
                args = td.get("arguments", {})
                if isinstance(name, str) and isinstance(args, dict):
                    return ToolCall(id=f"call_{time.time():.6f}", name=name, arguments=args)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                name = data["name"]
                args = data["arguments"]
                if isinstance(name, str) and isinstance(args, dict):
                    return ToolCall(id=f"call_{time.time():.6f}", name=name, arguments=args)
        return None

    def _extract_json_candidates(self, s: str) -> Iterable[str]:
        stack = 0
        start = None
        for i, ch in enumerate(s):
            if ch == "{":
                if stack == 0:
                    start = i
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0 and start is not None:
                    yield s[start: i + 1]
                    start = None

    # ------------- Debug rendering -------------
    def _render_trace_markdown(self, debug: Dict[str, Any]) -> str:
        lines = []
        lines.append(f"### Agent Trace — session `{debug['session_id'][:8]}`")
        lines.append("")
        lines.append(f"- **Iterations**: {debug['iterations']} | **Total**: {debug['timings']['total_duration_s']} s")
        cfg = debug["config"]
        lines.append(f"- **Cfg**: temp={cfg['temperature']} max_tokens={cfg['max_tokens']} chat_api={cfg['use_chat_api']} template=`{cfg['chat_template']}` stream={cfg['stream']}")
        if debug["tool_calls"]:
            first = debug["tool_calls"][0]
            lines.append("")
            lines.append("**Agent command (first tool call):**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps({"tool_call": {"name": first["name"], "arguments": first["arguments"]}}, ensure_ascii=False, indent=2))
            lines.append("```")
        else:
            lines.append("")
            lines.append("_No tool call emitted by model (direct answer)._")
            lines.append("")
        lines.append("**Timeline**")
        lines.append("")
        for ev in debug["events"]:
            t = f"{ev['t']:.3f}s"
            kind = ev["kind"]
            if kind == "user_message":
                lines.append(f"- [{t}] user → “{ev['message'][:80]}”")
            elif kind == "llm_request_begin":
                lines.append(f"- [{t}] llm_request (temp={ev['temperature']}, max_tokens={ev['max_tokens']}, stream={ev['stream']})")
            elif kind == "llm_response_raw":
                sample = ev.get("sample", "")[:80].replace("\n", " ")
                lines.append(f"- [{t}] llm_response ({ev['duration_s']}s) → “{sample}”")
            elif kind == "parsed_tool_call":
                ap = json.dumps(ev["arguments"], ensure_ascii=False)[:80]
                lines.append(f"- [{t}] parsed_tool_call: {ev['name']}({ap}…)")
            elif kind == "tool_result":
                rp = ev.get("result_preview", "")[:80].replace("\n", " ")
                lines.append(f"- [{t}] tool_result {ev['name']} ({ev['duration_s']}s) → “{rp}”")
            elif kind == "final_answer":
                lines.append(f"- [{t}] final_answer → “{ev['content_preview'].replace(chr(10), ' ')[:80]}”")
            elif kind == "guardrail_stop":
                lines.append(f"- [{t}] guardrail_stop: {ev['reason']}")
            elif kind == "rag_retrieval":
                lines.append(f"- [{t}] rag_retrieval: {ev['n_results']} docs (preview: {ev['preview']}…)")
            elif kind == "reflection_begin":
                lines.append(f"- [{t}] reflection_begin (final preview: {ev['current_final']}…)")
            elif kind == "reflection_response":
                sample = ev.get("sample", "")[:80].replace("\n", " ")
                lines.append(f"- [{t}] reflection_response ({ev['duration_s']}s) → “{sample}”")
            elif kind == "reflection_complete":
                lines.append(f"- [{t}] reflection_complete: {ev['reason']}")
            elif kind == "reflection_refine":
                lines.append(f"- [{t}] reflection_refine → “{ev['preview']}”")
        lines.append("")
        return "\n".join(lines)

# =====================================================================
# Convenience creator
# =====================================================================
def create_agent(
    llm_url: str = "http://localhost:8080",
    system_prompt: Optional[str] = None,
    use_chat_api: bool = True,
    chat_template: str = "chatml",
    db_path: str = "agent_sessions.db",
    embedding_model: Optional[str] = "all-MiniLM-L6-v2",
    *,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Agent:
    cfg = Config(
        llama_cpp_url=llm_url,
        use_chat_api=use_chat_api,
        chat_template=chat_template,
        use_toon_for_tools=True,  # Enable TOON by default
    )
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    # Apply logging options immediately
    get_logger("agent.framework", cfg.log_level, cfg.log_json)
    log.info("create_agent: url=%s chat_api=%s template=%s", cfg.llama_cpp_url, cfg.use_chat_api, cfg.chat_template)
    agent = Agent(
        llm_url=cfg.llama_cpp_url,
        system_prompt=system_prompt,
        config=cfg,
        use_chat_api=cfg.use_chat_api,
        chat_template=cfg.chat_template,
        db_path=db_path,
        embedding_model=embedding_model,
    )
    return agent