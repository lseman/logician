from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..config import Config
from ..db import DocumentDB, MessageDB
from ..messages import Message, MessageRole
from ..runtime_paths import session_db_path
from .palace import MemoryPalace

_DEFAULT_EMBEDDING_MODEL_NAME = (
    "BAAI/bge-m3|Snowflake/snowflake-arctic-embed-l-v2.0|"
    "Qwen/Qwen3-Embedding-0.6B|nomic-ai/nomic-embed-text-v1.5|"
    "intfloat/e5-mistral-7b-instruct|BAAI/bge-small-en-v1.5"
)


@dataclass
class Memory:
    """
    Manages short-term (conversation history) and long-term (vector/rag) memory.
    """

    config: Config
    db_path: str = field(default_factory=lambda: str(session_db_path()))
    embedding_model: Optional[str] = None
    lazy_rag: bool = True

    _db: Optional[MessageDB] = field(default=None, init=False)
    _doc_db: Optional[DocumentDB] = field(default=None, init=False)
    _palace: Optional[MemoryPalace] = field(default=None, init=False)

    def __post_init__(self):
        self._embedding_model_name = str(self.embedding_model or "").strip()
        message_vectors_enabled = bool(
            getattr(self.config, "message_history_vector_enabled", False)
            and self._embedding_model_name
        )
        self._db = MessageDB(
            db_path=self.db_path,
            vector_path=self.config.vector_path,
            embedding_model_name=self._embedding_model_name or _DEFAULT_EMBEDDING_MODEL_NAME,
            vector_enabled=message_vectors_enabled,
            vector_backend=str(getattr(self.config, "vector_backend", "")),
            index_on_write=bool(getattr(self.config, "message_vector_index_on_write", False)),
            rerank_enabled=False,
        )
        if not self.lazy_rag and self.config.rag_enabled:
            self._ensure_doc_db()

    def _ensure_doc_db(self) -> None:
        if self._doc_db is not None:
            return
        if not self.config.rag_enabled:
            return

        rag_backend = str(getattr(self.config, "rag_vector_backend", "") or "").strip()
        if not rag_backend:
            return

        self._doc_db = DocumentDB(
            vector_path=self.config.rag_vector_path,
            embedding_model_name=self._embedding_model_name or _DEFAULT_EMBEDDING_MODEL_NAME,
            vector_backend=rag_backend,
            rerank_enabled=bool(getattr(self.config, "rag_rerank_enabled", True)),
            rerank_fetch_k=int(getattr(self.config, "rag_rerank_fetch_k", 30)),
            min_similarity=float(getattr(self.config, "rag_min_similarity", 0.20)),
            hnsw_ef_search=int(getattr(self.config, "rag_hnsw_ef_search", 128)),
            hnsw_m=int(getattr(self.config, "rag_hnsw_m", 16)),
            hnsw_ef_construction=int(getattr(self.config, "rag_hnsw_ef_construction", 200)),
            per_source_max_chunks=int(getattr(self.config, "rag_per_source_max_chunks", 4)),
            query_cache_enabled=bool(getattr(self.config, "rag_query_cache_enabled", True)),
            query_cache_ttl_sec=int(getattr(self.config, "rag_query_cache_ttl_sec", 90)),
            query_cache_max_entries=int(getattr(self.config, "rag_query_cache_max_entries", 256)),
        )

    def _ensure_palace(self) -> None:
        if self._palace is not None:
            return
        if not getattr(self.config, "memory_palace_enabled", False):
            return
        try:
            self._palace = MemoryPalace(config=self.config)
        except Exception:
            self._palace = None

    def load_history(
        self,
        sid: str,
        message: str,
        use_semantic_retrieval: bool,
        retrieval_mode: str = "vector",
    ) -> List[Message]:
        """Loads conversation history with optional semantic search."""
        palace_context = self._build_palace_context(sid, message)
        history_limit = int(self.config.history_limit)
        db_limit = max(1, history_limit - (1 if palace_context else 0))
        history_recent_tail = int(self.config.history_recent_tail)
        history = self._db.load_history(
            sid,
            limit=db_limit,
            summarize_old=True,
            use_semantic=use_semantic_retrieval and bool(self._db and self._db.vector_enabled),
            semantic_query=message if use_semantic_retrieval else None,
            retrieval_mode=retrieval_mode,
            recent_tail=history_recent_tail,
        )
        if palace_context:
            return [Message(role=MessageRole.SYSTEM, content=palace_context), *history]
        return history

    def get_rag_context(
        self,
        message: str,
        event_cb: Callable[..., None],
        *,
        where: dict[str, Any] | None = None,
        n_results: int | None = None,
    ) -> Optional[str]:
        """Retrieves relevant RAG context for the message."""
        if not self.config.rag_enabled:
            return None

        rows = self.search_rag(
            message,
            where=where,
            n_results=n_results,
            event_cb=event_cb,
        )
        if not rows:
            return None

        rag_context = "\n\n".join(
            f"[Doc: {r['metadata'].get('source', 'unknown')}] {r['content']}" for r in rows
        )
        return rag_context

    def search_rag(
        self,
        message: str,
        *,
        where: dict[str, Any] | None = None,
        n_results: int | None = None,
        ef_search: int | None = None,
        event_cb: Callable[..., None] | None = None,
    ) -> list[dict[str, Any]]:
        """Return raw RAG rows with optional metadata filtering."""
        if not self.config.rag_enabled:
            return []

        if self.lazy_rag:
            self._ensure_doc_db()

        if self._doc_db is None:
            return []

        rag_results = self._doc_db.query(
            message,
            n_results=int(n_results or self.config.rag_top_k),
            where=where,
            ef_search=ef_search,
        )
        if not rag_results:
            return []

        if event_cb is not None:
            try:
                preview = "\n\n".join(
                    f"[Doc: {r['metadata'].get('source', 'unknown')}] {r['content']}"
                    for r in rag_results[:2]
                )
                event_cb(
                    "rag_retrieval",
                    n_results=len(rag_results),
                    preview=preview[:200],
                )
            except Exception:
                pass
        return rag_results

    def save_message(self, sid: str, message: Message) -> None:
        """Persists a message to the database and writes it into the Palace."""
        self._db.save_message(sid, message)
        if getattr(self.config, "memory_palace_enabled", False):
            try:
                self._ensure_palace()
                if self._palace is not None:
                    self._palace.save_message(sid, message)
            except Exception:
                pass

    def search_memory_palace(
        self,
        query: str,
        *,
        n_results: int = 8,
        session_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search raw Palace memory for conversation relevance."""
        if not getattr(self.config, "memory_palace_enabled", False):
            return []
        self._ensure_palace()
        if self._palace is None:
            return []
        return self._palace.search(query, n_results=n_results, session_filter=session_filter)

    def _build_palace_context(self, sid: str, query: str) -> str:
        if not getattr(self.config, "memory_palace_enabled", False):
            return ""
        if not getattr(self.config, "memory_palace_context_enabled", True):
            return ""
        query_text = str(query or "").strip()
        if not query_text:
            return ""
        self._ensure_palace()
        if self._palace is None:
            return ""

        max_results = int(getattr(self.config, "memory_palace_context_max_results", 4))
        max_chars = int(getattr(self.config, "memory_palace_context_max_chars", 1600))
        rows = self._palace.search(query_text, n_results=max(1, max_results * 3))
        filtered = [
            row
            for row in rows
            if str(row.get("session_id") or "") != str(sid)
            and str(row.get("role") or "") in {"system", "user", "assistant"}
            and str(row.get("content") or "").strip()
        ][:max_results]
        if not filtered:
            return ""

        lines = [
            "## Memory Palace Context",
            "Relevant raw messages from previous sessions:",
        ]
        used = len("\n".join(lines))
        for row in filtered:
            session_id = str(row.get("session_id") or "")
            role = str(row.get("role") or "")
            content = str(row.get("content") or "").strip()
            snippet = " ".join(content.split())
            entry = f"- [{session_id}] {role}: {snippet}"
            if used + len(entry) + 1 > max_chars:
                break
            lines.append(entry)
            used += len(entry) + 1
        if len(lines) <= 2:
            return ""
        return "\n".join(lines)

    def get_session_messages(self, sid: str) -> List[Message]:
        """Loads the full persisted transcript for a session."""
        return self._db.get_session_messages(sid)

    def count_session_messages(self, sid: str) -> int:
        """Returns the total persisted message count for a session."""
        return self._db.count_session_messages(sid)

    def save_runtime_state(self, sid: str, state: dict[str, Any] | None) -> None:
        """Persists runtime tool/data state for a session."""
        self._db.save_session_runtime_state(sid, state)

    def load_runtime_state(self, sid: str) -> dict[str, Any] | None:
        """Loads persisted runtime tool/data state for a session."""
        return self._db.load_session_runtime_state(sid)

    def clear_session_messages(self, sid: str) -> None:
        """Clears only the persisted transcript for a session."""
        self._db.clear_session_messages(sid)

    def clear_runtime_state(self, sid: str) -> None:
        """Clears only the persisted runtime state for a session."""
        self._db.clear_session_runtime_state(sid)

    def clear_session(self, sid: str) -> None:
        """Clears a session from memory."""
        self._db.clear_session(sid)
        if getattr(self.config, "memory_palace_enabled", False):
            try:
                self._ensure_palace()
                if self._palace is not None:
                    self._palace.clear_session(sid)
            except Exception:
                pass

    def list_sessions(self) -> List[tuple[str, str]]:
        """Lists all stored sessions."""
        return self._db.list_sessions()

    def semantic_search(
        self,
        query: str,
        session_id: str,
        k: int = 8,
        retrieval_mode: str = "vector",
    ) -> List[Message]:
        if not bool(self._db and self._db.vector_enabled):
            return []
        return self._db.global_semantic_search(
            query, k, session_filter=session_id, retrieval_mode=retrieval_mode
        )
