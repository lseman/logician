from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .config import Config
from .db import DocumentDB, MessageDB
from .messages import Message


@dataclass
class Memory:
    """
    Manages short-term (conversation history) and long-term (vector/rag) memory.
    """

    config: Config
    db_path: str = "agent_sessions.db"
    embedding_model: Optional[str] = None
    lazy_rag: bool = True

    _db: Optional[MessageDB] = field(default=None, init=False)
    _doc_db: Optional[DocumentDB] = field(default=None, init=False)

    def __post_init__(self):
        self._embedding_model_name = self.embedding_model or (
            "BAAI/bge-m3|Snowflake/snowflake-arctic-embed-l-v2.0|"
            "Qwen/Qwen3-Embedding-0.6B|nomic-ai/nomic-embed-text-v1.5|"
            "intfloat/e5-mistral-7b-instruct|BAAI/bge-small-en-v1.5"
        )
        self._db = MessageDB(
            db_path=self.db_path,
            vector_path=self.config.vector_path,
            embedding_model_name=self._embedding_model_name,
            vector_backend=str(getattr(self.config, "vector_backend", "usearch")),
            # Reranking on short chat turns is expensive and yields no benefit;
            # cross-encoder reranking is reserved for document RAG (DocumentDB).
            rerank_enabled=False,
        )
        if not self.lazy_rag and self.config.rag_enabled and self._embedding_model_name:
            self._ensure_doc_db()

    def _ensure_doc_db(self) -> None:
        if self._doc_db is not None:
            return
        if not self.config.rag_enabled:
            return
        if not self._embedding_model_name:
            return
        self._doc_db = DocumentDB(
            vector_path=self.config.rag_vector_path,
            embedding_model_name=self._embedding_model_name,
            vector_backend=str(getattr(self.config, "rag_vector_backend", "usearch")),
            rerank_enabled=bool(getattr(self.config, "rag_rerank_enabled", True)),
            rerank_fetch_k=int(getattr(self.config, "rag_rerank_fetch_k", 30)),
            min_similarity=float(getattr(self.config, "rag_min_similarity", 0.20)),
            hnsw_ef_search=int(getattr(self.config, "rag_hnsw_ef_search", 128)),
            hnsw_m=int(getattr(self.config, "rag_hnsw_m", 16)),
            hnsw_ef_construction=int(
                getattr(self.config, "rag_hnsw_ef_construction", 200)
            ),
            per_source_max_chunks=int(
                getattr(self.config, "rag_per_source_max_chunks", 4)
            ),
            query_cache_enabled=bool(
                getattr(self.config, "rag_query_cache_enabled", True)
            ),
            query_cache_ttl_sec=int(getattr(self.config, "rag_query_cache_ttl_sec", 90)),
            query_cache_max_entries=int(
                getattr(self.config, "rag_query_cache_max_entries", 256)
            ),
        )

    def load_history(
        self,
        sid: str,
        message: str,
        use_semantic_retrieval: bool,
        retrieval_mode: str = "vector",
    ) -> List[Message]:
        """Loads conversation history with optional semantic search."""
        history_recent_tail = int(self.config.history_recent_tail)
        return self._db.load_history(
            sid,
            limit=self.config.history_limit,
            summarize_old=True,
            use_semantic=use_semantic_retrieval,
            semantic_query=message if use_semantic_retrieval else None,
            retrieval_mode=retrieval_mode,
            recent_tail=history_recent_tail,
        )

    def get_rag_context(
        self, message: str, event_cb: Callable[..., None]
    ) -> Optional[str]:
        """Retrieves relevant RAG context for the message."""
        if not self.config.rag_enabled:
            return None

        if self.lazy_rag:
            self._ensure_doc_db()

        if self._doc_db is None:
            return None

        rag_results = self._doc_db.query(message, n_results=self.config.rag_top_k)
        if not rag_results:
            return None

        rag_context = "\n\n".join(
            f"[Doc: {r['metadata'].get('source', 'unknown')}] {r['content']}"
            for r in rag_results
        )

        event_cb(
            "rag_retrieval",
            n_results=len(rag_results),
            preview=rag_context[:200],
        )
        return rag_context

    def save_message(self, sid: str, message: Message) -> None:
        """Persists a message to the database."""
        self._db.save_message(sid, message)

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
        return self._db.global_semantic_search(
            query, k, session_filter=session_id, retrieval_mode=retrieval_mode
        )
