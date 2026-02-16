
from __future__ import annotations

from typing import Any, Callable, List, Optional
from dataclasses import dataclass, field

from .db import DocumentDB, MessageDB
from .messages import Message, MessageRole
from .config import Config

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
        self._embedding_model_name = self.embedding_model or "BAAI/bge-base-en-v1.5"
        self._db = MessageDB(
            db_path=self.db_path,
            vector_path=self.config.vector_path,
            embedding_model_name=self._embedding_model_name,
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
        self._doc_db = DocumentDB(embedding_model_name=self._embedding_model_name)

    def load_history(
        self,
        sid: str,
        message: str,
        use_semantic_retrieval: bool,
        retrieval_mode: str = "vector",
    ) -> List[Message]:
        """Loads conversation history with optional semantic search."""
        history_recent_tail = int(getattr(self.config, "history_recent_tail", 8))
        return self._db.load_history(
            sid,
            limit=getattr(self.config, "history_limit", 18),
            summarize_old=True,
            use_semantic=use_semantic_retrieval,
            semantic_query=message if use_semantic_retrieval else None,
            retrieval_mode=retrieval_mode,
            recent_tail=history_recent_tail,
        )

    def get_rag_context(self, message: str, event_cb: Callable[..., None]) -> Optional[str]:
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
