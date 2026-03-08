# agent_core/db_message.py
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from .core import _HNSWCollection, _SQLITE_PRAGMAS
from ..logging_utils import get_logger
from ..messages import Message, MessageRole


@dataclass
class MessageDB:
    db_path: str = "agent_sessions.db"
    vector_path: str = "message_history.vector"
    embedding_model_name: str = (
        "google/embeddinggemma-300m|intfloat/e5-small-v2|BAAI/bge-small-en-v1.5"
    )

    vector_enabled: bool = True
    lazy_vector: bool = True
    vector_collection_name: str = "messages"
    rerank_enabled: bool = True
    reranker_model_name: str = (
        "jinaai/jina-reranker-v2-base-multilingual|BAAI/bge-reranker-v2.5-gemma2-lightweight"
    )
    min_similarity: float = 0.18
    hybrid_rrf_k: int = 60
    hybrid_vector_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3

    _log: Any = field(init=False, repr=False)
    _conn: sqlite3.Connection | None = field(default=None, init=False, repr=False)
    _fts_enabled: bool = field(default=False, init=False, repr=False)

    _collection: _HNSWCollection | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._log = get_logger("agent.db")
        self._init_sqlite()
        if self.vector_enabled and (not self.lazy_vector):
            self._ensure_vector()

    def _init_sqlite(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        for p in _SQLITE_PRAGMAS:
            self._conn.execute(p)

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tool_call_id TEXT,
                name TEXT,
                vector_id TEXT UNIQUE
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_runtime_state (
                session_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        try:
            self._conn.execute("ALTER TABLE messages ADD COLUMN vector_id TEXT;")
        except Exception:
            pass
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(timestamp);"
        )
        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    session_id,
                    role,
                    content='messages',
                    content_rowid='id',
                    tokenize='unicode61 porter'
                );
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content, session_id, role)
                    VALUES (new.id, new.content, new.session_id, new.role);
                END;
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, session_id, role)
                    VALUES ('delete', old.id, old.content, old.session_id, old.role);
                END;
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, session_id, role)
                    VALUES ('delete', old.id, old.content, old.session_id, old.role);
                    INSERT INTO messages_fts(rowid, content, session_id, role)
                    VALUES (new.id, new.content, new.session_id, new.role);
                END;
                """
            )
            self._conn.execute(
                "INSERT INTO messages_fts(messages_fts) VALUES('rebuild');"
            )
            self._fts_enabled = True
        except Exception as e:
            self._log.warning("FTS5 unavailable, keyword/hybrid search disabled: %s", e)
            self._fts_enabled = False
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
            self._conn = None

    def _ensure_vector(self) -> None:
        if not self.vector_enabled:
            return
        if self._collection is not None:
            return
        try:
            self._collection = _HNSWCollection(
                root_path=self.vector_path,
                collection_name=self.vector_collection_name,
                embedding_model_name=self.embedding_model_name,
                rerank_enabled=self.rerank_enabled,
                reranker_model_name=self.reranker_model_name,
                min_similarity=self.min_similarity,
            )
        except ImportError as e:
            self.vector_enabled = False
            self._collection = None
            self._log.warning("Disabling vector memory: %s", e)

    @staticmethod
    def _hybrid_doc_key(row: dict[str, Any]) -> str:
        row_id = row.get("row_id")
        if row_id is not None:
            return f"row:{row_id}"
        return "|".join(
            [
                str(row.get("role", "")),
                str(row.get("name", "")),
                str(row.get("tool_call_id", "")),
                str(row.get("content", "")),
            ]
        )

    @staticmethod
    def _fts_query(query: str) -> str:
        toks = re.findall(r"[A-Za-z0-9_]+", query)
        return " ".join(toks)

    def _keyword_search_rows(
        self,
        query: str,
        k: int,
        *,
        session_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._conn is None or not self._fts_enabled:
            return []
        q = self._fts_query(query)
        if not q:
            return []

        if session_filter:
            rows = self._conn.execute(
                """
                SELECT m.id, m.role, m.content, m.name, m.tool_call_id, m.vector_id,
                       bm25(messages_fts) AS kw_score
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                WHERE messages_fts MATCH ? AND m.session_id = ?
                ORDER BY kw_score ASC
                LIMIT ?
                """,
                (q, session_filter, k),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT m.id, m.role, m.content, m.name, m.tool_call_id, m.vector_id,
                       bm25(messages_fts) AS kw_score
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                WHERE messages_fts MATCH ?
                ORDER BY kw_score ASC
                LIMIT ?
                """,
                (q, k),
            ).fetchall()

        return [
            {
                "row_id": int(mid),
                "role": str(role),
                "content": str(content),
                "name": name,
                "tool_call_id": tool_call_id,
                "vector_id": vector_id,
                "kw_score": float(score),
            }
            for (mid, role, content, name, tool_call_id, vector_id, score) in rows
        ]

    def _vector_search_rows(
        self,
        query: str,
        k: int,
        *,
        session_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._conn is None or (not self.vector_enabled):
            return []
        where: dict[str, Any] = {"session_id": session_filter} if session_filter else {}
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
            include=["metadatas", "documents", "distances", "ids"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        rows: list[dict[str, Any]] = []
        for doc, meta, dist, vector_id in zip(docs, metas, dists, ids):
            rows.append(
                {
                    "row_id": None,
                    "role": str(meta["role"]),
                    "content": str(doc),
                    "name": (meta.get("name") or None),
                    "tool_call_id": (meta.get("tool_call_id") or None),
                    "vector_id": str(vector_id),
                    "distance": float(dist),
                }
            )

        vector_ids = [r["vector_id"] for r in rows if r.get("vector_id")]
        if not vector_ids:
            return rows

        placeholders = ",".join("?" for _ in vector_ids)
        db_rows = self._conn.execute(
            f"""
            SELECT id, vector_id, role, content, name, tool_call_id
            FROM messages
            WHERE vector_id IN ({placeholders})
            """,
            tuple(vector_ids),
        ).fetchall()
        by_vector_id: dict[str, tuple[int, str, str, str | None, str | None]] = {
            str(vector_id): (int(row_id), str(role), str(content), name, tool_call_id)
            for (row_id, vector_id, role, content, name, tool_call_id) in db_rows
        }
        for row in rows:
            v_id = str(row.get("vector_id", ""))
            mapped = by_vector_id.get(v_id)
            if mapped is None:
                continue
            row["row_id"] = mapped[0]
            row["role"] = mapped[1]
            row["content"] = mapped[2]
            row["name"] = mapped[3]
            row["tool_call_id"] = mapped[4]
        return rows

    def _hybrid_search_rows(
        self,
        query: str,
        k: int,
        *,
        session_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        keyword_rows = self._keyword_search_rows(query, k, session_filter=session_filter)
        vector_rows = self._vector_search_rows(query, k, session_filter=session_filter)

        if not keyword_rows:
            return vector_rows[:k]
        if not vector_rows:
            return keyword_rows[:k]

        scores: dict[str, float] = {}
        rows_by_key: dict[str, dict[str, Any]] = {}
        kw_w = float(max(0.0, self.hybrid_keyword_weight))
        vec_w = float(max(0.0, self.hybrid_vector_weight))

        for rank, row in enumerate(keyword_rows, start=1):
            key = self._hybrid_doc_key(row)
            rows_by_key[key] = row
            scores[key] = scores.get(key, 0.0) + kw_w / float(self.hybrid_rrf_k + rank)

        for rank, row in enumerate(vector_rows, start=1):
            key = self._hybrid_doc_key(row)
            rows_by_key[key] = row
            scores[key] = scores.get(key, 0.0) + vec_w / float(self.hybrid_rrf_k + rank)

        ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
        return [rows_by_key[key] for key in ranked_keys[:k]]

    def _search_messages(
        self,
        query: str,
        k: int,
        *,
        session_filter: str | None = None,
        mode: str = "vector",
    ) -> list[dict[str, Any]]:
        mode_norm = str(mode or "vector").strip().lower()
        if mode_norm == "keyword":
            return self._keyword_search_rows(query, k, session_filter=session_filter)
        if mode_norm == "hybrid":
            return self._hybrid_search_rows(query, k, session_filter=session_filter)
        return self._vector_search_rows(query, k, session_filter=session_filter)

    @property
    def collection(self) -> _HNSWCollection:
        self._ensure_vector()
        if self._collection is None:
            raise RuntimeError(
                "Vector store is disabled but semantic operation was requested."
            )
        return self._collection

    def save_message(self, session_id: str, msg: Message) -> int:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        vectorize = bool(getattr(msg, "vectorize", True))
        vector_id = (
            str(uuid.uuid4()) if (self.vector_enabled and vectorize) else None
        )

        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO messages (session_id, role, content, tool_call_id, name, vector_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    msg.role.value,
                    msg.content,
                    msg.tool_call_id,
                    msg.name,
                    vector_id,
                ),
            )
            rowid = int(cur.lastrowid)
            self._conn.commit()

        if self.vector_enabled and vectorize and vector_id is not None:
            try:
                self._ensure_vector()
                if not self.vector_enabled or self._collection is None:
                    return rowid
                meta = {
                    "session_id": session_id,
                    "role": msg.role.value,
                    "name": msg.name or "",
                    "tool_call_id": msg.tool_call_id or "",
                }
                self.collection.add(
                    documents=[msg.content],
                    metadatas=[meta],
                    ids=[vector_id],
                )
            except Exception as e:
                self._log.exception("Vector add failed (rowid=%s): %s", rowid, e)

        return rowid

    def load_history(
        self,
        session_id: str,
        limit: int = 20,
        summarize_old: bool = True,
        use_semantic: bool = False,
        semantic_query: str | None = None,
        retrieval_mode: str = "vector",
        recent_tail: int = 8,
    ) -> list[Message]:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        def to_messages(rows: list[tuple[Any, ...]]) -> list[Message]:
            return [
                Message(
                    role=MessageRole(role),
                    content=content,
                    name=name,
                    tool_call_id=tool_call_id,
                )
                for (_, role, content, name, tool_call_id) in rows
            ]

        def summary_checkpoint_row() -> tuple[Any, ...] | None:
            return self._conn.execute(
                """
                SELECT id, role, content, name, tool_call_id
                FROM messages
                WHERE session_id=? AND role=? AND content LIKE ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (
                    session_id,
                    MessageRole.SYSTEM.value,
                    "[Session summary inserted by /compact]%",
                ),
            ).fetchone()

        recent_tail = max(0, int(recent_tail))
        limit = max(1, int(limit))

        if use_semantic and semantic_query:
            mode_norm = str(retrieval_mode or "vector").strip().lower()
            if mode_norm in ("vector", "hybrid") and (not self.vector_enabled):
                use_semantic = False
            elif mode_norm in ("keyword", "hybrid") and (not self._fts_enabled):
                if mode_norm == "keyword":
                    use_semantic = False
                else:
                    mode_norm = "vector"
            else:
                # Always include the latest turns, then fill remaining budget with semantic hits.
                recent_id_rows = self._conn.execute(
                    """
                    SELECT id
                    FROM messages
                    WHERE session_id=?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, min(limit, recent_tail)),
                ).fetchall()
                recent_ids = [int(r[0]) for r in recent_id_rows]
                recent_id_set = set(recent_ids)

                rows = self._search_messages(
                    semantic_query,
                    k=max(limit * 2, limit + recent_tail),
                    session_filter=session_id,
                    mode=mode_norm,
                )
                semantic_ids = [
                    int(r["row_id"]) for r in rows if r.get("row_id") is not None
                ]
                chosen_ids: list[int] = list(reversed(recent_ids))
                for msg_id in semantic_ids:
                    if msg_id not in recent_id_set and msg_id not in chosen_ids:
                        chosen_ids.append(msg_id)
                    if len(chosen_ids) >= limit:
                        break

                if not chosen_ids:
                    use_semantic = False
                else:
                    placeholders = ",".join("?" for _ in chosen_ids)
                    selected_rows = self._conn.execute(
                        f"""
                        SELECT id, role, content, name, tool_call_id
                        FROM messages
                        WHERE id IN ({placeholders})
                        ORDER BY id ASC
                        """,
                        tuple(chosen_ids),
                    ).fetchall()
                    if summarize_old and limit > 1:
                        summary_row = summary_checkpoint_row()
                        if summary_row is not None:
                            summary_id = int(summary_row[0])
                            selected_ids = {int(row[0]) for row in selected_rows}
                            if summary_id not in selected_ids:
                                selected_rows = [summary_row, *selected_rows[-(limit - 1) :]]
                    return to_messages(selected_rows)

        rows = self._conn.execute(
            """
            SELECT id, role, content, name, tool_call_id
            FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        rows = list(reversed(rows))
        if summarize_old and limit > 1:
            summary_row = summary_checkpoint_row()
            if summary_row is not None:
                summary_id = int(summary_row[0])
                row_ids = {int(row[0]) for row in rows}
                if summary_id not in row_ids:
                    rows = [summary_row, *rows[-(limit - 1) :]]
        return to_messages(rows)

    def get_session_messages(self, session_id: str) -> list[Message]:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        rows = self._conn.execute(
            """
            SELECT role, content, name, tool_call_id
            FROM messages
            WHERE session_id=?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            Message(
                role=MessageRole(role),
                content=content,
                name=name,
                tool_call_id=tool_call_id,
            )
            for (role, content, name, tool_call_id) in rows
        ]

    def count_session_messages(self, session_id: str) -> int:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        row = self._conn.execute(
            """
            SELECT COUNT(*)
            FROM messages
            WHERE session_id=?
            """,
            (session_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def save_session_runtime_state(
        self, session_id: str, state: dict[str, Any] | None
    ) -> None:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        payload = json.dumps(state or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO session_runtime_state (session_id, state_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (session_id, payload),
            )
            self._conn.commit()

    def load_session_runtime_state(self, session_id: str) -> dict[str, Any] | None:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        row = self._conn.execute(
            """
            SELECT state_json
            FROM session_runtime_state
            WHERE session_id=?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(str(row[0]))
        except Exception:
            self._log.exception(
                "Failed to decode runtime state for session=%s", session_id[:8]
            )
            return None

    def clear_session_messages(self, session_id: str) -> None:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        self._log.info("Clearing session messages %s", session_id[:8])
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self._conn.commit()

        if self.vector_enabled:
            try:
                self.collection.delete(where={"session_id": session_id})
            except Exception as e:
                self._log.warning(
                    "Vector delete failed for session=%s: %s", session_id[:8], e
                )

    def clear_session_runtime_state(self, session_id: str) -> None:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        with self._lock:
            self._conn.execute(
                "DELETE FROM session_runtime_state WHERE session_id=?",
                (session_id,),
            )
            self._conn.commit()

    def clear_session(self, session_id: str) -> None:
        self.clear_session_messages(session_id)
        self.clear_session_runtime_state(session_id)

    def list_sessions(self) -> list[tuple[str, str]]:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        rows = self._conn.execute(
            """
            SELECT session_id, MAX(timestamp) FROM messages
            GROUP BY session_id
            ORDER BY MAX(timestamp) DESC
            """
        ).fetchall()
        return [(str(session_id), str(last_ts)) for (session_id, last_ts) in rows]

    def global_semantic_search(
        self,
        query: str,
        k: int = 8,
        session_filter: str | None = None,
        retrieval_mode: str = "vector",
    ) -> list[Message]:
        mode_norm = str(retrieval_mode or "vector").strip().lower()
        if mode_norm in ("vector", "hybrid") and (not self.vector_enabled):
            return []
        if mode_norm in ("keyword", "hybrid") and (not self._fts_enabled):
            if mode_norm == "keyword":
                return []
            mode_norm = "vector"

        rows = self._search_messages(
            query, k=k, session_filter=session_filter, mode=mode_norm
        )
        return [
            Message(
                role=MessageRole(str(r["role"])),
                content=str(r["content"]),
                name=(str(r["name"]) if r.get("name") else None),
                tool_call_id=(
                    str(r["tool_call_id"]) if r.get("tool_call_id") else None
                ),
            )
            for r in rows
        ]
