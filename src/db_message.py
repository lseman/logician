# agent_core/db_message.py
from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from .db_core import _HNSWCollection, _SQLITE_PRAGMAS
from .logging_utils import get_logger
from .messages import Message, MessageRole


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
        self._collection = _HNSWCollection(
            root_path=self.vector_path,
            collection_name=self.vector_collection_name,
            embedding_model_name=self.embedding_model_name,
            rerank_enabled=self.rerank_enabled,
            reranker_model_name=self.reranker_model_name,
            min_similarity=self.min_similarity,
        )

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
        sql_rows = self._conn.execute(
            f"""
            SELECT id, vector_id, role, content, name, tool_call_id
            FROM messages
            WHERE vector_id IN ({placeholders})
            """,
            tuple(vector_ids),
        ).fetchall()
        by_vector_id: dict[str, tuple[int, str, str, str | None, str | None]] = {
            str(v_id): (int(mid), str(role), str(content), name, tool_id)
            for (mid, v_id, role, content, name, tool_id) in sql_rows
        }

        for row in rows:
            v_id = str(row.get("vector_id", ""))
            mapped = by_vector_id.get(v_id)
            if mapped is None:
                continue
            mid, role, content, name, tool_id = mapped
            row["row_id"] = mid
            row["role"] = role
            row["content"] = content
            row["name"] = name
            row["tool_call_id"] = tool_id
        return rows

    def _hybrid_search_rows(
        self,
        query: str,
        k: int,
        *,
        session_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        if k <= 0:
            return []

        fetch_k = max(k, k * 3)
        vector_rows = self._vector_search_rows(
            query, fetch_k, session_filter=session_filter
        )
        keyword_rows = self._keyword_search_rows(
            query, fetch_k, session_filter=session_filter
        )

        rrf_k = max(1, int(self.hybrid_rrf_k))
        vec_w = float(max(0.0, self.hybrid_vector_weight))
        kw_w = float(max(0.0, self.hybrid_keyword_weight))

        fused_scores: dict[str, float] = {}
        fused_rows: dict[str, dict[str, Any]] = {}

        for rank, row in enumerate(vector_rows, start=1):
            key = self._hybrid_doc_key(row)
            fused_scores[key] = fused_scores.get(key, 0.0) + (vec_w / (rrf_k + rank))
            fused_rows[key] = row

        for rank, row in enumerate(keyword_rows, start=1):
            key = self._hybrid_doc_key(row)
            fused_scores[key] = fused_scores.get(key, 0.0) + (kw_w / (rrf_k + rank))
            if key not in fused_rows:
                fused_rows[key] = row

        ranked_keys = sorted(
            fused_scores.keys(), key=lambda kk: fused_scores[kk], reverse=True
        )
        out: list[dict[str, Any]] = []
        for key in ranked_keys[:k]:
            row = dict(fused_rows[key])
            row["hybrid_score"] = float(fused_scores[key])
            out.append(row)
        return out

    @staticmethod
    def _rows_to_messages(rows: list[dict[str, Any]]) -> list[Message]:
        out: list[Message] = []
        for row in rows:
            try:
                role = MessageRole(str(row.get("role")))
            except Exception:
                continue
            out.append(
                Message(
                    role=role,
                    content=str(row.get("content", "")),
                    name=(row.get("name") or None),
                    tool_call_id=(row.get("tool_call_id") or None),
                )
            )
        return out

    def _search_messages(
        self,
        query: str,
        *,
        k: int,
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

        vector_id = str(uuid.uuid4()) if self.vector_enabled else None

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

        if self.vector_enabled and vector_id is not None:
            try:
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
                semantic_ids: list[int] = []
                for row in rows:
                    rid = row.get("row_id")
                    if rid is None:
                        continue
                    rid_i = int(rid)
                    if rid_i in recent_id_set:
                        continue
                    semantic_ids.append(rid_i)

                budget_left = max(0, limit - len(recent_ids))
                selected_ids = recent_ids + semantic_ids[:budget_left]

                if selected_ids:
                    placeholders = ",".join("?" for _ in selected_ids)
                    selected_rows = self._conn.execute(
                        f"""
                        SELECT role, content, name, tool_call_id
                        FROM messages
                        WHERE id IN ({placeholders})
                        ORDER BY id ASC
                        """,
                        tuple(selected_ids),
                    ).fetchall()
                    msgs = [
                        Message(role=MessageRole(r), content=c, name=n, tool_call_id=t)
                        for (r, c, n, t) in selected_rows
                    ]
                else:
                    msgs = []

                if not msgs:
                    # Fallback if semantic rows could not be mapped to SQL row IDs.
                    msgs = self._rows_to_messages(rows[:limit])
                return self._maybe_summarize(msgs, summarize_old)

        rows = self._conn.execute(
            """
            SELECT role, content, name, tool_call_id
            FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

        msgs = [
            Message(role=MessageRole(r), content=c, name=n, tool_call_id=t)
            for (r, c, n, t) in reversed(rows)
        ]
        return self._maybe_summarize(msgs, summarize_old)

    def _maybe_summarize(
        self, msgs: list[Message], summarize_old: bool
    ) -> list[Message]:
        if not summarize_old:
            return msgs
        if len(msgs) <= 14:
            return msgs

        cut = len(msgs) // 2
        old = msgs[:cut]
        keybits = "; ".join(m.content[:48].replace("\n", " ") for m in old[-4:])
        summary = Message(
            role=MessageRole.SYSTEM,
            content=f"[Summary of {len(old)} prior turns] Key points: {keybits} ...",
        )
        return [summary] + msgs[cut:]

    def clear_session(self, session_id: str) -> None:
        if self._conn is None:
            raise RuntimeError("SQLite connection is not initialized.")

        self._log.info("Clearing session %s", session_id[:8])

        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self._conn.commit()

        if self.vector_enabled:
            try:
                self.collection.delete(where={"session_id": session_id})
            except Exception as e:
                self._log.exception(
                    "Vector delete failed for session=%s: %s", session_id[:8], e
                )

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
        return [(str(sid), str(ts)) for (sid, ts) in rows]

    def global_semantic_search(
        self,
        query: str,
        k: int = 20,
        session_filter: str | None = None,
        retrieval_mode: str = "vector",
    ) -> list[Message]:
        mode_norm = str(retrieval_mode or "vector").strip().lower()
        if mode_norm in ("vector", "hybrid") and (not self.vector_enabled):
            return []
        if mode_norm == "keyword" and (not self._fts_enabled):
            return []
        if mode_norm == "hybrid" and (not self._fts_enabled):
            mode_norm = "vector"
        rows = self._search_messages(
            query, k=k, session_filter=session_filter, mode=mode_norm
        )
        return self._rows_to_messages(rows)
