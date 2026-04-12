from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:  # pragma: no cover
    _rf_fuzz = None

from ..aaak_dialect import compress_text_to_aaak
from ..config import Config
from ..messages import Message


def _rapidfuzz_score(left: str, right: str) -> float:
    if _rf_fuzz is None or not left or not right:
        return 0.0

    try:
        if hasattr(_rf_fuzz, "WRatio"):
            return float(_rf_fuzz.WRatio(left, right)) / 100.0
    except Exception:
        pass

    score = 0.0
    for fn in ("token_set_ratio", "partial_ratio", "ratio"):
        scorer = getattr(_rf_fuzz, fn, None)
        if scorer is None:
            continue
        try:
            score = max(score, float(scorer(left, right)) / 100.0)
        except Exception:
            pass
    return score


def _fuzzy_rank_rows(rows: list[tuple], query: str, n_results: int) -> list[tuple]:
    scored: list[tuple[float, tuple]] = []
    for row in rows:
        text = " ".join(str(item or "") for item in (row[5], row[3], row[2], row[4]))
        fuzz_score = _rapidfuzz_score(text, query)
        if fuzz_score > 0.0:
            scored.append((fuzz_score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[: int(max(1, n_results))]]


@dataclass
class MemoryPalace:
    config: Config
    _conn: sqlite3.Connection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ensure_db()

    def _ensure_db(self) -> None:
        if self._conn is not None:
            return
        if not self.config.memory_palace_enabled:
            return

        db_path = Path(str(getattr(self.config, "memory_palace_db_path", "memory_palace.db")))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                name TEXT,
                tool_call_id TEXT,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS raw_messages_fts USING fts5(
                    content,
                    session_id,
                    role,
                    name,
                    tool_call_id,
                    content='raw_messages',
                    content_rowid='rowid'
                );
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS raw_messages_ai AFTER INSERT ON raw_messages BEGIN
                    INSERT INTO raw_messages_fts(rowid, content, session_id, role, name, tool_call_id)
                    VALUES (new.rowid, new.content, new.session_id, new.role, new.name, new.tool_call_id);
                END;
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS raw_messages_ad AFTER DELETE ON raw_messages BEGIN
                    INSERT INTO raw_messages_fts(raw_messages_fts, rowid, content, session_id, role, name, tool_call_id)
                    VALUES ('delete', old.rowid, old.content, old.session_id, old.role, old.name, old.tool_call_id);
                END;
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS raw_messages_au AFTER UPDATE ON raw_messages BEGIN
                    INSERT INTO raw_messages_fts(raw_messages_fts, rowid, content, session_id, role, name, tool_call_id)
                    VALUES ('delete', old.rowid, old.content, old.session_id, old.role, old.name, old.tool_call_id);
                    INSERT INTO raw_messages_fts(rowid, content, session_id, role, name, tool_call_id)
                    VALUES (new.rowid, new.content, new.session_id, new.role, new.name, new.tool_call_id);
                END;
                """
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            self._conn.rollback()

    def save_message(
        self, session_id: str, message: Message, *, timestamp: float | None = None
    ) -> str:
        if not self.config.memory_palace_enabled:
            raise RuntimeError("MemoryPalace is disabled in configuration.")

        self._ensure_db()
        if self._conn is None:
            raise RuntimeError("MemoryPalace database not available.")

        raw_content = str(message.content or "")
        if not raw_content.strip():
            return ""
        content = raw_content

        ts = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(timestamp if timestamp is not None else time.time()),
        )
        if getattr(self.config, "memory_palace_apply_aaak", True):
            content = (
                compress_text_to_aaak(
                    content,
                    metadata={
                        "session": str(session_id),
                        "date": ts,
                        "source": str(message.name or "memory_palace"),
                        "title": str(message.role.value),
                    },
                ).strip()
                or content
            )

        row_id = f"palace-{uuid.uuid4().hex}"
        self._conn.execute(
            "INSERT OR REPLACE INTO raw_messages (id, session_id, role, name, tool_call_id, content, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row_id,
                str(session_id),
                str(message.role.value),
                str(message.name or ""),
                str(message.tool_call_id or ""),
                content,
                ts,
            ),
        )
        self._conn.commit()
        return row_id

    def search(
        self, query: str, *, n_results: int = 10, session_filter: str | None = None
    ) -> list[dict[str, Any]]:
        if not self.config.memory_palace_enabled:
            return []

        self._ensure_db()
        if self._conn is None:
            return []

        query_text = str(query or "").strip()
        where_clause = ""
        params: list[Any] = []
        if session_filter:
            where_clause = "WHERE session_id = ?"
            params.append(str(session_filter))

        rows: list[tuple] = []
        if query_text:
            try:
                cursor = self._conn.execute(
                    "SELECT id, session_id, role, name, tool_call_id, content, timestamp FROM raw_messages_fts "
                    f"{where_clause + (' AND ' if where_clause else 'WHERE ')}raw_messages_fts MATCH ? LIMIT ?;",
                    (*params, query_text, int(max(1, n_results))),
                )
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                rows = []

            if not rows:
                if _rf_fuzz is not None:
                    scan_limit = max(200, int(max(1, n_results)) * 20)
                    rows = self._conn.execute(
                        "SELECT id, session_id, role, name, tool_call_id, content, timestamp FROM raw_messages "
                        f"{where_clause} ORDER BY rowid DESC LIMIT ?;",
                        (*params, scan_limit),
                    ).fetchall()
                    rows = _fuzzy_rank_rows(rows, query_text, int(max(1, n_results)))
                else:
                    rows = self._conn.execute(
                        "SELECT id, session_id, role, name, tool_call_id, content, timestamp FROM raw_messages "
                        f"{where_clause + (' AND ' if where_clause else 'WHERE ')}content LIKE ? LIMIT ?;",
                        (*params, f"%{query_text}%", int(max(1, n_results))),
                    ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, session_id, role, name, tool_call_id, content, timestamp FROM raw_messages "
                f"{where_clause} ORDER BY rowid DESC LIMIT ?;",
                (*params, int(max(1, n_results))),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "name": row[3],
                    "tool_call_id": row[4],
                    "content": row[5],
                    "timestamp": row[6],
                    "metadata": {
                        "session_id": row[1],
                        "role": row[2],
                        "name": row[3],
                        "tool_call_id": row[4],
                        "timestamp": row[6],
                    },
                }
            )
        return results

    def count(self, session_filter: str | None = None) -> int:
        if not self.config.memory_palace_enabled:
            return 0

        self._ensure_db()
        if self._conn is None:
            return 0

        if session_filter:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM raw_messages WHERE session_id = ?;",
                (str(session_filter),),
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM raw_messages;")
        return int(cursor.fetchone()[0] or 0)

    def clear_session(self, session_id: str) -> None:
        if not self.config.memory_palace_enabled:
            return

        self._ensure_db()
        if self._conn is None:
            return

        self._conn.execute(
            "DELETE FROM raw_messages WHERE session_id = ?;",
            (str(session_id),),
        )
        self._conn.commit()
