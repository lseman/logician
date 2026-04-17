from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..config import Config
from ..memory import Memory
from ..project_memory import (
    get_memory_dir,
    get_obs_dir,
    load_index,
    parse_frontmatter,
)
from ..runtime_paths import legacy_state_root, session_db_path
from ..repo_graph import load_repo_graph
from ..repo_registry import load_repo_index
from ..tools.core.tasks import load_persisted_todos


def get_db_path() -> Path:
    managed = session_db_path()
    if managed.exists():
        return managed
    legacy = legacy_state_root() / "agent_sessions.db"
    if legacy.exists():
        return legacy.resolve()
    return managed


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _message_columns(conn: sqlite3.Connection) -> set[str]:
    try:
        rows = conn.execute("PRAGMA table_info(messages)").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row["name"] or "") for row in rows}


def _decode_thinking_log(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if str(item or "").strip()]


def _truncate(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _session_title(text: str, fallback: str = "New conversation") -> str:
    compact = " ".join(str(text or "").split())
    if not compact:
        return fallback
    compact = re.sub(r"^please\s+", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"^[\"'`]+|[\"'`]+$", "", compact).strip()
    compact = compact.split("\n", 1)[0].strip()
    compact = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0].strip()
    if not compact:
        return fallback
    words = compact.split()
    title = " ".join(words[:7])
    if len(words) > 7 or len(compact) > len(title):
        title = title.rstrip(".,:;!?") + "..."
    if len(title) > 56:
        title = title[:56].rstrip() + "..."
    return title[:1].upper() + title[1:]


def _latest_runtime_todos() -> list[dict[str, Any]]:
    db_path = get_db_path()
    if not db_path.exists():
        return []
    conn = _connect_db()
    try:
        row = conn.execute(
            """
            SELECT state_json
            FROM session_runtime_state
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    if row is None:
        return []
    try:
        payload = json.loads(str(row["state_json"]))
    except (TypeError, json.JSONDecodeError):
        return []
    items = payload.get("todo_items") or []
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict)]


def get_current_todos() -> list[dict[str, Any]]:
    persisted = load_persisted_todos()
    if persisted:
        return persisted
    return _latest_runtime_todos()


def _ensure_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS session_metadata "
        "(session_id TEXT PRIMARY KEY, custom_title TEXT)"
    )


def rename_session(session_id: str, title: str) -> None:
    session_id = str(session_id or "").strip()
    title = str(title or "").strip()[:80]
    if not session_id:
        return
    db_path = get_db_path()
    if not db_path.exists():
        return
    conn = _connect_db()
    try:
        _ensure_metadata_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO session_metadata (session_id, custom_title) VALUES (?, ?)",
            (session_id, title),
        )
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        conn.close()


def list_sessions(limit: int = 24) -> list[dict[str, Any]]:
    db_path = get_db_path()
    if not db_path.exists():
        return []
    conn = _connect_db()
    try:
        rows = conn.execute(
            """
            SELECT
                session_id,
                MAX(timestamp) AS last_updated,
                COUNT(*) AS message_count
            FROM messages
            GROUP BY session_id
            ORDER BY MAX(timestamp) DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()

        custom_titles: dict[str, str] = {}
        try:
            _ensure_metadata_table(conn)
            for meta in conn.execute("SELECT session_id, custom_title FROM session_metadata").fetchall():
                if meta["custom_title"]:
                    custom_titles[str(meta["session_id"])] = str(meta["custom_title"])
        except sqlite3.Error:
            pass

        sessions: list[dict[str, Any]] = []
        for row in rows:
            title_row = conn.execute(
                """
                SELECT content
                FROM messages
                WHERE session_id=? AND role='user'
                ORDER BY id ASC
                LIMIT 1
                """,
                (str(row["session_id"]),),
            ).fetchone()
            if title_row is None:
                title_row = conn.execute(
                    """
                    SELECT content
                    FROM messages
                    WHERE session_id=?
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (str(row["session_id"]),),
                ).fetchone()
            preview_row = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id=?
                ORDER BY id DESC
                LIMIT 1
                """,
                (str(row["session_id"]),),
            ).fetchone()
            preview_content = str(preview_row["content"] or "") if preview_row is not None else ""
            preview_role = str(preview_row["role"] or "") if preview_row is not None else ""
            title_content = str(title_row["content"] or "") if title_row is not None else ""
            sessions.append(
                {
                    "id": str(row["session_id"]),
                    "last_updated": str(row["last_updated"] or ""),
                    "message_count": int(row["message_count"] or 0),
                    "title": custom_titles.get(str(row["session_id"])) or _session_title(title_content),
                    "preview": _truncate(preview_content),
                    "last_role": preview_role,
                }
            )
        return sessions
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def truncate_session_at(session_id: str, at_id: int) -> None:
    """Delete all messages with id >= at_id from session_id."""
    db_path = get_db_path()
    if not db_path.exists():
        return
    conn = _connect_db()
    try:
        conn.execute(
            "DELETE FROM messages WHERE session_id=? AND id >= ?",
            (session_id, at_id),
        )
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        conn.close()


def get_session_detail(session_id: str) -> dict[str, Any]:
    session_id = str(session_id or "").strip()
    if not session_id:
        return {"id": "", "messages": [], "runtime": {}, "todos": []}

    db_path = get_db_path()
    if not db_path.exists():
        return {"id": session_id, "messages": [], "runtime": {}, "todos": []}

    conn = _connect_db()
    try:
        message_columns = _message_columns(conn)
        has_thinking_log = "thinking_log" in message_columns
        message_rows = conn.execute(
            f"""
            SELECT id, role, content, name, tool_call_id, timestamp
            {", thinking_log" if has_thinking_log else ""}
            FROM messages
            WHERE session_id=?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        runtime_row = conn.execute(
            """
            SELECT state_json, updated_at
            FROM session_runtime_state
            WHERE session_id=?
            """,
            (session_id,),
        ).fetchone()
    except sqlite3.Error:
        return {"id": session_id, "messages": [], "runtime": {}, "todos": []}
    finally:
        conn.close()

    runtime: dict[str, Any] = {}
    if runtime_row is not None:
        try:
            runtime = json.loads(str(runtime_row["state_json"]))
        except (TypeError, json.JSONDecodeError):
            runtime = {}

    return {
        "id": session_id,
        "messages": [
            {
                "id": int(row["id"]),
                "role": str(row["role"] or ""),
                "content": str(row["content"] or ""),
                "name": str(row["name"] or ""),
                "tool_call_id": str(row["tool_call_id"] or ""),
                "timestamp": str(row["timestamp"] or ""),
                "thinking_log": (
                    _decode_thinking_log(row["thinking_log"])
                    if has_thinking_log
                    else []
                ),
            }
            for row in message_rows
        ],
        "runtime": runtime,
        "todos": [
            dict(item)
            for item in list(runtime.get("todo_items") or [])
            if isinstance(item, dict)
        ],
        "updated_at": str(runtime_row["updated_at"] or "") if runtime_row is not None else "",
    }


def get_memory_overview() -> dict[str, Any]:
    index = load_index()
    memory_dir = get_memory_dir()
    summary_path = memory_dir / "MEMORY.md"
    return {
        "count": len(index),
        "types": {
            key: sum(1 for item in index if str(item.get("type") or "") == key)
            for key in sorted(
                {
                    str(item.get("type") or "").strip()
                    for item in index
                    if str(item.get("type") or "").strip()
                }
            )
        },
        "summary_markdown": (
            summary_path.read_text(encoding="utf-8", errors="replace")
            if summary_path.exists()
            else ""
        ),
    }


def list_memory_observations(
    *,
    query: str = "",
    obs_type: str = "",
    limit: int = 120,
) -> list[dict[str, Any]]:
    rows = load_index()
    lowered_query = str(query or "").strip().lower()
    lowered_type = str(obs_type or "").strip().lower()
    obs_dir = get_obs_dir()
    items: list[dict[str, Any]] = []

    for entry in reversed(rows):
        entry_type = str(entry.get("type") or "").strip().lower()
        if lowered_type and entry_type != lowered_type:
            continue

        obs_id = int(entry.get("id", 0) or 0)
        obs_path = obs_dir / f"{obs_id:04d}.md"
        content = ""
        if obs_path.exists():
            raw = obs_path.read_text(encoding="utf-8", errors="replace")
            parts = raw.split("---", 2)
            content = parts[-1].strip() if len(parts) >= 3 else raw.strip()

        haystack = " ".join(
            [
                str(entry.get("title") or ""),
                str(entry.get("preview") or ""),
                content,
                " ".join(str(path) for path in (entry.get("files") or [])),
            ]
        ).lower()
        if lowered_query and lowered_query not in haystack:
            continue

        items.append(
            {
                "id": obs_id,
                "session": str(entry.get("session") or ""),
                "type": entry_type,
                "emoji": str(entry.get("emoji") or ""),
                "timestamp": str(entry.get("timestamp") or ""),
                "title": str(entry.get("title") or ""),
                "preview": str(entry.get("preview") or ""),
                "content": content,
                "files": [str(path) for path in (entry.get("files") or [])],
                "path": str(obs_path),
            }
        )
        if len(items) >= max(1, int(limit)):
            break
    return items


def list_memory_facts(limit: int = 60) -> list[dict[str, Any]]:
    memory_dir = get_memory_dir()
    fact_rows: list[dict[str, Any]] = []
    for file_path in sorted(memory_dir.glob("*.md")):
        if file_path.name == "MEMORY.md":
            continue
        mem_type, description = parse_frontmatter(
            file_path.read_text(encoding="utf-8", errors="replace")
        )
        if not description:
            continue
        fact_rows.append(
            {
                "name": file_path.stem,
                "type": mem_type,
                "description": description,
                "path": str(file_path),
            }
        )

    facts_dir = memory_dir / "facts"
    if facts_dir.exists():
        for file_path in sorted(facts_dir.glob("*.md")):
            mem_type, description = parse_frontmatter(
                file_path.read_text(encoding="utf-8", errors="replace")
            )
            if not description:
                continue
            fact_rows.append(
                {
                    "name": file_path.stem,
                    "type": mem_type,
                    "description": description,
                    "path": str(file_path),
                }
            )
    return fact_rows[: max(1, int(limit))]


def list_repos() -> list[dict[str, Any]]:
    return [dict(item) for item in load_repo_index()]


def _find_repo(repo_id: str) -> dict[str, Any] | None:
    normalized = str(repo_id or "").strip()
    if not normalized:
        return None
    for repo in load_repo_index():
        if str(repo.get("id") or "").strip() == normalized:
            return dict(repo)
    return None


def _rel_path_for_repo(repo: dict[str, Any], path_value: str) -> str:
    repo_path = Path(str(repo.get("path") or "")).resolve()
    try:
        return str(Path(path_value).resolve().relative_to(repo_path))
    except Exception:
        return ""


def _graph_payload(
    repo: dict[str, Any],
    *,
    focus_paths: list[str] | None = None,
    query: str = "",
    max_nodes: int = 120,
) -> dict[str, Any]:
    graph = load_repo_graph(repo)
    node_map = dict(graph.get("nodes") or {})
    all_edges = list(graph.get("edges") or [])
    focus = {
        str(path).strip()
        for path in (focus_paths or [])
        if str(path).strip()
    }
    include_ids: set[str] = {f"repo:{repo.get('id')}"}
    include_edges: list[dict[str, Any]] = []

    if focus:
        file_ids = {f"file:{path}" for path in focus}
        include_ids.update(file_ids)
        for edge in all_edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            target_rel_path = str(edge.get("target_rel_path") or "").strip()
            source_rel_path = ""
            if source.startswith("file:"):
                source_rel_path = source.split("file:", 1)[1]
            if (
                source in file_ids
                or target in file_ids
                or target_rel_path in focus
                or source_rel_path in focus
            ):
                include_edges.append(edge)
                include_ids.add(source)
                include_ids.add(target)
            if len(include_ids) >= max_nodes:
                break
    else:
        for node_id, node in node_map.items():
            kind = str(node.get("kind") or "")
            if kind == "file" or node_id.startswith("repo:"):
                include_ids.add(node_id)
            if len(include_ids) >= max_nodes:
                break
        include_edges = [
            edge
            for edge in all_edges
            if str(edge.get("source") or "") in include_ids
            and str(edge.get("target") or "") in include_ids
        ]

    query_terms = {
        term.lower()
        for term in str(query or "").split()
        if len(term.strip()) >= 2
    }
    nodes = []
    for node_id in include_ids:
        node = node_map.get(node_id)
        if not node:
            if node_id.startswith("repo:"):
                nodes.append(
                    {
                        "id": node_id,
                        "label": str(repo.get("name") or repo.get("id") or "repo"),
                        "kind": "repo",
                        "rel_path": "",
                        "highlight": False,
                    }
                )
            continue
        label = str(node.get("name") or node.get("rel_path") or node_id)
        rel_path = str(node.get("rel_path") or "")
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "kind": str(node.get("kind") or "unknown"),
                "symbol_kind": str(node.get("symbol_kind") or ""),
                "rel_path": rel_path,
                "line": int(node.get("line", 0) or 0),
                "highlight": bool(rel_path in focus)
                or any(term in label.lower() for term in query_terms),
            }
        )

    edges = [
        {
            "source": str(edge.get("source") or ""),
            "target": str(edge.get("target") or ""),
            "kind": str(edge.get("kind") or ""),
        }
        for edge in include_edges[: max_nodes * 3]
        if str(edge.get("source") or "") in include_ids
        and str(edge.get("target") or "") in include_ids
    ]

    return {
        "repo": dict(repo),
        "focus_paths": sorted(focus),
        "nodes": nodes[:max_nodes],
        "edges": edges,
    }


def get_repo_graph(repo_id: str, *, query: str = "", max_nodes: int = 120) -> dict[str, Any]:
    repo = _find_repo(repo_id)
    if repo is None:
        raise KeyError(repo_id)
    return _graph_payload(repo, query=query, max_nodes=max_nodes)


def get_repo_focus_graph(
    repo_id: str,
    *,
    focus_paths: list[str] | None = None,
    query: str = "",
    max_nodes: int = 120,
) -> dict[str, Any]:
    repo = _find_repo(repo_id)
    if repo is None:
        raise KeyError(repo_id)
    return _graph_payload(
        repo,
        focus_paths=focus_paths,
        query=query,
        max_nodes=max_nodes,
    )


def extract_repo_focus_paths(
    repo_id: str,
    *,
    arguments: dict[str, Any] | None = None,
    result_output: str = "",
) -> list[str]:
    repo = _find_repo(repo_id)
    if repo is None:
        return []

    repo_root = Path(str(repo.get("path") or "")).resolve()
    collected: set[str] = set()

    def _maybe_add_path(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            normalized = text.replace("\\", "/").lstrip("./")
            if normalized:
                collected.add(normalized)
            return
        try:
            rel_path = str(candidate.resolve().relative_to(repo_root))
        except Exception:
            return
        if rel_path:
            collected.add(rel_path)

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key or "").strip().lower()
                if lowered in {
                    "path",
                    "file",
                    "file_path",
                    "filename",
                    "rel_path",
                    "repo_rel_path",
                    "source",
                }:
                    _maybe_add_path(item)
                else:
                    _walk(item)
            return
        if isinstance(value, list):
            for item in value:
                _walk(item)
            return
        if isinstance(value, tuple):
            for item in value:
                _walk(item)

    _walk(dict(arguments or {}))

    raw = str(result_output or "").strip()
    if raw.startswith("{") or raw.startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            _walk(payload)

    return sorted(collected)


def search_rag(
    query: str,
    *,
    repo_id: str = "",
    n_results: int = 8,
) -> dict[str, Any]:
    repo = _find_repo(repo_id) if repo_id else None
    memory = Memory(config=Config(), db_path=str(get_db_path()), lazy_rag=True)
    rows = memory.search_rag(query, n_results=max(1, int(n_results)))

    if repo is not None:
        repo_path = Path(str(repo.get("path") or "")).resolve()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            path_value = str(metadata.get("path") or "")
            if not path_value:
                continue
            try:
                Path(path_value).resolve().relative_to(repo_path)
            except Exception:
                continue
            filtered.append(row)
        rows = filtered

    result_rows = []
    focus_paths: list[str] = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        source_path = str(metadata.get("path") or "")
        rel_path = _rel_path_for_repo(repo, source_path) if repo is not None else ""
        if rel_path:
            focus_paths.append(rel_path)
        result_rows.append(
            {
                "content": str(row.get("content") or ""),
                "metadata": metadata,
                "rel_path": rel_path,
                "source": str(metadata.get("source") or ""),
            }
        )

    return {
        "query": str(query),
        "results": result_rows,
        "graph": (
            _graph_payload(repo, focus_paths=focus_paths, query=query)
            if repo is not None
            else {"nodes": [], "edges": [], "focus_paths": []}
        ),
    }
