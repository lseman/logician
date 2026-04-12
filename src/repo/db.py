from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .registry import get_repo_root

_DEFAULT_DB_FILENAME = "{repo_id}.db"


def get_repo_db_path(repo_id: str, *, base_dir: str | Path | None = None) -> Path:
    root = get_repo_root(base_dir)
    return root / _DEFAULT_DB_FILENAME.format(repo_id=str(repo_id).strip())


def _sqlite_connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _initialize_repo_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS repo_nodes (
            id TEXT PRIMARY KEY,
            repo_id TEXT NOT NULL,
            kind TEXT,
            name TEXT,
            path TEXT,
            rel_path TEXT,
            language TEXT,
            file_kind TEXT,
            symbol_kind TEXT,
            line INTEGER,
            metadata TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS repo_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id TEXT NOT NULL,
            kind TEXT,
            source TEXT,
            target TEXT,
            target_rel_path TEXT,
            import_path TEXT,
            symbol_name TEXT,
            local_symbol TEXT,
            line INTEGER,
            metadata TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_repo_nodes_repo_id ON repo_nodes(repo_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_repo_edges_repo_id ON repo_edges(repo_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_repo_edges_source ON repo_edges(repo_id, source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_repo_edges_target ON repo_edges(repo_id, target)")


def _flatten_record(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def _repo_id_from_repo(repo: dict[str, Any]) -> str:
    repo_id = str(repo.get("id") or "").strip()
    if not repo_id:
        raise ValueError("Repository record is missing an id")
    return repo_id


def _db_path_from_repo(repo: dict[str, Any]) -> Path:
    artifacts = dict(repo.get("artifacts") or {})
    path = artifacts.get("db_path")
    if path:
        return Path(str(path)).expanduser().resolve()
    return get_repo_db_path(_repo_id_from_repo(repo))


def write_repo_graph_db(repo: dict[str, Any], records: list[dict[str, Any]]) -> str:
    repo_id = _repo_id_from_repo(repo)
    db_path = _db_path_from_repo(repo)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _sqlite_connect(db_path)
    try:
        _initialize_repo_db(conn)
        conn.execute("DELETE FROM repo_nodes WHERE repo_id = ?", (repo_id,))
        conn.execute("DELETE FROM repo_edges WHERE repo_id = ?", (repo_id,))

        node_insert = """
            INSERT OR REPLACE INTO repo_nodes
            (id, repo_id, kind, name, path, rel_path, language, file_kind, symbol_kind, line, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        edge_insert = """
            INSERT INTO repo_edges
            (repo_id, kind, source, target, target_rel_path, import_path, symbol_name, local_symbol, line, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for record in records:
            if str(record.get("record_type") or "") == "node":
                conn.execute(
                    node_insert,
                    (
                        str(record.get("id") or "").strip(),
                        repo_id,
                        str(record.get("kind") or "").strip(),
                        str(record.get("name") or "").strip(),
                        str(record.get("path") or "").strip(),
                        str(record.get("rel_path") or "").strip(),
                        str(record.get("language") or "").strip(),
                        str(record.get("file_kind") or "").strip(),
                        str(record.get("symbol_kind") or "").strip(),
                        int(record.get("line") or 0),
                        _flatten_record(record),
                    ),
                )
            elif str(record.get("record_type") or "") == "edge":
                conn.execute(
                    edge_insert,
                    (
                        repo_id,
                        str(record.get("kind") or "").strip(),
                        str(record.get("source") or "").strip(),
                        str(record.get("target") or "").strip(),
                        str(record.get("target_rel_path") or "").strip(),
                        str(record.get("import_path") or "").strip(),
                        str(record.get("symbol_name") or "").strip(),
                        str(record.get("local_symbol") or "").strip(),
                        int(record.get("line") or 0),
                        _flatten_record(record),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    return str(db_path)


def load_repo_graph_db(repo: dict[str, Any]) -> dict[str, Any]:
    db_path = _db_path_from_repo(repo)
    if not db_path.exists():
        return {"nodes": {}, "edges": []}

    conn = _sqlite_connect(db_path)
    try:
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        for row in conn.execute(
            "SELECT * FROM repo_nodes WHERE repo_id = ?", (str(repo.get("id") or ""),)
        ):
            try:
                payload = json.loads(str(row["metadata"] or "{}"))
            except Exception:
                payload = {
                    "record_type": "node",
                    "id": str(row["id"] or ""),
                    "kind": str(row["kind"] or ""),
                    "name": str(row["name"] or ""),
                    "path": str(row["path"] or ""),
                    "rel_path": str(row["rel_path"] or ""),
                    "language": str(row["language"] or ""),
                    "file_kind": str(row["file_kind"] or ""),
                    "symbol_kind": str(row["symbol_kind"] or ""),
                    "line": int(row["line"] or 0),
                }
            nodes[str(payload.get("id") or "").strip()] = payload

        for row in conn.execute(
            "SELECT * FROM repo_edges WHERE repo_id = ?", (str(repo.get("id") or ""),)
        ):
            try:
                payload = json.loads(str(row["metadata"] or "{}"))
            except Exception:
                payload = {
                    "record_type": "edge",
                    "repo_id": str(repo.get("id") or ""),
                    "kind": str(row["kind"] or ""),
                    "source": str(row["source"] or ""),
                    "target": str(row["target"] or ""),
                    "target_rel_path": str(row["target_rel_path"] or ""),
                    "import_path": str(row["import_path"] or ""),
                    "symbol_name": str(row["symbol_name"] or ""),
                    "local_symbol": str(row["local_symbol"] or ""),
                    "line": int(row["line"] or 0),
                }
            edges.append(payload)
    finally:
        conn.close()

    return {"nodes": nodes, "edges": edges}


def search_repo_db(
    repo: dict[str, Any],
    *,
    query: str,
    kind: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    db_path = _db_path_from_repo(repo)
    if not db_path.exists():
        return []

    terms = [term.lower() for term in str(query or "").split() if term.strip()]
    if not terms:
        return []

    sql = ["SELECT * FROM repo_nodes WHERE repo_id = ?"]
    params: list[Any] = [str(repo.get("id") or "")]
    if kind:
        sql.append("AND lower(kind) = ?")
        params.append(str(kind).strip().lower())

    for term in terms:
        sql.append(
            "AND (lower(name) LIKE ? OR lower(rel_path) LIKE ? OR lower(path) LIKE ? OR lower(symbol_kind) LIKE ? OR lower(metadata) LIKE ?)"
        )
        param = f"%{term}%"
        params.extend([param] * 5)

    sql.append("LIMIT ?")
    params.append(min(max(1, int(limit or 20)), 200))
    query_str = " ".join(sql)

    conn = _sqlite_connect(db_path)
    try:
        results: list[dict[str, Any]] = []
        for row in conn.execute(query_str, tuple(params)):
            try:
                payload = json.loads(str(row["metadata"] or "{}"))
            except Exception:
                payload = {
                    "record_type": "node",
                    "id": str(row["id"] or ""),
                    "repo_id": str(row["repo_id"] or ""),
                    "kind": str(row["kind"] or ""),
                    "name": str(row["name"] or ""),
                    "path": str(row["path"] or ""),
                    "rel_path": str(row["rel_path"] or ""),
                    "language": str(row["language"] or ""),
                    "file_kind": str(row["file_kind"] or ""),
                    "symbol_kind": str(row["symbol_kind"] or ""),
                    "line": int(row["line"] or 0),
                }
            results.append(payload)
        return results
    finally:
        conn.close()
