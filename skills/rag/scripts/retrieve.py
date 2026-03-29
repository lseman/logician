"""RAG retrieval/introspection tools.

Focused on querying and inspecting indexed documents.

Tool inventory
--------------
rag_search -- semantic query over RAG chunks
rag_list   -- list indexed sources/chunk counts
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from src.repo_graph import related_repo_context
from src.repo_registry import load_repo_index

if "_safe_json" not in globals():

    def _safe_json(obj: Any) -> str:  # type: ignore[misc]
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json.dumps({"status": "error", "error": repr(obj)})


if "_as_list" not in globals():

    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, (str, bytes, bytearray)):
            return [value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            converted = tolist()
            if converted is None:
                return []
            if isinstance(converted, list):
                return converted
            if isinstance(converted, tuple):
                return list(converted)
            return [converted]
        try:
            return list(value)
        except Exception:
            return [value]


if "_first_batch" not in globals():

    def _first_batch(value: Any) -> list[Any]:
        seq = _as_list(value)
        if not seq:
            return []
        first = seq[0]
        if isinstance(first, dict):
            return seq
        return _as_list(first)


if "_looks_like_row_dict" not in globals():

    def _looks_like_row_dict(value: Any) -> bool:
        return isinstance(value, dict) and (
            "metadata" in value or "content" in value or "distance" in value
        )


if "_normalize_query_rows" not in globals():

    def _normalize_query_rows(results: Any) -> list[dict[str, Any]]:
        if results is None:
            return []

        rows = _as_list(results)
        if rows and _looks_like_row_dict(rows[0]):
            return rows
        batched_rows = _first_batch(rows)
        if batched_rows and _looks_like_row_dict(batched_rows[0]):
            return batched_rows

        if isinstance(results, dict):
            docs = _first_batch(results.get("documents"))
            metas = _first_batch(results.get("metadatas"))
            dists = _first_batch(results.get("distances"))
            if not dists:
                dists = _first_batch(results.get("scores"))
            return [
                {"content": doc, "metadata": meta or {}, "distance": float(dist)}
                for doc, meta, dist in zip(docs, metas, dists)
            ]

        return []


if "_get_doc_db" not in globals():

    def _get_doc_db():
        try:
            from src.db.document import DocumentDB

            return DocumentDB()
        except Exception as exc:
            raise RuntimeError(f"Cannot initialise DocumentDB: {exc}") from exc


if "_doc_db_from_agent" not in globals():

    def _doc_db_from_agent():
        try:
            mem = globals().get("ctx") and getattr(globals()["ctx"], "memory", None)
            if mem is not None:
                doc_db = getattr(mem, "_doc_db", None)
                if doc_db is not None:
                    return doc_db
                if hasattr(mem, "_ensure_doc_db"):
                    mem._ensure_doc_db()
                    if mem._doc_db is not None:
                        return mem._doc_db
        except Exception:
            pass
        return _get_doc_db()


def _ctx_active_repo_ids() -> list[str]:
    try:
        active_repos = getattr(globals().get("ctx"), "active_repos", []) or []
    except Exception:
        active_repos = []
    repo_ids: list[str] = []
    for item in active_repos:
        if not isinstance(item, dict):
            continue
        repo_id = str(item.get("id") or "").strip()
        if repo_id and repo_id not in repo_ids:
            repo_ids.append(repo_id)
    return repo_ids


def _parse_repo_ids(repo_id: str = "", repo_ids: str = "") -> list[str]:
    selected: list[str] = []
    single = str(repo_id or "").strip()
    if single:
        selected.append(single)

    raw = str(repo_ids or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                value = str(item or "").strip()
                if value and value not in selected:
                    selected.append(value)
        else:
            for chunk in raw.replace("\n", ",").split(","):
                value = str(chunk or "").strip()
                if value and value not in selected:
                    selected.append(value)

    if not selected:
        selected.extend(_ctx_active_repo_ids())
    return selected


def _repo_where(repo_id: str = "", repo_ids: str = "") -> dict[str, Any] | None:
    selected = _parse_repo_ids(repo_id=repo_id, repo_ids=repo_ids)
    if not selected:
        return None
    if len(selected) == 1:
        return {"repo_id": selected[0]}
    return {"$or": [{"repo_id": item} for item in selected]}


def _repo_lookup() -> dict[str, dict[str, Any]]:
    return {
        str(item.get("id") or "").strip(): dict(item)
        for item in load_repo_index()
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }


def _paths_where(paths: list[str]) -> dict[str, Any] | None:
    clean = [str(path or "").strip() for path in paths if str(path or "").strip()]
    if not clean:
        return None
    if len(clean) == 1:
        return {"repo_rel_path": clean[0]}
    return {"$or": [{"repo_rel_path": path} for path in clean]}


def _merge_where(lhs: dict[str, Any] | None, rhs: dict[str, Any] | None) -> dict[str, Any] | None:
    if lhs and rhs:
        return {"$and": [lhs, rhs]}
    return lhs or rhs


def rag_search(
    query: str,
    top_k: int = 5,
    ef_search: int = 0,
    repo_id: str = "",
    repo_ids: str = "",
) -> str:
    """Use when: Inspect what RAG context the agent would retrieve for a given query.

    Triggers: what's in rag, search memory, check rag, debug retrieval, what do you know about.
    Inputs:
      query (str, required): The search query.
      top_k (int, optional): Number of results to return (default 5).
      ef_search (int, optional): ANN query-time breadth override; 0 uses current default.
    Returns: JSON with matching chunks, sources, and similarity distances.
    Side effects: Read-only.
    """
    # try:
    doc_db = _doc_db_from_agent()
    query_ef = int(ef_search) if int(ef_search) > 0 else None
    where = _repo_where(repo_id=repo_id, repo_ids=repo_ids)
    results = doc_db.query(
        query,
        n_results=min(int(top_k), 50),
        where=where,
        ef_search=query_ef,
    )

    # check if results is a numpy array or similar and convert to list if needed
    if hasattr(results, "tolist") and callable(getattr(results, "tolist")):
        results = results.tolist()
    rows = _normalize_query_rows(results)
    if not rows:
        return _safe_json(
            {
                "status": "ok",
                "query": query,
                "results": [],
                "message": "No matches found — RAG store may be empty.",
            }
        )

    hits = [
        {
            "source": (r.get("metadata") or {}).get("source", "unknown"),
            "path": (r.get("metadata") or {}).get("path", ""),
            "repo_id": (r.get("metadata") or {}).get("repo_id", ""),
            "repo_name": (r.get("metadata") or {}).get("repo_name", ""),
            "repo_rel_path": (r.get("metadata") or {}).get("repo_rel_path", ""),
            "distance": round(float(r.get("distance", 0.0)), 4),
            "chunk": (r.get("metadata") or {}).get("chunk", 0),
            "preview": str(r.get("content", ""))[:300]
            + ("…" if len(str(r.get("content", ""))) > 300 else ""),
        }
        for r in rows
    ]
    repo_filter = _parse_repo_ids(repo_id=repo_id, repo_ids=repo_ids)

    graph_expansion: list[dict[str, Any]] = []
    expanded_hits: list[dict[str, Any]] = []
    if repo_filter:
        repo_map = _repo_lookup()
        doc_db = _doc_db_from_agent()
        for active_repo_id in repo_filter[:4]:
            repo = repo_map.get(active_repo_id)
            if repo is None:
                continue
            seed_paths = [
                str((row.get("metadata") or {}).get("repo_rel_path", "")).strip()
                for row in rows
                if str((row.get("metadata") or {}).get("repo_id", "")).strip()
                == active_repo_id
                and str((row.get("metadata") or {}).get("repo_rel_path", "")).strip()
            ]
            if not seed_paths:
                continue
            related = related_repo_context(
                repo,
                rel_paths=seed_paths,
                query=query,
                limit=6,
            )
            related_files = [
                str(item.get("rel_path") or "").strip()
                for item in list(related.get("related_files") or [])
                if str(item.get("rel_path") or "").strip()
            ]
            graph_expansion.append(
                {
                    "repo_id": active_repo_id,
                    "seed_paths": seed_paths[:6],
                    "related_files": list(related.get("related_files") or []),
                    "related_symbols": list(related.get("related_symbols") or []),
                }
            )
            extra_where = _merge_where(
                {"repo_id": active_repo_id},
                _paths_where(related_files[:6]),
            )
            if extra_where is None:
                continue
            extra_rows = doc_db.query(
                query,
                n_results=min(max(int(top_k), 2), 6),
                where=extra_where,
                ef_search=query_ef,
            )
            for extra in _normalize_query_rows(extra_rows):
                meta = extra.get("metadata") or {}
                expanded_hits.append(
                    {
                        "source": meta.get("source", "unknown"),
                        "path": meta.get("path", ""),
                        "repo_id": meta.get("repo_id", ""),
                        "repo_name": meta.get("repo_name", ""),
                        "repo_rel_path": meta.get("repo_rel_path", ""),
                        "distance": round(float(extra.get("distance", 0.0)), 4),
                        "chunk": meta.get("chunk", 0),
                        "preview": str(extra.get("content", ""))[:220]
                        + ("…" if len(str(extra.get("content", ""))) > 220 else ""),
                    }
                )

    seen_expanded: set[tuple[str, str, int]] = set()
    dedup_expanded: list[dict[str, Any]] = []
    primary_keys = {
        (
            str(item.get("repo_id") or ""),
            str(item.get("repo_rel_path") or ""),
            int(item.get("chunk", 0) or 0),
        )
        for item in hits
    }
    for item in expanded_hits:
        key = (
            str(item.get("repo_id") or ""),
            str(item.get("repo_rel_path") or ""),
            int(item.get("chunk", 0) or 0),
        )
        if key in primary_keys or key in seen_expanded:
            continue
        seen_expanded.add(key)
        dedup_expanded.append(item)
        if len(dedup_expanded) >= 8:
            break
    return _safe_json(
        {
            "status": "ok",
            "query": query,
            "count": len(hits),
            "repo_filter": repo_filter,
            "results": hits,
            "graph_expansion": graph_expansion,
            "expanded_results": dedup_expanded,
        }
    )
    # except Exception as exc:
    # return _safe_json({"status": "error", "error": str(exc)})


def rag_list(
    max_sources: int = 40,
    include_paths: bool = True,
    repo_id: str = "",
    repo_ids: str = "",
) -> str:
    """Use when: You need to verify what has already been indexed into RAG.

    Triggers: list rag, show indexed docs, what files are in rag, rag inventory.
    Inputs:
      max_sources (int, optional): Maximum source rows to return (default 40, max 300).
      include_paths (bool, optional): Include source file paths when available (default True).
    Returns: JSON summary of indexed sources and chunk totals.
    Side effects: Read-only.
    """
    try:
        limit = max(1, min(int(max_sources), 300))
        doc_db = _doc_db_from_agent()
        collection = doc_db.collection
        grouped: dict[tuple[str, str], int] = defaultdict(int)
        where = _repo_where(repo_id=repo_id, repo_ids=repo_ids)
        total_chunks = int(collection.count(where=where))
        total_with_deleted = int(collection.count(where=where, include_deleted=True))
        deleted_chunks = max(0, total_with_deleted - total_chunks)
        records = collection.get(
            include=["metadatas"],
            include_deleted=False,
            where=where,
        )
        metadatas = records.get("metadatas", [])

        for meta in metadatas:
            source = str(meta.get("source", "unknown") or "unknown")
            path = str(meta.get("path", "") or "")
            key = (source, path if include_paths else "")
            grouped[key] += 1

        ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
        sources: list[dict[str, Any]] = []
        for (source, path), chunks in ranked[:limit]:
            row: dict[str, Any] = {"source": source, "chunks": int(chunks)}
            if include_paths and path:
                row["path"] = path
            sources.append(row)

        return _safe_json(
            {
                "status": "ok",
                "total_chunks": total_chunks,
                "deleted_chunks": deleted_chunks,
                "unique_sources": len(grouped),
                "returned_sources": len(sources),
                "repo_filter": _parse_repo_ids(repo_id=repo_id, repo_ids=repo_ids),
                "sources": sources,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


__tools__ = [rag_search, rag_list]
