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


def rag_search(query: str, top_k: int = 5, ef_search: int = 0) -> str:
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
    results = doc_db.query(
        query,
        n_results=min(int(top_k), 50),
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
            "distance": round(float(r.get("distance", 0.0)), 4),
            "chunk": (r.get("metadata") or {}).get("chunk", 0),
            "preview": str(r.get("content", ""))[:300]
            + ("…" if len(str(r.get("content", ""))) > 300 else ""),
        }
        for r in rows
    ]
    return _safe_json({"status": "ok", "query": query, "count": len(hits), "results": hits})
    # except Exception as exc:
    # return _safe_json({"status": "error", "error": str(exc)})


def rag_list(max_sources: int = 40, include_paths: bool = True) -> str:
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
        total_chunks = int(collection.count())
        total_with_deleted = int(collection.count(include_deleted=True))
        deleted_chunks = max(0, total_with_deleted - total_chunks)
        records = collection.get(include=["metadatas"], include_deleted=False)
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
                "sources": sources,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


__tools__ = [rag_search, rag_list]
