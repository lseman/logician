"""RAG speed/quality tuning tools.

Focused on runtime retrieval trade-offs (latency vs quality).

Tool inventory
--------------
rag_tuning_status  -- inspect active retrieval knobs
rag_apply_profile  -- apply speed/balanced/quality retrieval profile
rag_benchmark      -- benchmark retrieval latency for a query
"""

from __future__ import annotations

import json
import time
from statistics import mean
from typing import Any

if "_safe_json" not in globals():

    def _safe_json(obj: Any) -> str:  # type: ignore[misc]
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json.dumps({"status": "error", "error": repr(obj)})

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


def _active_chunk_counts(collection: Any) -> tuple[int, int]:
    if hasattr(collection, "count"):
        try:
            active = int(collection.count())
            total = int(collection.count(include_deleted=True))
            return active, max(0, total - active)
        except Exception:
            pass

    payload_by_label = getattr(collection, "_payload_by_label", {}) or {}
    if not isinstance(payload_by_label, dict):
        return 0, 0
    active = 0
    deleted = 0
    for rec in payload_by_label.values():
        if not isinstance(rec, dict):
            continue
        if rec.get("deleted", False):
            deleted += 1
        else:
            active += 1
    return active, deleted


def _tuning_snapshot(doc_db: Any) -> dict[str, Any]:
    collection = doc_db.collection
    reranker = getattr(collection, "_reranker", None)
    embedder = getattr(collection, "_embedder", None)
    configured_backend = str(
        getattr(doc_db, "vector_backend", "")
        or getattr(
            getattr(getattr(globals().get("ctx"), "memory", None), "config", None),
            "rag_vector_backend",
            "",
        )
        or "unknown"
    )

    active_chunks, deleted_chunks = _active_chunk_counts(collection)
    snapshot = {
        "configured_vector_backend": configured_backend,
        "vector_backend": str(getattr(collection, "_backend", "unknown")),
        "embedding_model": str(
            getattr(embedder, "resolved_model_name", "")
            or getattr(doc_db, "embedding_model_name", "")
        ),
        "rerank_enabled": bool(
            getattr(reranker, "enabled", getattr(doc_db, "rerank_enabled", False))
        ),
        "reranker_model": str(
            getattr(reranker, "model_name", "")
            or getattr(doc_db, "reranker_model_name", "")
        ),
        "rerank_fetch_k": int(getattr(doc_db, "rerank_fetch_k", 0) or 0),
        "min_similarity": float(
            getattr(collection, "_min_similarity", getattr(doc_db, "min_similarity", 0.0))
            or 0.0
        ),
        "hnsw_ef_search": int(getattr(collection, "_ef_search", 0) or 0),
        "hnsw_m": int(getattr(collection, "_m", 0) or 0),
        "hnsw_ef_construction": int(getattr(collection, "_ef_construction", 0) or 0),
        "collection_dir": str(getattr(collection, "_dir", "")),
        "active_chunks": active_chunks,
        "deleted_chunks": deleted_chunks,
    }
    if snapshot["configured_vector_backend"] != snapshot["vector_backend"]:
        snapshot["warning"] = (
            "Configured RAG backend differs from the live runtime backend. "
            "Reload the agent/runtime so the new backend takes effect."
        )
    return snapshot


def rag_tuning_status() -> str:
    """Use when: You want to understand current RAG retrieval speed/quality settings.

    Triggers: rag config, rag settings, retrieval knobs, rerank status, index tuning.
    Returns: JSON snapshot of active retrieval settings.
    Side effects: Read-only.
    """
    try:
        doc_db = _doc_db_from_agent()
        return _safe_json({"status": "ok", "tuning": _tuning_snapshot(doc_db)})
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def rag_apply_profile(
    profile: str = "balanced",
    min_similarity: float = -1.0,
    rerank_fetch_k: int = 0,
    hnsw_ef_search: int = 0,
) -> str:
    """Use when: You need a quick latency-vs-quality switch for RAG retrieval.

    Triggers: tune rag, faster retrieval, higher quality retrieval, rerank settings.
    Inputs:
      profile (str, optional): speed | balanced | quality | max_quality.
      min_similarity (float, optional): Override threshold; -1 keeps profile default.
      rerank_fetch_k (int, optional): Override fetch_k; 0 keeps profile default.
      hnsw_ef_search (int, optional): Override ANN search breadth; 0 keeps profile default.
    Returns: JSON with before/after tuning snapshots.
    Side effects: Updates runtime retrieval settings for this process.
    """
    try:
        presets: dict[str, dict[str, Any]] = {
            "speed": {
                "rerank_enabled": False,
                "rerank_fetch_k": 18,
                "min_similarity": 0.26,
                "hnsw_ef_search": 96,
            },
            "balanced": {
                "rerank_enabled": True,
                "rerank_fetch_k": 30,
                "min_similarity": 0.20,
                "hnsw_ef_search": 200,
            },
            "quality": {
                "rerank_enabled": True,
                "rerank_fetch_k": 48,
                "min_similarity": 0.15,
                "hnsw_ef_search": 320,
            },
            "max_quality": {
                "rerank_enabled": True,
                "rerank_fetch_k": 64,
                "min_similarity": 0.10,
                "hnsw_ef_search": 420,
            },
        }
        p = str(profile or "balanced").strip().lower()
        if p not in presets:
            return _safe_json(
                {
                    "status": "error",
                    "error": f"Unknown profile: {profile}",
                    "supported_profiles": sorted(presets.keys()),
                }
            )

        doc_db = _doc_db_from_agent()
        collection = doc_db.collection
        reranker = getattr(collection, "_reranker", None)

        before = _tuning_snapshot(doc_db)
        chosen = dict(presets[p])

        if min_similarity >= 0.0:
            chosen["min_similarity"] = float(max(0.0, min(min_similarity, 1.0)))
        if int(rerank_fetch_k) > 0:
            chosen["rerank_fetch_k"] = int(max(1, min(int(rerank_fetch_k), 200)))
        if int(hnsw_ef_search) > 0:
            chosen["hnsw_ef_search"] = int(max(8, min(int(hnsw_ef_search), 2000)))

        if hasattr(doc_db, "rerank_fetch_k"):
            doc_db.rerank_fetch_k = int(chosen["rerank_fetch_k"])
        if hasattr(doc_db, "rerank_enabled"):
            doc_db.rerank_enabled = bool(chosen["rerank_enabled"])
        if reranker is not None and hasattr(reranker, "enabled"):
            reranker.enabled = bool(chosen["rerank_enabled"])

        if hasattr(doc_db, "min_similarity"):
            doc_db.min_similarity = float(chosen["min_similarity"])
        if hasattr(collection, "_min_similarity"):
            collection._min_similarity = float(chosen["min_similarity"])
        if hasattr(doc_db, "hnsw_ef_search"):
            doc_db.hnsw_ef_search = int(chosen["hnsw_ef_search"])
        if hasattr(collection, "_ef_search"):
            collection._ef_search = int(chosen["hnsw_ef_search"])
        index = getattr(collection, "_index", None)
        if index is not None and hasattr(index, "set_ef"):
            try:
                index.set_ef(int(chosen["hnsw_ef_search"]))
            except Exception:
                pass
        cache = getattr(doc_db, "_query_cache", None)
        if isinstance(cache, dict):
            cache.clear()

        persist_state = getattr(collection, "_persist_state", None)
        if callable(persist_state):
            try:
                persist_state()
            except Exception:
                pass

        after = _tuning_snapshot(doc_db)
        return _safe_json(
            {
                "status": "ok",
                "profile": p,
                "applied": chosen,
                "before": before,
                "after": after,
                "note": "applies to current runtime; reindex is not required",
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def rag_benchmark(
    query: str,
    top_k: int = 20,
    runs: int = 4,
    ef_search: int = 0,
) -> str:
    """Use when: Measure RAG retrieval latency and validate tuning impact.

    Triggers: benchmark rag, rag latency, retrieval speed, compare profiles.
    Inputs:
      query (str, required): Query text.
      top_k (int, optional): Requested retrieval depth (default 20).
      runs (int, optional): Number of benchmark runs (default 4, max 20).
      ef_search (int, optional): Query-time ANN breadth override; 0 keeps current setting.
    Returns: JSON latency summary in milliseconds.
    Side effects: Read-only.
    """
    try:
        q = str(query or "").strip()
        if not q:
            return _safe_json({"status": "error", "error": "Query is empty"})

        n_runs = max(1, min(int(runs), 20))
        k = max(1, min(int(top_k), 100))
        query_ef = int(ef_search) if int(ef_search) > 0 else None

        doc_db = _doc_db_from_agent()
        latencies_ms: list[float] = []
        counts: list[int] = []

        for _ in range(n_runs):
            t0 = time.perf_counter()
            rows = doc_db.query(q, n_results=k, ef_search=query_ef)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(round(dt, 3))
            counts.append(len(rows))

        sorted_ms = sorted(latencies_ms)
        p50 = sorted_ms[len(sorted_ms) // 2]
        p95 = sorted_ms[min(len(sorted_ms) - 1, int(round((len(sorted_ms) - 1) * 0.95)))]

        return _safe_json(
            {
                "status": "ok",
                "query": q,
                "top_k": k,
                "runs": n_runs,
                "latency_ms": {
                    "avg": round(float(mean(latencies_ms)), 3),
                    "min": round(float(min(latencies_ms)), 3),
                    "p50": round(float(p50), 3),
                    "p95": round(float(p95), 3),
                    "max": round(float(max(latencies_ms)), 3),
                    "samples": latencies_ms,
                },
                "result_counts": counts,
                "tuning": _tuning_snapshot(doc_db),
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


__tools__ = [rag_tuning_status, rag_apply_profile, rag_benchmark]
