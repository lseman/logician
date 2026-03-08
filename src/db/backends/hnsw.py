from __future__ import annotations

import inspect
import json
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...logging_utils import get_logger
from ..embeddings import (
    _EmbeddingRuntime,
    _RerankerRuntime,
    _lazy_import_hnswlib,
    _stable_collection_name,
)


class _HNSWCollection:
    def __init__(
        self,
        *,
        root_path: str,
        collection_name: str,
        embedding_model_name: str,
        rerank_enabled: bool,
        reranker_model_name: str,
        space: str = "cosine",
        ef_construction: int = 200,
        m: int = 16,
        ef_search: int = 128,
        min_similarity: float = 0.18,
    ) -> None:
        self._log = get_logger("agent.vector")
        self._backend = "hnsw"
        self._embedder = _EmbeddingRuntime(embedding_model_name, self._log)
        self._reranker = _RerankerRuntime(
            reranker_model_name, rerank_enabled, self._log
        )

        stable_name = _stable_collection_name(collection_name, embedding_model_name)
        self._dir = Path(root_path) / stable_name
        self._dir.mkdir(parents=True, exist_ok=True)

        self._state_file = self._dir / "state.json"
        self._payload_file = self._dir / "payload.jsonl"
        self._index_file = self._dir / "index.bin"

        self._space = space
        self._ef_construction = int(ef_construction)
        self._m = int(m)
        self._ef_search = int(ef_search)
        self._min_similarity = float(max(0.0, min(1.0, min_similarity)))

        self._hnswlib = _lazy_import_hnswlib()
        self._index: Any | None = None

        self._lock = threading.RLock()
        self._payload_by_label: dict[int, dict[str, Any]] = {}
        self._id_to_label: dict[str, int] = {}
        self._next_label = 0
        self._dim: int | None = None
        self._max_elements = 0
        self._knn_filter_supported: bool | None = None

        self._load()

    def _new_index(self, *, dim: int, max_elements: int) -> Any:
        idx = self._hnswlib.Index(space=self._space, dim=dim)
        idx.init_index(
            max_elements=max_elements,
            ef_construction=self._ef_construction,
            M=self._m,
            allow_replace_deleted=True,
        )
        idx.set_ef(self._ef_search)
        return idx

    def _load_payload(self) -> None:
        if not self._payload_file.exists():
            return
        for line in self._payload_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = int(rec["label"])
            self._payload_by_label[label] = rec
            ext_id = str(rec["id"])
            if not rec.get("deleted", False):
                self._id_to_label[ext_id] = label
            self._next_label = max(self._next_label, label + 1)

    def _persist_payload(self) -> None:
        lines = []
        for label in sorted(self._payload_by_label.keys()):
            lines.append(json.dumps(self._payload_by_label[label], ensure_ascii=False))
        self._payload_file.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    def _persist_state(self) -> None:
        state = {
            "backend": self._backend,
            "dim": self._dim,
            "next_label": self._next_label,
            "max_elements": self._max_elements,
            "space": self._space,
            "ef_construction": self._ef_construction,
            "m": self._m,
            "ef_search": self._ef_search,
            "min_similarity": self._min_similarity,
            "embedding_model": self._embedder.resolved_model_name,
        }
        self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _ensure_index(self) -> None:
        if self._index is not None:
            return

        state = {}
        if self._state_file.exists():
            try:
                state = json.loads(self._state_file.read_text(encoding="utf-8"))
            except Exception:
                state = {}

        if "dim" in state and state["dim"] is not None:
            self._dim = int(state["dim"])
        if "max_elements" in state and state["max_elements"]:
            self._max_elements = int(state["max_elements"])
        if "next_label" in state and state["next_label"]:
            self._next_label = max(self._next_label, int(state["next_label"]))

        if self._dim is None:
            self._dim = self._embedder.dim()

        if self._max_elements <= 0:
            self._max_elements = max(1024, self._next_label + 128)

        loaded = False
        if self._index_file.exists():
            try:
                idx = self._hnswlib.Index(space=self._space, dim=self._dim)
                idx.load_index(str(self._index_file), max_elements=self._max_elements)
                idx.set_ef(self._ef_search)
                self._index = idx
                loaded = True
            except Exception as exc:
                self._log.warning(
                    "Failed to load HNSW index (%s), rebuilding from payload.",
                    exc,
                )

        if not loaded:
            self._index = self._new_index(
                dim=self._dim, max_elements=self._max_elements
            )
            if self._active_count() > 0:
                self._rebuild_index_from_payload()
                self._save()

    def _load(self) -> None:
        with self._lock:
            self._load_payload()
            self._ensure_index()

    def _save(self) -> None:
        assert self._index is not None
        self._index.save_index(str(self._index_file))
        self._persist_payload()
        self._persist_state()

    def _index_add_items(self, vecs: np.ndarray, labels: np.ndarray) -> None:
        assert self._index is not None
        self._index.add_items(vecs, labels)

    def _index_mark_deleted(self, label: int) -> None:
        assert self._index is not None
        self._index.mark_deleted(label)

    def _ensure_capacity(self, n_new: int) -> None:
        assert self._index is not None
        required = self._next_label + n_new + 1
        if required <= self._max_elements:
            return
        new_cap = max(required, int(self._max_elements * 1.5), 1024)
        self._index.resize_index(new_cap)
        self._max_elements = new_cap

    def _rebuild_index_from_payload(self, *, batch_size: int = 256) -> None:
        assert self._index is not None
        active_records = [
            (int(label), rec)
            for label, rec in sorted(self._payload_by_label.items())
            if not rec.get("deleted", False)
        ]
        if not active_records:
            return

        self._log.info(
            "Rebuilding %s index from payload: %d records",
            str(getattr(self, "_backend", "vector")),
            len(active_records),
        )
        step = max(1, int(batch_size))
        for start in range(0, len(active_records), step):
            batch = active_records[start : start + step]
            labels = np.asarray([label for label, _ in batch], dtype=np.int64)
            docs = [str(rec.get("content", "")) for _, rec in batch]
            vecs = self._embedder.embed_documents(docs)
            if vecs.dtype != np.float32:
                vecs = vecs.astype(np.float32, copy=False)
            self._index_add_items(vecs, labels)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, (list, tuple, set, frozenset)):
            return list(value)
        return [value]

    @classmethod
    def _metadata_op_match(
        cls,
        value: Any,
        op: str,
        expected: Any,
        *,
        key_exists: bool,
    ) -> bool:
        normalized = str(op or "").strip().lower()
        if normalized in ("$eq", "eq"):
            return value == expected
        if normalized in ("$ne", "ne"):
            return value != expected
        if normalized in ("$gt", "gt"):
            try:
                return value is not None and value > expected
            except Exception:
                return False
        if normalized in ("$gte", "gte"):
            try:
                return value is not None and value >= expected
            except Exception:
                return False
        if normalized in ("$lt", "lt"):
            try:
                return value is not None and value < expected
            except Exception:
                return False
        if normalized in ("$lte", "lte"):
            try:
                return value is not None and value <= expected
            except Exception:
                return False
        if normalized in ("$in", "in"):
            allowed = cls._as_list(expected)
            if isinstance(value, (list, tuple, set, frozenset)):
                return any(item in allowed for item in value)
            return value in allowed
        if normalized in ("$nin", "nin"):
            blocked = cls._as_list(expected)
            if isinstance(value, (list, tuple, set, frozenset)):
                return all(item not in blocked for item in value)
            return value not in blocked
        if normalized in ("$contains", "contains"):
            if isinstance(value, str):
                return str(expected) in value
            if isinstance(value, (list, tuple, set, frozenset)):
                return expected in value
            if isinstance(value, dict):
                return expected in value
            return False
        if normalized in ("$exists", "exists"):
            wants = bool(expected)
            return key_exists if wants else (not key_exists)
        return False

    @classmethod
    def _metadata_field_matches(cls, meta: dict[str, Any], key: str, expected: Any) -> bool:
        key_exists = key in meta
        value = meta.get(key)
        if isinstance(expected, dict):
            for op, operand in expected.items():
                if not cls._metadata_op_match(
                    value, str(op), operand, key_exists=key_exists
                ):
                    return False
            return True
        if isinstance(expected, (list, tuple, set, frozenset)):
            return cls._metadata_op_match(
                value,
                "$in",
                list(expected),
                key_exists=key_exists,
            )
        return value == expected

    @classmethod
    def _metadata_matches(cls, meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
        if not where:
            return True
        if not isinstance(where, dict):
            return False

        and_clauses = where.get("$and")
        if and_clauses is not None:
            if not isinstance(and_clauses, list):
                return False
            if any(not cls._metadata_matches(meta, clause) for clause in and_clauses):
                return False

        or_clauses = where.get("$or")
        if or_clauses is not None:
            if not isinstance(or_clauses, list) or not or_clauses:
                return False
            if not any(cls._metadata_matches(meta, clause) for clause in or_clauses):
                return False

        if "$not" in where and cls._metadata_matches(meta, where.get("$not")):
            return False

        for key, expected in where.items():
            if key in ("$and", "$or", "$not"):
                continue
            if not cls._metadata_field_matches(meta, key, expected):
                return False
        return True

    def _supports_knn_filter(self) -> bool:
        if self._knn_filter_supported is not None:
            return bool(self._knn_filter_supported)
        assert self._index is not None
        try:
            sig = inspect.signature(self._index.knn_query)
            self._knn_filter_supported = "filter" in sig.parameters
        except Exception:
            self._knn_filter_supported = False
        return bool(self._knn_filter_supported)

    def _prefilter_active_labels(self, where: dict[str, Any] | None) -> set[int] | None:
        if not where:
            return None
        labels: set[int] = set()
        for label, rec in self._payload_by_label.items():
            if rec.get("deleted", False):
                continue
            meta = rec.get("metadata", {}) or {}
            if self._metadata_matches(meta, where):
                labels.add(int(label))
        return labels

    def _knn_query_with_optional_filter(
        self,
        qv: np.ndarray,
        *,
        k: int,
        allowed_labels: set[int] | None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        assert self._index is not None
        if allowed_labels is None:
            labels, distances = self._index.knn_query(qv, k=k)
            return labels[0], distances[0], False
        if not allowed_labels:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32), True
        if not self._supports_knn_filter():
            labels, distances = self._index.knn_query(qv, k=k)
            return labels[0], distances[0], False

        try:
            labels, distances = self._index.knn_query(
                qv,
                k=k,
                filter=lambda label: int(label) in allowed_labels,
            )
            return labels[0], distances[0], True
        except TypeError:
            self._knn_filter_supported = False
            labels, distances = self._index.knn_query(qv, k=k)
            return labels[0], distances[0], False
        except Exception as exc:
            self._log.debug(
                "HNSW query-time metadata filter unavailable; using post-filter: %s",
                exc,
            )
            labels, distances = self._index.knn_query(qv, k=k)
            return labels[0], distances[0], False

    def add(
        self,
        *,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
        embeddings: list[list[float]] | None = None,
        progress_callback: Callable[..., None] | None = None,
        batch_size: int = 512,
    ) -> None:
        if not documents:
            return
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError("documents, metadatas, ids must be same length")

        total = len(documents)
        step = max(1, int(batch_size))
        batch_count = (total + step - 1) // step

        def _emit_progress(stage: str, **payload: Any) -> None:
            if progress_callback is None:
                return
            event = {
                "stage": stage,
                "total": total,
                **payload,
            }
            try:
                progress_callback(event)
                return
            except TypeError:
                pass
            except Exception as exc:
                self._log.debug("add progress_callback failed: %s", exc)
                return
            try:
                progress_callback(**event)
            except Exception as exc:
                self._log.debug("add progress_callback failed: %s", exc)

        _emit_progress("start", added=0, batch_index=0, batch_count=batch_count)

        with self._lock:
            self._ensure_index()
            assert self._index is not None

            provided_vecs: np.ndarray | None = None
            if embeddings is not None:
                provided_vecs = np.asarray(embeddings)
                if provided_vecs.dtype != np.float32:
                    provided_vecs = provided_vecs.astype(np.float32, copy=False)
                if provided_vecs.ndim != 2:
                    raise ValueError("embeddings must be 2D")
                if int(provided_vecs.shape[0]) != total:
                    raise ValueError("embeddings count must match documents length")

            self._ensure_capacity(total)
            added = 0
            for batch_index, start in enumerate(range(0, total, step), start=1):
                end = min(total, start + step)
                docs_batch = documents[start:end]
                metas_batch = metadatas[start:end]
                ids_batch = ids[start:end]
                batch_len = len(docs_batch)

                if provided_vecs is None:
                    vecs = self._embedder.embed_documents(docs_batch)
                    if vecs.dtype != np.float32:
                        vecs = vecs.astype(np.float32, copy=False)
                    if vecs.ndim != 2:
                        raise ValueError("embeddings must be 2D")
                    if int(vecs.shape[0]) != batch_len:
                        raise ValueError("embedder returned unexpected vector count")
                else:
                    vecs = provided_vecs[start:end]

                labels = np.arange(
                    self._next_label, self._next_label + batch_len, dtype=np.int64
                )
                self._next_label += batch_len

                self._index_add_items(vecs, labels)

                for i, label in enumerate(labels.tolist()):
                    ext_id = ids_batch[i]
                    if ext_id in self._id_to_label:
                        # Mark previous record deleted so latest wins.
                        old = self._id_to_label[ext_id]
                        prev = self._payload_by_label.get(old)
                        if prev is not None and not prev.get("deleted", False):
                            prev["deleted"] = True
                            try:
                                self._index_mark_deleted(old)
                            except Exception:
                                pass

                    rec = {
                        "label": label,
                        "id": ext_id,
                        "content": docs_batch[i],
                        "metadata": metas_batch[i],
                        "deleted": False,
                    }
                    self._payload_by_label[label] = rec
                    self._id_to_label[ext_id] = label

                added += batch_len
                _emit_progress(
                    "batch",
                    added=added,
                    batch_index=batch_index,
                    batch_count=batch_count,
                    batch_size=batch_len,
                )

            self._save()
        _emit_progress(
            "done",
            added=total,
            batch_index=batch_count,
            batch_count=batch_count,
        )

    def delete(self, *, where: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._ensure_index()
            assert self._index is not None

            changed = False
            for label, rec in self._payload_by_label.items():
                if rec.get("deleted", False):
                    continue
                meta = rec.get("metadata", {}) or {}
                if not self._metadata_matches(meta, where):
                    continue
                rec["deleted"] = True
                ext_id = str(rec.get("id", ""))
                if ext_id in self._id_to_label and self._id_to_label[ext_id] == label:
                    del self._id_to_label[ext_id]
                try:
                    self._index_mark_deleted(label)
                except Exception:
                    pass
                changed = True

            if changed:
                self._save()

    def _active_count(self) -> int:
        return sum(
            0 if r.get("deleted", False) else 1 for r in self._payload_by_label.values()
        )

    @staticmethod
    def _normalize_get_include(include: list[str] | None) -> list[str]:
        requested = include or ["documents", "metadatas"]
        normalized: list[str] = []
        for name in requested:
            key = str(name or "").strip().lower()
            if key not in ("documents", "metadatas"):
                continue
            if key not in normalized:
                normalized.append(key)
        return normalized

    def _records_for_get(
        self,
        *,
        ids: list[str] | None,
        where: dict[str, Any] | None,
        include_deleted: bool,
    ) -> list[dict[str, Any]]:
        if ids is not None and (not include_deleted):
            selected: list[dict[str, Any]] = []
            for ext_id in (str(v) for v in ids if str(v)):
                label = self._id_to_label.get(ext_id)
                if label is None:
                    continue
                rec = self._payload_by_label.get(label)
                if rec is None or rec.get("deleted", False):
                    continue
                meta = rec.get("metadata", {}) or {}
                if not self._metadata_matches(meta, where):
                    continue
                selected.append(rec)
            return selected

        id_filter = {str(v) for v in ids} if ids is not None else None
        selected: list[dict[str, Any]] = []
        for label in sorted(self._payload_by_label.keys()):
            rec = self._payload_by_label.get(label)
            if rec is None:
                continue
            if (not include_deleted) and rec.get("deleted", False):
                continue
            if id_filter is not None and str(rec.get("id", "")) not in id_filter:
                continue
            meta = rec.get("metadata", {}) or {}
            if not self._metadata_matches(meta, where):
                continue
            selected.append(rec)
        return selected

    def count(
        self,
        *,
        where: dict[str, Any] | None = None,
        include_deleted: bool = False,
    ) -> int:
        """Count records matching optional metadata filter."""
        with self._lock:
            if not include_deleted and where is None:
                return int(self._active_count())
            count = 0
            for rec in self._payload_by_label.values():
                if (not include_deleted) and rec.get("deleted", False):
                    continue
                meta = rec.get("metadata", {}) or {}
                if not self._metadata_matches(meta, where):
                    continue
                count += 1
            return int(count)

    def get(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int = 0,
        include: list[str] | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """Return stored records by id/metadata filter with pagination.

        Supported include fields: ``documents`` and ``metadatas``.
        ``ids`` are always returned.
        """
        with self._lock:
            include_norm = self._normalize_get_include(include)
            offset_n = max(0, int(offset))
            limit_n = None if limit is None else max(0, int(limit))

            rows = self._records_for_get(
                ids=ids,
                where=where,
                include_deleted=bool(include_deleted),
            )
            if offset_n:
                rows = rows[offset_n:]
            if limit_n is not None:
                rows = rows[:limit_n]

            resp: dict[str, Any] = {
                "ids": [str(r.get("id", "")) for r in rows],
            }
            if "documents" in include_norm:
                resp["documents"] = [str(r.get("content", "")) for r in rows]
            if "metadatas" in include_norm:
                resp["metadatas"] = [r.get("metadata", {}) or {} for r in rows]
            return resp

    def peek(
        self,
        *,
        limit: int = 10,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return first ``limit`` non-deleted records in insertion order."""
        return self.get(limit=max(1, int(limit)), include=include)

    def _collect_hits(
        self,
        labels: np.ndarray,
        distances: np.ndarray,
        *,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for label, dist in zip(labels.tolist(), distances.tolist()):
            rec = self._payload_by_label.get(int(label))
            if rec is None or rec.get("deleted", False):
                continue
            meta = rec.get("metadata", {}) or {}
            if not self._metadata_matches(meta, where):
                continue
            out.append(
                {
                    "id": rec.get("id"),
                    "content": rec.get("content", ""),
                    "metadata": meta,
                    "distance": float(dist),
                }
            )
        return out

    def query(
        self,
        *,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
        ef_search: int | None = None,
    ) -> dict[str, Any]:
        include = include or ["documents", "metadatas"]
        top_k = max(1, int(n_results))

        with self._lock:
            self._ensure_index()
            assert self._index is not None

            restore_ef = False
            if ef_search is not None:
                try:
                    override = max(8, int(ef_search))
                    self._index.set_ef(override)
                    restore_ef = True
                except Exception:
                    restore_ef = False

            try:
                prefiltered_labels = self._prefilter_active_labels(where)
                can_prefilter = bool(
                    prefiltered_labels is not None and self._supports_knn_filter()
                )
                if prefiltered_labels is not None and not prefiltered_labels:
                    active = 0
                elif can_prefilter:
                    active = len(prefiltered_labels)
                else:
                    active = self._active_count()
                if active == 0:
                    return {
                        "ids": [[]],
                        "documents": [[]],
                        "metadatas": [[]],
                        "distances": [[]],
                    }

                if query_embeddings is not None and len(query_embeddings) > 0:
                    qv = np.asarray(query_embeddings[0])
                    if qv.dtype != np.float32:
                        qv = qv.astype(np.float32, copy=False)
                    query_text = query_texts[0] if query_texts else ""
                elif query_texts is not None and len(query_texts) > 0:
                    query_text = str(query_texts[0])
                    qv = self._embedder.embed_query(query_text)
                else:
                    raise ValueError(
                        "Either query_texts or query_embeddings must be provided"
                    )

                # Retrieve wider then filter/rerank.
                k1 = min(active, max(top_k, top_k * 4))
                labels, distances, prefiltered = self._knn_query_with_optional_filter(
                    qv,
                    k=k1,
                    allowed_labels=prefiltered_labels if can_prefilter else None,
                )
                hits = self._collect_hits(
                    labels,
                    distances,
                    where=None if prefiltered else where,
                )

                if len(hits) < top_k and k1 < active:
                    labels2, distances2, prefiltered2 = self._knn_query_with_optional_filter(
                        qv,
                        k=active,
                        allowed_labels=prefiltered_labels if can_prefilter else None,
                    )
                    hits = self._collect_hits(
                        labels2,
                        distances2,
                        where=None if prefiltered2 else where,
                    )

                # For cosine distance with normalized vectors, similarity ~= 1 - distance.
                if self._space == "cosine":
                    hits = [
                        h
                        for h in hits
                        if (1.0 - float(h.get("distance", 1.0))) >= self._min_similarity
                    ]

                # Rerank only when we have text query.
                if query_texts and query_texts[0]:
                    hits = self._reranker.rerank(str(query_texts[0]), hits, top_k=top_k)
                else:
                    hits = hits[:top_k]

                out_ids = [h["id"] for h in hits]
                out_docs = [h["content"] for h in hits]
                out_meta = [h["metadata"] for h in hits]
                out_dist = [h["distance"] for h in hits]

                resp: dict[str, Any] = {}
                if "ids" in include or True:
                    resp["ids"] = [out_ids]
                if "documents" in include:
                    resp["documents"] = [out_docs]
                if "metadatas" in include:
                    resp["metadatas"] = [out_meta]
                if "distances" in include:
                    resp["distances"] = [out_dist]

                # Keep fields available for compatibility.
                if "documents" not in resp:
                    resp["documents"] = [out_docs]
                if "metadatas" not in resp:
                    resp["metadatas"] = [out_meta]
                return resp
            finally:
                if restore_ef:
                    try:
                        self._index.set_ef(self._ef_search)
                    except Exception:
                        pass


__all__ = ["_HNSWCollection"]
