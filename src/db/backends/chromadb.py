from __future__ import annotations

import json
import threading
from typing import Any, Callable

import numpy as np

from ...logging_utils import get_logger
from ..embeddings import (
    _EmbeddingRuntime,
    _lazy_import_chromadb,
    _RerankerRuntime,
    _resolve_vector_collection_dir,
)
from .hnsw import _HNSWCollection


class _ChromaDBCollection(_HNSWCollection):
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
        self._backend = "chromadb"
        self._embedder = _EmbeddingRuntime(embedding_model_name, self._log)
        self._reranker = _RerankerRuntime(
            reranker_model_name, rerank_enabled, self._log
        )

        self._dir = _resolve_vector_collection_dir(
            root_path=root_path,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            backend=self._backend,
        )
        self._dir.mkdir(parents=True, exist_ok=True)

        self._state_file = self._dir / "state.json"
        self._payload_file = self._dir / "payload.jsonl"
        self._persist_dir = self._dir / "chroma"

        self._space = self._normalize_space(space)
        self._ef_construction = int(ef_construction)
        self._m = int(m)
        self._ef_search = int(ef_search)
        self._min_similarity = float(max(0.0, min(1.0, min_similarity)))

        self._chromadb: Any | None = None
        self._client: Any | None = None
        self._collection: Any | None = None
        self._chroma_collection_name = "documents"

        self._lock = threading.RLock()
        self._payload_by_label: dict[int, dict[str, Any]] = {}
        self._id_to_label: dict[str, int] = {}
        self._next_label = 0
        self._dim: int | None = None
        self._max_elements = 0
        self._knn_filter_supported: bool | None = False

        with self._lock:
            self._load_payload()

    @staticmethod
    def _normalize_space(space: str) -> str:
        normalized = str(space or "").strip().lower()
        if normalized in ("cos", "cosine"):
            return "cosine"
        if normalized in ("ip", "inner_product", "dot"):
            return "ip"
        if normalized in ("l2", "euclidean"):
            return "l2"
        return "cosine"

    def _ensure_backend(self) -> Any:
        if self._chromadb is None:
            self._chromadb = _lazy_import_chromadb()
        return self._chromadb

    def _persist_state(self) -> None:
        state = {
            "backend": self._backend,
            "dim": self._dim,
            "next_label": self._next_label,
            "space": self._space,
            "ef_construction": self._ef_construction,
            "m": self._m,
            "ef_search": self._ef_search,
            "min_similarity": self._min_similarity,
            "embedding_model": self._embedder.resolved_model_name,
            "collection_name": self._chroma_collection_name,
        }
        self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _save(self) -> None:
        self._persist_payload()
        self._persist_state()

    def _create_collection(self) -> Any:
        assert self._client is not None
        return self._client.get_or_create_collection(
            name=self._chroma_collection_name,
            metadata={"hnsw:space": self._space},
        )

    def _reset_collection(self) -> Any:
        assert self._client is not None
        try:
            self._client.delete_collection(name=self._chroma_collection_name)
        except Exception:
            pass
        self._collection = self._create_collection()
        return self._collection

    def _ensure_index(self) -> None:
        if self._collection is not None:
            return

        state = {}
        if self._state_file.exists():
            try:
                state = json.loads(self._state_file.read_text(encoding="utf-8"))
            except Exception:
                state = {}

        if "dim" in state and state["dim"] is not None:
            self._dim = int(state["dim"])
        if "next_label" in state and state["next_label"]:
            self._next_label = max(self._next_label, int(state["next_label"]))

        saved_model = str(state.get("embedding_model", "") or "").strip()
        if saved_model:
            try:
                self._embedder.prefer_candidate(saved_model)
            except Exception:
                pass

        saved_backend = str(state.get("backend", "") or "").strip().lower()
        if saved_backend and saved_backend != self._backend:
            raise RuntimeError(
                f"Vector backend mismatch for {self._dir}: saved backend is "
                f"'{saved_backend}', runtime backend is '{self._backend}'. "
                "Use the matching backend or rebuild the collection."
            )

        saved_name = str(state.get("collection_name", "") or "").strip()
        if saved_name:
            self._chroma_collection_name = saved_name

        if self._dim is None:
            self._dim = self._embedder.dim()

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        chromadb = self._ensure_backend()
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._create_collection()

        expected_active = self._active_count()
        loaded_count = -1
        try:
            loaded_count = int(self._collection.count())
        except Exception:
            loaded_count = -1
        if loaded_count >= 0 and loaded_count != expected_active:
            self._log.warning(
                "Discarding stale Chroma collection (loaded_count=%s expected_active=%s). Rebuilding from payload.",
                loaded_count,
                expected_active,
            )
            self._reset_collection()
            if expected_active > 0:
                self._rebuild_index_from_payload()
            self._save()

    def _upsert_records(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: np.ndarray,
        labels: list[int],
    ) -> None:
        assert self._collection is not None
        kwargs = {
            "ids": ids,
            "documents": documents,
            "embeddings": np.asarray(embeddings, dtype=np.float32).tolist(),
            "metadatas": [{"_agent_label": int(label)} for label in labels],
        }
        upsert = getattr(self._collection, "upsert", None)
        if callable(upsert):
            upsert(**kwargs)
            return
        self._collection.delete(ids=ids)
        self._collection.add(**kwargs)

    def _delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        assert self._collection is not None
        self._collection.delete(ids=ids)

    def _rebuild_index_from_payload(self, *, batch_size: int = 256) -> None:
        assert self._collection is not None
        active_records = [
            rec
            for _label, rec in sorted(self._payload_by_label.items())
            if not rec.get("deleted", False)
        ]
        if not active_records:
            return

        self._log.info(
            "Rebuilding %s collection from payload: %d records",
            self._backend,
            len(active_records),
        )
        step = max(1, int(batch_size))
        for start in range(0, len(active_records), step):
            batch = active_records[start : start + step]
            docs = [str(rec.get("content", "")) for rec in batch]
            ids = [str(rec.get("id", "")) for rec in batch]
            labels = [int(rec.get("label", 0)) for rec in batch]
            vecs = self._embedder.embed_documents(docs)
            if vecs.dtype != np.float32:
                vecs = vecs.astype(np.float32, copy=False)
            self._upsert_records(
                ids=ids,
                documents=docs,
                embeddings=vecs,
                labels=labels,
            )

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
            event = {"stage": stage, "total": total, **payload}
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
            assert self._collection is not None

            provided_vecs: np.ndarray | None = None
            if embeddings is not None:
                provided_vecs = np.asarray(embeddings)
                if provided_vecs.dtype != np.float32:
                    provided_vecs = provided_vecs.astype(np.float32, copy=False)
                if provided_vecs.ndim != 2:
                    raise ValueError("embeddings must be 2D")
                if int(provided_vecs.shape[0]) != total:
                    raise ValueError("embeddings count must match documents length")

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

                latest_docs: dict[str, str] = {}
                latest_vecs: dict[str, np.ndarray] = {}
                latest_labels: dict[str, int] = {}

                for offset, ext_id in enumerate(ids_batch):
                    label = int(self._next_label)
                    self._next_label += 1

                    prev_label = self._id_to_label.get(ext_id)
                    if prev_label is not None:
                        prev = self._payload_by_label.get(prev_label)
                        if prev is not None and not prev.get("deleted", False):
                            prev["deleted"] = True

                    rec = {
                        "label": label,
                        "id": ext_id,
                        "content": docs_batch[offset],
                        "metadata": metas_batch[offset],
                        "deleted": False,
                    }
                    self._payload_by_label[label] = rec
                    self._id_to_label[ext_id] = label

                    latest_docs[ext_id] = docs_batch[offset]
                    latest_vecs[ext_id] = np.asarray(vecs[offset], dtype=np.float32)
                    latest_labels[ext_id] = label

                upsert_ids = list(latest_docs.keys())
                upsert_docs = [latest_docs[ext_id] for ext_id in upsert_ids]
                upsert_vecs = np.asarray(
                    [latest_vecs[ext_id] for ext_id in upsert_ids],
                    dtype=np.float32,
                )
                upsert_labels = [latest_labels[ext_id] for ext_id in upsert_ids]
                self._upsert_records(
                    ids=upsert_ids,
                    documents=upsert_docs,
                    embeddings=upsert_vecs,
                    labels=upsert_labels,
                )

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
            changed_ids: list[str] = []
            for label, rec in self._payload_by_label.items():
                if rec.get("deleted", False):
                    continue
                meta = rec.get("metadata", {}) or {}
                if not self._metadata_matches(meta, where):
                    continue
                rec["deleted"] = True
                ext_id = str(rec.get("id", ""))
                if self._id_to_label.get(ext_id) == label:
                    self._id_to_label.pop(ext_id, None)
                changed_ids.append(ext_id)

            if not changed_ids:
                return

            self._ensure_index()
            self._delete_ids(changed_ids)
            self._save()

    @staticmethod
    def _first_batch(value: Any) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if converted is None:
                return []
            value = converted
        if not isinstance(value, list):
            return [value]
        if not value:
            return []
        first = value[0]
        if isinstance(first, list):
            return first
        return value

    def _query_collection(
        self,
        *,
        qv: np.ndarray,
        n_results: int,
    ) -> tuple[list[str], list[float]]:
        assert self._collection is not None
        results = self._collection.query(
            query_embeddings=[np.asarray(qv, dtype=np.float32).reshape(-1).tolist()],
            n_results=max(1, int(n_results)),
            include=["distances"],
        )
        ids = [str(v) for v in self._first_batch(results.get("ids")) if str(v)]
        distances = [
            float(v) for v in self._first_batch(results.get("distances"))[: len(ids)]
        ]
        if len(distances) < len(ids):
            distances.extend([0.0] * (len(ids) - len(distances)))
        return ids, distances

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
        del ef_search
        include = include or ["documents", "metadatas"]
        top_k = max(1, int(n_results))

        with self._lock:
            self._ensure_index()
            assert self._collection is not None

            active_count = self._active_count()
            if active_count <= 0:
                return {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                }

            if query_embeddings is not None and len(query_embeddings) > 0:
                qv = np.asarray(query_embeddings[0], dtype=np.float32)
                query_text = query_texts[0] if query_texts else ""
            elif query_texts is not None and len(query_texts) > 0:
                query_text = str(query_texts[0])
                qv = self._embedder.embed_query(query_text)
            else:
                raise ValueError("Either query_texts or query_embeddings must be provided")

            requested = min(active_count, max(top_k, top_k * 4))
            ids_1, dists_1 = self._query_collection(qv=qv, n_results=requested)
            hits = self._collect_hits(
                np.asarray(ids_1, dtype=object),
                np.asarray(dists_1, dtype=np.float32),
                where=where,
            )

            if len(hits) < top_k and requested < active_count:
                ids_2, dists_2 = self._query_collection(qv=qv, n_results=active_count)
                hits = self._collect_hits(
                    np.asarray(ids_2, dtype=object),
                    np.asarray(dists_2, dtype=np.float32),
                    where=where,
                )

            if self._space == "cosine":
                hits = [
                    h
                    for h in hits
                    if (1.0 - float(h.get("distance", 1.0))) >= self._min_similarity
                ]

            if query_texts and query_texts[0]:
                hits = self._reranker.rerank(str(query_texts[0]), hits, top_k=top_k)
            else:
                hits = hits[:top_k]

            out_ids = [str(h["id"]) for h in hits]
            out_docs = [str(h["content"]) for h in hits]
            out_meta = [h["metadata"] for h in hits]
            out_dist = [float(h["distance"]) for h in hits]

            resp: dict[str, Any] = {"ids": [out_ids]}
            if "documents" in include:
                resp["documents"] = [out_docs]
            if "metadatas" in include:
                resp["metadatas"] = [out_meta]
            if "distances" in include:
                resp["distances"] = [out_dist]
            if "documents" not in resp:
                resp["documents"] = [out_docs]
            if "metadatas" not in resp:
                resp["metadatas"] = [out_meta]
            return resp

    def _collect_hits(
        self,
        ids: np.ndarray,
        distances: np.ndarray,
        *,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ext_id, dist in zip(ids.tolist(), distances.tolist()):
            label = self._id_to_label.get(str(ext_id))
            if label is None:
                continue
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


__all__ = ["_ChromaDBCollection"]
