# agent_core/db.py
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import get_logger
from .messages import Message, MessageRole


# ============================================================================
# Helpers: optional imports + model handling
# ============================================================================
def _lazy_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer
    except Exception as e:
        raise ImportError(
            "sentence-transformers not installed or failed to import. "
            "Run: pip install sentence-transformers"
        ) from e


def _lazy_import_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        return CrossEncoder
    except Exception as e:
        raise ImportError(
            "CrossEncoder not available from sentence-transformers. "
            "Run: pip install sentence-transformers"
        ) from e


def _lazy_import_hnswlib():
    try:
        import hnswlib  # type: ignore

        return hnswlib
    except Exception as e:
        raise ImportError(
            "hnswlib not installed or failed to import. Run: pip install hnswlib"
        ) from e


def _lazy_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _lazy_import_bitsandbytes_config():
    try:
        from transformers import BitsAndBytesConfig  # type: ignore

        return BitsAndBytesConfig
    except Exception:
        return None


def _resolve_model_load_kwargs(
    *,
    quant_mode_env: str,
    force_cuda_env: str,
) -> tuple[dict[str, Any], str]:
    """
    Build optional kwargs for SentenceTransformer/CrossEncoder constructors.
    Quantization is opt-in via env vars:
      - quant_mode_env: off|8bit|4bit
      - force_cuda_env: 1/true/yes to force cuda device
    """
    kwargs: dict[str, Any] = {}
    notes: list[str] = []

    torch_mod = _lazy_import_torch()
    has_cuda = bool(torch_mod is not None and torch_mod.cuda.is_available())
    force_cuda = os.getenv(force_cuda_env, "0").strip().lower() in ("1", "true", "yes")
    device = "cuda" if (has_cuda or force_cuda) else "cpu"
    kwargs["device"] = device
    notes.append(f"device={device}")

    qmode = os.getenv(quant_mode_env, "off").strip().lower()
    if qmode in ("8bit", "4bit"):
        BitsAndBytesConfig = _lazy_import_bitsandbytes_config()
        if BitsAndBytesConfig is not None:
            try:
                if qmode == "8bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    bnb_kwargs: dict[str, Any] = {"load_in_4bit": True}
                    if torch_mod is not None and hasattr(torch_mod, "float16"):
                        bnb_kwargs["bnb_4bit_compute_dtype"] = torch_mod.float16
                    kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
                notes.append(f"quant={qmode}")
            except Exception:
                notes.append(f"quant={qmode}_build_failed")
        else:
            notes.append(f"quant={qmode}_transformers_missing")
    else:
        notes.append("quant=off")

    return kwargs, ", ".join(notes)


def _embedding_candidates(name: str) -> list[str]:
    parts = [p.strip() for p in str(name).split("|")]
    return [p for p in parts if p]


def _prepare_embedding_input(model_name: str, text: str, *, for_query: bool) -> str:
    lower = model_name.lower()
    if "nomic-embed-text" in lower:
        prefix = "search_query: " if for_query else "search_document: "
        return prefix + text
    return text


def _stable_collection_name(base: str, embedding_model_name: str) -> str:
    primary = _embedding_candidates(embedding_model_name)
    model_key = primary[0] if primary else str(embedding_model_name)
    model_slug = model_key.replace("/", "_").replace("-", "_")
    digest = hashlib.sha256(model_key.encode("utf-8")).hexdigest()[:12]
    return f"{base}__{model_slug}__{digest}"


# ============================================================================
# Embedding + reranking wrappers
# ============================================================================
class _EmbeddingRuntime:
    _GLOBAL_ENCODERS: dict[str, Any] = {}
    _GLOBAL_LOCK = threading.RLock()

    def __init__(self, embedding_model_name: str, log: Any) -> None:
        self._model_name_raw = embedding_model_name
        self._log = log
        self._encoder: Any | None = None
        self._resolved_model_name: str | None = None
        self._lock = threading.RLock()

    @property
    def resolved_model_name(self) -> str:
        return self._resolved_model_name or self._model_name_raw

    def _ensure_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder

        SentenceTransformer = _lazy_import_sentence_transformers()
        last_err: Exception | None = None
        for candidate in _embedding_candidates(self._model_name_raw):
            with self._GLOBAL_LOCK:
                cached = self._GLOBAL_ENCODERS.get(candidate)
            if cached is not None:
                self._encoder = cached
                self._resolved_model_name = candidate
                self._log.info("Reusing cached embedding model: %s", candidate)
                return cached

            try:
                # Lock around load to avoid duplicate model initialization
                # when multiple DB runtimes start concurrently.
                with self._GLOBAL_LOCK:
                    cached2 = self._GLOBAL_ENCODERS.get(candidate)
                    if cached2 is not None:
                        self._encoder = cached2
                        self._resolved_model_name = candidate
                        self._log.info("Reusing cached embedding model: %s", candidate)
                        return cached2

                    self._log.info("Loading embedding model: %s", candidate)
                    kwargs, mode_note = _resolve_model_load_kwargs(
                        quant_mode_env="AGENT_EMBED_QUANT",
                        force_cuda_env="AGENT_FORCE_CUDA",
                    )
                    self._log.info("Embedding load kwargs: %s", mode_note)
                    try:
                        enc = SentenceTransformer(candidate, **kwargs)
                    except TypeError:
                        # Some sentence-transformers versions may not accept quantization_config.
                        if "quantization_config" in kwargs:
                            self._log.warning(
                                "Embedding quantization kwargs unsupported; retrying without quantization."
                            )
                            kwargs = {
                                k: v
                                for k, v in kwargs.items()
                                if k != "quantization_config"
                            }
                        enc = SentenceTransformer(candidate, **kwargs)
                    self._GLOBAL_ENCODERS[candidate] = enc

                self._encoder = enc
                self._resolved_model_name = candidate
                return enc
            except Exception as e:
                last_err = e
                self._log.warning("Embedding model failed %s: %s", candidate, e)

        if last_err is not None:
            raise last_err
        raise ValueError("No embedding model candidate provided.")

    def dim(self) -> int:
        with self._lock:
            enc = self._ensure_encoder()
            sample = _prepare_embedding_input(
                self.resolved_model_name, "hello", for_query=False
            )
            vec = enc.encode([sample], normalize_embeddings=True)[0]
            return int(np.asarray(vec).shape[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            enc = self._ensure_encoder()
            prepared = [
                _prepare_embedding_input(self.resolved_model_name, t, for_query=False)
                for t in texts
            ]
            vecs = np.asarray(enc.encode(prepared, normalize_embeddings=True))
            if vecs.dtype != np.float32:
                vecs = vecs.astype(np.float32, copy=False)
            return vecs

    def embed_query(self, query: str) -> np.ndarray:
        with self._lock:
            enc = self._ensure_encoder()
            q = _prepare_embedding_input(
                self.resolved_model_name, query, for_query=True
            )
            vec = np.asarray(enc.encode([q], normalize_embeddings=True)[0])
            if vec.dtype != np.float32:
                vec = vec.astype(np.float32, copy=False)
            return vec


class _RerankerRuntime:
    def __init__(self, model_name: str, enabled: bool, log: Any) -> None:
        self.model_name = model_name
        self.enabled = bool(enabled)
        self._log = log
        self._model: Any | None = None
        self._lock = threading.RLock()

    def _ensure(self) -> Any | None:
        if not self.enabled:
            return None
        if self._model is not None:
            return self._model
        try:
            CrossEncoder = _lazy_import_cross_encoder()
            self._log.info("Loading reranker: %s", self.model_name)
            kwargs, mode_note = _resolve_model_load_kwargs(
                quant_mode_env="AGENT_RERANK_QUANT",
                force_cuda_env="AGENT_FORCE_CUDA",
            )
            self._log.info("Reranker load kwargs: %s", mode_note)
            try:
                self._model = CrossEncoder(self.model_name, **kwargs)
            except TypeError:
                if "quantization_config" in kwargs:
                    self._log.warning(
                        "Reranker quantization kwargs unsupported; retrying without quantization."
                    )
                    kwargs = {
                        k: v for k, v in kwargs.items() if k != "quantization_config"
                    }
                self._model = CrossEncoder(self.model_name, **kwargs)
            return self._model
        except Exception as e:
            self._log.warning("Reranker unavailable (%s): %s", self.model_name, e)
            self._model = None
            return None

    def rerank(
        self,
        query: str,
        rows: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        model = self._ensure()
        if model is None or not rows:
            return rows[:top_k]
        with self._lock:
            try:
                pairs = [[query, r.get("content", "")] for r in rows]
                scores = model.predict(pairs)
                ranked = sorted(
                    enumerate(rows), key=lambda x: float(scores[x[0]]), reverse=True
                )
                out: list[dict[str, Any]] = []
                for idx, row in ranked[:top_k]:
                    out.append({**row, "rerank_score": float(scores[idx])})
                return out
            except Exception as e:
                self._log.warning("Reranker predict failed: %s", e)
                return rows[:top_k]


# ============================================================================
# Local persistent HNSW collection (Chroma-like add/query/delete surface)
# ============================================================================
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
        ef_construction: int = 256,
        m: int = 24,
        ef_search: int = 200,
        min_similarity: float = 0.18,
    ) -> None:
        self._log = get_logger("agent.vector")
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

        if self._index_file.exists():
            idx = self._hnswlib.Index(space=self._space, dim=self._dim)
            idx.load_index(str(self._index_file), max_elements=self._max_elements)
            idx.set_ef(self._ef_search)
            self._index = idx
        else:
            self._index = self._new_index(
                dim=self._dim, max_elements=self._max_elements
            )

    def _load(self) -> None:
        with self._lock:
            self._load_payload()
            self._ensure_index()

    def _save(self) -> None:
        assert self._index is not None
        self._index.save_index(str(self._index_file))
        self._persist_payload()
        self._persist_state()

    def _ensure_capacity(self, n_new: int) -> None:
        assert self._index is not None
        required = self._next_label + n_new + 1
        if required <= self._max_elements:
            return
        new_cap = max(required, int(self._max_elements * 1.5), 1024)
        self._index.resize_index(new_cap)
        self._max_elements = new_cap

    @staticmethod
    def _metadata_matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
        if not where:
            return True
        for k, v in where.items():
            if meta.get(k) != v:
                return False
        return True

    def add(
        self,
        *,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        if not documents:
            return
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError("documents, metadatas, ids must be same length")

        with self._lock:
            self._ensure_index()
            assert self._index is not None

            vecs = (
                np.asarray(embeddings)
                if embeddings is not None
                else self._embedder.embed_documents(documents)
            )
            if vecs.dtype != np.float32:
                vecs = vecs.astype(np.float32, copy=False)
            if vecs.ndim != 2:
                raise ValueError("embeddings must be 2D")

            self._ensure_capacity(len(documents))
            labels = np.arange(
                self._next_label, self._next_label + len(documents), dtype=np.int64
            )
            self._next_label += len(documents)

            self._index.add_items(vecs, labels)

            for i, label in enumerate(labels.tolist()):
                ext_id = ids[i]
                if ext_id in self._id_to_label:
                    # Mark previous record deleted so latest wins.
                    old = self._id_to_label[ext_id]
                    prev = self._payload_by_label.get(old)
                    if prev is not None and not prev.get("deleted", False):
                        prev["deleted"] = True
                        try:
                            self._index.mark_deleted(old)
                        except Exception:
                            pass

                rec = {
                    "label": label,
                    "id": ext_id,
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "deleted": False,
                }
                self._payload_by_label[label] = rec
                self._id_to_label[ext_id] = label

            self._save()

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
                    self._index.mark_deleted(label)
                except Exception:
                    pass
                changed = True

            if changed:
                self._save()

    def _active_count(self) -> int:
        return sum(
            0 if r.get("deleted", False) else 1 for r in self._payload_by_label.values()
        )

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
    ) -> dict[str, Any]:
        include = include or ["documents", "metadatas"]
        top_k = max(1, int(n_results))

        with self._lock:
            self._ensure_index()
            assert self._index is not None

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
            labels, distances = self._index.knn_query(qv, k=k1)
            hits = self._collect_hits(labels[0], distances[0], where=where)

            if len(hits) < top_k and k1 < active:
                labels2, distances2 = self._index.knn_query(qv, k=active)
                hits = self._collect_hits(labels2[0], distances2[0], where=where)

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


# ============================================================================
# SQLite tuning knobs
# ============================================================================
_SQLITE_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA temp_store=MEMORY;",
    "PRAGMA foreign_keys=ON;",
    "PRAGMA cache_size=-65536;",  # ~64MB
)
