from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np

from ...logging_utils import get_logger
from ..embeddings import (
    _EmbeddingRuntime,
    _lazy_import_usearch_index,
    _RerankerRuntime,
    _resolve_vector_collection_dir,
)
from .hnsw import _HNSWCollection


class _USEARCHCollection(_HNSWCollection):
    @staticmethod
    def _first_non_none_attr(obj: Any, names: tuple[str, ...]) -> Any:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
        return None

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
        self._backend = "usearch"
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
        self._index_file = self._dir / "index.usearch"

        self._space = space
        self._ef_construction = int(ef_construction)
        self._m = int(m)
        self._ef_search = int(ef_search)
        self._min_similarity = float(max(0.0, min(1.0, min_similarity)))

        self._usearch_index_cls: Any | None = None
        self._index: Any | None = None

        self._lock = threading.RLock()
        self._payload_by_label: dict[int, dict[str, Any]] = {}
        self._id_to_label: dict[str, int] = {}
        self._next_label = 0
        self._dim: int | None = None
        self._max_elements = 0
        self._knn_filter_supported: bool | None = False

        with self._lock:
            self._load_payload()

    def _ensure_backend(self) -> Any:
        if self._usearch_index_cls is None:
            self._usearch_index_cls = _lazy_import_usearch_index()
        return self._usearch_index_cls

    def _metric_name(self) -> str:
        space = str(self._space or "").strip().lower()
        if space in ("cos", "cosine"):
            return "cos"
        if space in ("ip", "dot", "inner_product"):
            return "ip"
        if space in ("l2", "euclidean"):
            return "l2sq"
        return "cos"

    def _new_index(self, *, dim: int, max_elements: int) -> Any:
        del max_elements
        metric = self._metric_name()
        index_cls = self._ensure_backend()
        options: list[dict[str, Any]] = [
            {
                "ndim": dim,
                "metric": metric,
                "connectivity": self._m,
                "expansion_add": self._ef_construction,
                "expansion_search": self._ef_search,
            },
            {"ndim": dim, "metric": metric, "connectivity": self._m},
            {"ndim": dim, "metric": metric},
            {"ndim": dim},
        ]
        for kwargs in options:
            try:
                return index_cls(**kwargs)
            except TypeError:
                continue
        return index_cls(dim)

    def _load_usearch_index(self, idx: Any) -> bool:
        if not self._index_file.exists():
            return False
        for method_name in ("load", "restore", "view"):
            method = getattr(idx, method_name, None)
            if not callable(method):
                continue
            for call in (
                lambda m=method: m(str(self._index_file)),
                lambda m=method: m(path=str(self._index_file)),
            ):
                try:
                    call()
                    return True
                except TypeError:
                    continue
                except Exception:
                    break
        return False

    def _save_usearch_index(self) -> None:
        assert self._index is not None
        for method_name in ("save", "dump"):
            method = getattr(self._index, method_name, None)
            if not callable(method):
                continue
            for call in (
                lambda m=method: m(str(self._index_file)),
                lambda m=method: m(path=str(self._index_file)),
            ):
                try:
                    call()
                    return
                except TypeError:
                    continue
        raise RuntimeError("USEARCH index does not expose a compatible save() method.")

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
        if "next_label" in state and state["next_label"]:
            self._next_label = max(self._next_label, int(state["next_label"]))
        if "max_elements" in state and state["max_elements"]:
            self._max_elements = int(state["max_elements"])

        saved_backend = str(state.get("backend", "") or "").strip().lower()
        if saved_backend and saved_backend != self._backend:
            raise RuntimeError(
                f"Vector backend mismatch for {self._dir}: saved backend is "
                f"'{saved_backend}', runtime backend is '{self._backend}'. "
                "Use the matching backend or rebuild the collection."
            )

        if self._dim is None:
            self._dim = self._embedder.dim()
        if self._max_elements <= 0:
            self._max_elements = max(1024, self._next_label + 128)

        idx = self._new_index(dim=self._dim, max_elements=self._max_elements)
        loaded = False
        if self._index_file.exists():
            loaded = self._load_usearch_index(idx)
            if not loaded:
                self._log.warning(
                    "USEARCH index file exists but could not be loaded; rebuilding: %s",
                    self._index_file.name,
                )
        self._index = idx
        if (not loaded) and self._active_count() > 0:
            self._rebuild_index_from_payload()
            self._save()

    def _save(self) -> None:
        self._save_usearch_index()
        self._persist_payload()
        self._persist_state()

    def _ensure_capacity(self, n_new: int) -> None:
        required = self._next_label + n_new + 1
        if required > self._max_elements:
            self._max_elements = required

    def _supports_knn_filter(self) -> bool:
        return False

    def _index_add_items(self, vecs: np.ndarray, labels: np.ndarray) -> None:
        assert self._index is not None
        calls = (
            lambda: self._index.add(labels, vecs),
            lambda: self._index.add(keys=labels, vectors=vecs),
            lambda: self._index.add(keys=labels, values=vecs),
            lambda: self._index.add(vecs, labels),
        )
        last_exc: Exception | None = None
        for call in calls:
            try:
                call()
                return
            except TypeError:
                continue
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("USEARCH index add() signature is unsupported.")

    def _index_mark_deleted(self, label: int) -> None:
        assert self._index is not None
        for method_name in ("remove", "delete"):
            method = getattr(self._index, method_name, None)
            if not callable(method):
                continue
            for call in (
                lambda m=method: m(int(label)),
                lambda m=method: m(np.asarray([int(label)], dtype=np.int64)),
                lambda m=method: m(keys=[int(label)]),
            ):
                try:
                    call()
                    return
                except TypeError:
                    continue
                except Exception:
                    continue

    def _search_usearch(self, qv: np.ndarray, *, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None
        query = np.asarray(qv, dtype=np.float32).reshape(-1)
        calls = (
            lambda: self._index.search(query, k),
            lambda: self._index.search(query, count=k),
            lambda: self._index.search(vector=query, count=k),
            lambda: self._index.search(vectors=query, count=k),
        )
        result: Any | None = None
        last_exc: Exception | None = None
        for call in calls:
            try:
                result = call()
                break
            except TypeError:
                continue
            except Exception as exc:
                last_exc = exc
                continue
        if result is None:
            if last_exc is not None:
                raise last_exc
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)

        keys: Any = None
        dists: Any = None
        if isinstance(result, tuple):
            if len(result) >= 1:
                keys = result[0]
            if len(result) >= 2:
                dists = result[1]
        else:
            keys = self._first_non_none_attr(
                result,
                ("keys", "ids", "labels"),
            )
            dists = self._first_non_none_attr(
                result,
                ("distances", "scores", "values"),
            )
            if keys is None and hasattr(result, "__iter__"):
                tmp_keys: list[int] = []
                tmp_dists: list[float] = []
                for item in result:
                    key = getattr(item, "key", None)
                    if key is None:
                        key = getattr(item, "id", None)
                    if key is None:
                        continue
                    tmp_keys.append(int(key))
                    dist = getattr(item, "distance", None)
                    if dist is None:
                        dist = getattr(item, "score", 0.0)
                    tmp_dists.append(float(dist))
                keys = tmp_keys
                dists = tmp_dists

        keys_arr = np.asarray(keys if keys is not None else [], dtype=np.int64).reshape(-1)
        if dists is None:
            dists_arr = np.zeros(keys_arr.shape[0], dtype=np.float32)
        else:
            dists_arr = np.asarray(dists, dtype=np.float32).reshape(-1)
        if dists_arr.shape[0] != keys_arr.shape[0]:
            n = min(dists_arr.shape[0], keys_arr.shape[0])
            keys_arr = keys_arr[:n]
            dists_arr = dists_arr[:n]
        return keys_arr[:k], dists_arr[:k]

    def _knn_query_with_optional_filter(
        self,
        qv: np.ndarray,
        *,
        k: int,
        allowed_labels: set[int] | None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        labels, distances = self._search_usearch(qv, k=k)
        if allowed_labels is None:
            return labels, distances, False
        if not allowed_labels:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32), True
        keep = np.asarray([int(lbl) in allowed_labels for lbl in labels], dtype=bool)
        return labels[keep], distances[keep], True


__all__ = ["_USEARCHCollection"]
