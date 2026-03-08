from __future__ import annotations

from typing import Any

import src.db.core as core


class _DummyHNSW:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _DummyUSEARCH:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FailingUSEARCH:
    def __init__(self, **_kwargs: Any) -> None:
        raise ImportError("usearch missing")


def _factory_kwargs() -> dict[str, Any]:
    return {
        "root_path": "/tmp",
        "collection_name": "x",
        "embedding_model_name": "model",
        "rerank_enabled": False,
        "reranker_model_name": "reranker",
    }


def test_factory_selects_usearch_when_available() -> None:
    old_h = core._HNSWCollection
    old_u = core._USEARCHCollection
    try:
        core._HNSWCollection = _DummyHNSW  # type: ignore[assignment]
        core._USEARCHCollection = _DummyUSEARCH  # type: ignore[assignment]
        coll = core.create_vector_collection(backend="usearch", **_factory_kwargs())
        assert isinstance(coll, _DummyUSEARCH)
    finally:
        core._HNSWCollection = old_h  # type: ignore[assignment]
        core._USEARCHCollection = old_u  # type: ignore[assignment]


def test_factory_falls_back_to_hnsw_when_usearch_missing() -> None:
    old_h = core._HNSWCollection
    old_u = core._USEARCHCollection
    try:
        core._HNSWCollection = _DummyHNSW  # type: ignore[assignment]
        core._USEARCHCollection = _FailingUSEARCH  # type: ignore[assignment]
        coll = core.create_vector_collection(backend="usearch", **_factory_kwargs())
        assert isinstance(coll, _DummyHNSW)
    finally:
        core._HNSWCollection = old_h  # type: ignore[assignment]
        core._USEARCHCollection = old_u  # type: ignore[assignment]


def test_factory_rejects_unknown_backend() -> None:
    try:
        core.create_vector_collection(backend="unknown", **_factory_kwargs())
        assert False, "Expected ValueError for unsupported backend"
    except ValueError:
        assert True
