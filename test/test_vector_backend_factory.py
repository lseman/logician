from __future__ import annotations

from typing import Any

import src.db.core as core
import src.db.backends.factory as factory
from src.db.backends.chromadb import _ChromaDBCollection
from src.db.backends.hnsw import _HNSWCollection
from src.db.backends.usearch import _USEARCHCollection


class _DummyHNSW:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _DummyUSEARCH:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _DummyChromaDB:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FailingUSEARCH:
    def __init__(self, **_kwargs: Any) -> None:
        raise ImportError("usearch missing")


class _FailingChromaDB:
    def __init__(self, **_kwargs: Any) -> None:
        raise ImportError("chromadb missing")


def _factory_kwargs() -> dict[str, Any]:
    return {
        "root_path": "/tmp",
        "collection_name": "x",
        "embedding_model_name": "model",
        "rerank_enabled": False,
        "reranker_model_name": "reranker",
    }


def test_factory_selects_usearch_when_available() -> None:
    old_c = factory._ChromaDBCollection
    old_h = factory._HNSWCollection
    old_u = factory._USEARCHCollection
    try:
        factory._ChromaDBCollection = _DummyChromaDB  # type: ignore[assignment]
        factory._HNSWCollection = _DummyHNSW  # type: ignore[assignment]
        factory._USEARCHCollection = _DummyUSEARCH  # type: ignore[assignment]
        coll = core.create_vector_collection(backend="usearch", **_factory_kwargs())
        assert isinstance(coll, _DummyUSEARCH)
    finally:
        factory._ChromaDBCollection = old_c  # type: ignore[assignment]
        factory._HNSWCollection = old_h  # type: ignore[assignment]
        factory._USEARCHCollection = old_u  # type: ignore[assignment]


def test_factory_selects_chromadb_when_available() -> None:
    old_c = factory._ChromaDBCollection
    old_h = factory._HNSWCollection
    old_u = factory._USEARCHCollection
    try:
        factory._ChromaDBCollection = _DummyChromaDB  # type: ignore[assignment]
        factory._HNSWCollection = _DummyHNSW  # type: ignore[assignment]
        factory._USEARCHCollection = _DummyUSEARCH  # type: ignore[assignment]
        coll = core.create_vector_collection(backend="chromadb", **_factory_kwargs())
        assert isinstance(coll, _DummyChromaDB)
    finally:
        factory._ChromaDBCollection = old_c  # type: ignore[assignment]
        factory._HNSWCollection = old_h  # type: ignore[assignment]
        factory._USEARCHCollection = old_u  # type: ignore[assignment]


def test_factory_raises_when_usearch_missing() -> None:
    old_c = factory._ChromaDBCollection
    old_h = factory._HNSWCollection
    old_u = factory._USEARCHCollection
    try:
        factory._ChromaDBCollection = _DummyChromaDB  # type: ignore[assignment]
        factory._HNSWCollection = _DummyHNSW  # type: ignore[assignment]
        factory._USEARCHCollection = _FailingUSEARCH  # type: ignore[assignment]
        try:
            core.create_vector_collection(backend="usearch", **_factory_kwargs())
            assert False, "Expected ImportError when USEARCH backend is unavailable"
        except ImportError:
            assert True
    finally:
        factory._ChromaDBCollection = old_c  # type: ignore[assignment]
        factory._HNSWCollection = old_h  # type: ignore[assignment]
        factory._USEARCHCollection = old_u  # type: ignore[assignment]


def test_factory_raises_when_chromadb_missing() -> None:
    old_c = factory._ChromaDBCollection
    old_h = factory._HNSWCollection
    old_u = factory._USEARCHCollection
    try:
        factory._ChromaDBCollection = _FailingChromaDB  # type: ignore[assignment]
        factory._HNSWCollection = _DummyHNSW  # type: ignore[assignment]
        factory._USEARCHCollection = _DummyUSEARCH  # type: ignore[assignment]
        try:
            core.create_vector_collection(backend="chromadb", **_factory_kwargs())
            assert False, "Expected ImportError when ChromaDB backend is unavailable"
        except ImportError:
            assert True
    finally:
        factory._ChromaDBCollection = old_c  # type: ignore[assignment]
        factory._HNSWCollection = old_h  # type: ignore[assignment]
        factory._USEARCHCollection = old_u  # type: ignore[assignment]


def test_factory_rejects_unknown_backend() -> None:
    try:
        core.create_vector_collection(backend="unknown", **_factory_kwargs())
        assert False, "Expected ValueError for unsupported backend"
    except ValueError:
        assert True


def test_hnsw_collection_init_defers_backend_import(tmp_path) -> None:
    import src.db.backends.hnsw as hnsw_mod

    old_lazy_import = hnsw_mod._lazy_import_hnswlib
    try:
        def _boom() -> Any:
            raise AssertionError("hnswlib should not load during collection init")

        hnsw_mod._lazy_import_hnswlib = _boom  # type: ignore[assignment]
        coll = _HNSWCollection(
            root_path=str(tmp_path),
            collection_name="messages",
            embedding_model_name="model",
            rerank_enabled=False,
            reranker_model_name="reranker",
        )
        assert coll._index is None
    finally:
        hnsw_mod._lazy_import_hnswlib = old_lazy_import  # type: ignore[assignment]


def test_usearch_collection_init_defers_backend_import(tmp_path) -> None:
    import src.db.backends.usearch as usearch_mod

    old_lazy_import = usearch_mod._lazy_import_usearch_index
    try:
        def _boom() -> Any:
            raise AssertionError("usearch should not load during collection init")

        usearch_mod._lazy_import_usearch_index = _boom  # type: ignore[assignment]
        coll = _USEARCHCollection(
            root_path=str(tmp_path),
            collection_name="messages",
            embedding_model_name="model",
            rerank_enabled=False,
            reranker_model_name="reranker",
        )
        assert coll._index is None
    finally:
        usearch_mod._lazy_import_usearch_index = old_lazy_import  # type: ignore[assignment]


def test_chromadb_collection_init_defers_backend_import(tmp_path) -> None:
    import src.db.backends.chromadb as chroma_mod

    old_lazy_import = chroma_mod._lazy_import_chromadb
    try:
        def _boom() -> Any:
            raise AssertionError("chromadb should not load during collection init")

        chroma_mod._lazy_import_chromadb = _boom  # type: ignore[assignment]
        coll = _ChromaDBCollection(
            root_path=str(tmp_path),
            collection_name="messages",
            embedding_model_name="model",
            rerank_enabled=False,
            reranker_model_name="reranker",
        )
        assert coll._collection is None
    finally:
        chroma_mod._lazy_import_chromadb = old_lazy_import  # type: ignore[assignment]
