from __future__ import annotations

from .chromadb import _ChromaDBCollection
from .hnsw import _HNSWCollection
from .usearch import _USEARCHCollection


def create_vector_collection(
    *,
    backend: str = "",
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
) -> _HNSWCollection:
    backend_norm = str(backend or "").strip().lower()
    if not backend_norm:
        raise ValueError(
            "No vector backend configured. Set Config.vector_backend or pass a backend explicitly."
        )
    if backend_norm not in ("chromadb", "hnsw", "usearch"):
        raise ValueError(
            f"Unsupported vector backend '{backend}'. Expected one of: chromadb, hnsw, usearch."
        )

    kwargs = {
        "root_path": root_path,
        "collection_name": collection_name,
        "embedding_model_name": embedding_model_name,
        "rerank_enabled": rerank_enabled,
        "reranker_model_name": reranker_model_name,
        "space": space,
        "ef_construction": ef_construction,
        "m": m,
        "ef_search": ef_search,
        "min_similarity": min_similarity,
    }

    if backend_norm == "chromadb":
        return _ChromaDBCollection(**kwargs)
    if backend_norm == "hnsw":
        return _HNSWCollection(**kwargs)

    return _USEARCHCollection(**kwargs)


__all__ = ["create_vector_collection"]
