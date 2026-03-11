from __future__ import annotations

import json
from pathlib import Path

from src.db.backends.chromadb import _ChromaDBCollection
from src.db.backends.hnsw import _HNSWCollection
from src.db.embeddings import _stable_collection_name


def test_chromadb_uses_backend_specific_dir_when_legacy_store_is_other_backend(
    tmp_path: Path,
) -> None:
    stable = _stable_collection_name("messages", "model")
    legacy_dir = tmp_path / stable
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "state.json").write_text(
        json.dumps({"backend": "usearch"}),
        encoding="utf-8",
    )

    coll = _ChromaDBCollection(
        root_path=str(tmp_path),
        collection_name="messages",
        embedding_model_name="model",
        rerank_enabled=False,
        reranker_model_name="reranker",
    )

    assert coll._dir == tmp_path / "chromadb" / stable


def test_hnsw_keeps_legacy_dir_when_legacy_store_matches_backend(tmp_path: Path) -> None:
    stable = _stable_collection_name("messages", "model")
    legacy_dir = tmp_path / stable
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "state.json").write_text(
        json.dumps({"backend": "hnsw"}),
        encoding="utf-8",
    )

    coll = _HNSWCollection(
        root_path=str(tmp_path),
        collection_name="messages",
        embedding_model_name="model",
        rerank_enabled=False,
        reranker_model_name="reranker",
    )

    assert coll._dir == legacy_dir
