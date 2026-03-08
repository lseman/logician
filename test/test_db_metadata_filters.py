from __future__ import annotations

from src.db.core import _HNSWCollection


def test_metadata_matches_equality_back_compat() -> None:
    meta = {"source": "orders.csv", "chunk": 4, "lang": "en"}
    assert _HNSWCollection._metadata_matches(meta, {"source": "orders.csv"})
    assert not _HNSWCollection._metadata_matches(meta, {"source": "users.csv"})


def test_metadata_matches_range_operators() -> None:
    meta = {"chunk": 4, "score": 0.87}
    assert _HNSWCollection._metadata_matches(
        meta, {"chunk": {"$gte": 3, "$lt": 10}}
    )
    assert _HNSWCollection._metadata_matches(
        meta, {"score": {"$gt": 0.8, "$lte": 0.9}}
    )
    assert not _HNSWCollection._metadata_matches(meta, {"chunk": {"$gt": 4}})


def test_metadata_matches_in_and_nin() -> None:
    meta = {"source": "docs/a.md", "tags": ["rag", "python"]}
    assert _HNSWCollection._metadata_matches(
        meta, {"source": {"$in": ["docs/a.md", "docs/b.md"]}}
    )
    assert _HNSWCollection._metadata_matches(
        meta, {"source": ["docs/a.md", "docs/b.md"]}
    )
    assert _HNSWCollection._metadata_matches(meta, {"tags": {"$in": ["java", "rag"]}})
    assert _HNSWCollection._metadata_matches(
        meta, {"tags": {"$nin": ["java", "go"]}}
    )
    assert not _HNSWCollection._metadata_matches(
        meta, {"source": {"$nin": ["docs/a.md"]}}
    )


def test_metadata_matches_logical_clauses() -> None:
    meta = {"source": "docs/a.md", "chunk": 3, "lang": "en"}
    where = {
        "$and": [
            {"chunk": {"$gte": 2}},
            {"$or": [{"source": "docs/b.md"}, {"source": "docs/a.md"}]},
        ],
        "$not": {"lang": "pt"},
    }
    assert _HNSWCollection._metadata_matches(meta, where)


def test_metadata_matches_exists_and_contains() -> None:
    meta = {"path": "src/db/core.py", "tags": ["rag", "vector"]}
    assert _HNSWCollection._metadata_matches(
        meta, {"path": {"$contains": "db/core"}}
    )
    assert _HNSWCollection._metadata_matches(meta, {"tags": {"$contains": "rag"}})
    assert _HNSWCollection._metadata_matches(meta, {"missing": {"$exists": False}})
    assert not _HNSWCollection._metadata_matches(meta, {"missing": {"$exists": True}})
