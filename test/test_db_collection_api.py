from __future__ import annotations

import threading

import numpy as np

from src.db.core import _HNSWCollection


def _fake_collection() -> _HNSWCollection:
    coll = _HNSWCollection.__new__(_HNSWCollection)
    coll._lock = threading.RLock()
    coll._payload_by_label = {
        0: {
            "label": 0,
            "id": "a",
            "content": "doc a",
            "metadata": {"source": "s1", "chunk": 0},
            "deleted": False,
        },
        1: {
            "label": 1,
            "id": "b",
            "content": "doc b",
            "metadata": {"source": "s2", "chunk": 1},
            "deleted": True,
        },
        2: {
            "label": 2,
            "id": "c",
            "content": "doc c",
            "metadata": {"source": "s1", "chunk": 2},
            "deleted": False,
        },
    }
    coll._id_to_label = {"a": 0, "c": 2}
    return coll


def test_count_respects_deleted_and_where() -> None:
    coll = _fake_collection()
    assert coll.count() == 2
    assert coll.count(include_deleted=True) == 3
    assert coll.count(where={"source": "s1"}) == 2
    assert coll.count(where={"chunk": {"$gte": 1}}) == 1


def test_get_ids_and_order() -> None:
    coll = _fake_collection()
    rows = coll.get(ids=["c", "a"], include=["documents"])
    assert rows["ids"] == ["c", "a"]
    assert rows["documents"] == ["doc c", "doc a"]


def test_get_with_offset_limit() -> None:
    coll = _fake_collection()
    rows = coll.get(include=["metadatas"], offset=1, limit=1)
    assert rows["ids"] == ["c"]
    assert rows["metadatas"][0]["chunk"] == 2


def test_peek_first_records() -> None:
    coll = _fake_collection()
    rows = coll.peek(limit=1)
    assert rows["ids"] == ["a"]


class _FakeLog:
    def debug(self, *_args, **_kwargs) -> None:
        return None


class _FakeIndex:
    def __init__(self) -> None:
        self.add_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def add_items(self, vecs: np.ndarray, labels: np.ndarray) -> None:
        self.add_calls.append((np.asarray(vecs), np.asarray(labels)))

    def mark_deleted(self, _label: int) -> None:
        return None


class _FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.ones((n, 2), dtype=np.float32)


def _fake_collection_for_add() -> _HNSWCollection:
    coll = _HNSWCollection.__new__(_HNSWCollection)
    coll._lock = threading.RLock()
    coll._index = _FakeIndex()
    coll._embedder = _FakeEmbedder()
    coll._log = _FakeLog()
    coll._payload_by_label = {}
    coll._id_to_label = {}
    coll._next_label = 0
    coll._ensure_index = lambda: None  # type: ignore[method-assign]
    coll._ensure_capacity = lambda _n: None  # type: ignore[method-assign]
    coll._save = lambda: None  # type: ignore[method-assign]
    return coll


def test_add_progress_callback_receives_batches() -> None:
    coll = _fake_collection_for_add()
    events: list[dict[str, object]] = []

    coll.add(
        documents=["d1", "d2", "d3", "d4", "d5"],
        metadatas=[{"i": i} for i in range(5)],
        ids=[f"id_{i}" for i in range(5)],
        batch_size=2,
        progress_callback=lambda e: events.append(e),
    )

    assert len(events) == 5  # start + 3 batch + done
    assert [str(e.get("stage")) for e in events] == [
        "start",
        "batch",
        "batch",
        "batch",
        "done",
    ]
    assert int(events[-1].get("added", 0)) == 5
    assert len(coll._index.add_calls) == 3


def test_add_progress_callback_supports_kwargs_signature() -> None:
    coll = _fake_collection_for_add()
    events: list[dict[str, object]] = []

    def _cb(**event: object) -> None:
        events.append(event)

    coll.add(
        documents=["d1", "d2", "d3"],
        metadatas=[{"i": 1}, {"i": 2}, {"i": 3}],
        ids=["a", "b", "c"],
        batch_size=2,
        progress_callback=_cb,
    )
    assert len(events) >= 3
    assert str(events[-1].get("stage")) == "done"
