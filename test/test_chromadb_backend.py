import threading
import unittest

import numpy as np

from src.db.backends.chromadb import _ChromaDBCollection


class _FakeCollection:
    def query(self, **_kwargs):
        return {
            "ids": [["c", "a", "missing"]],
            "distances": [[0.05, 0.22, 0.4]],
        }


class _FakeReranker:
    def rerank(self, _query: str, hits, *, top_k: int):
        return hits[:top_k]


class _FakeEmbedder:
    def embed_query(self, _text: str) -> np.ndarray:
        return np.asarray([1.0, 2.0], dtype=np.float32)


class ChromaBackendTests(unittest.TestCase):
    def test_query_maps_ids_back_to_payload_and_applies_filters(self) -> None:
        coll = _ChromaDBCollection.__new__(_ChromaDBCollection)
        coll._lock = threading.RLock()
        coll._collection = _FakeCollection()
        coll._ensure_index = lambda: None  # type: ignore[method-assign]
        coll._embedder = _FakeEmbedder()
        coll._reranker = _FakeReranker()
        coll._space = "cosine"
        coll._min_similarity = 0.5
        coll._payload_by_label = {
            0: {
                "label": 0,
                "id": "a",
                "content": "doc a",
                "metadata": {"source": "s1"},
                "deleted": False,
            },
            1: {
                "label": 1,
                "id": "c",
                "content": "doc c",
                "metadata": {"source": "s2"},
                "deleted": False,
            },
        }
        coll._id_to_label = {"a": 0, "c": 1}
        coll._active_count = lambda: 2  # type: ignore[method-assign]

        rows = coll.query(
            query_texts=["find"],
            n_results=2,
            where={"source": "s2"},
            include=["documents", "metadatas", "distances"],
        )

        self.assertEqual(rows["ids"], [["c"]])
        self.assertEqual(rows["documents"], [["doc c"]])
        self.assertEqual(rows["metadatas"], [[{"source": "s2"}]])
        self.assertEqual(len(rows["distances"]), 1)
        self.assertEqual(len(rows["distances"][0]), 1)
        self.assertAlmostEqual(rows["distances"][0][0], 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
