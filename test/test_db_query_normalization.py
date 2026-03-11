import unittest

import numpy as np

from src.db.document import DocumentDB
from src.db.message import MessageDB


class _FakeCollection:
    def __init__(self, results):
        self._results = results

    def query(self, **_kwargs):
        return self._results


class DBQueryNormalizationTests(unittest.TestCase):
    def test_document_db_query_handles_numpy_batches(self) -> None:
        db = DocumentDB.__new__(DocumentDB)
        db.vector_enabled = True
        db.rerank_fetch_k = 30
        db.query_cache_enabled = False
        db._ensure = lambda: None  # type: ignore[method-assign]
        db._collection = _FakeCollection(
            {
                "documents": np.asarray([["doc a", "doc b"]], dtype=object),
                "metadatas": np.asarray(
                    [[{"source": "s1"}, {"source": "s2"}]],
                    dtype=object,
                ),
                "distances": np.asarray([[0.1, 0.2]], dtype=np.float32),
            }
        )
        db._cache_key = lambda **_kwargs: "k"  # type: ignore[method-assign]
        db._cache_get = lambda _key: None  # type: ignore[method-assign]
        db._cache_put = lambda _key, _rows: None  # type: ignore[method-assign]
        db._diversify_rows = lambda rows, *, n_results: rows[:n_results]  # type: ignore[method-assign]

        rows = db.query("hello", n_results=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["content"], "doc a")
        self.assertEqual(rows[1]["metadata"]["source"], "s2")
        self.assertAlmostEqual(rows[1]["distance"], 0.2, places=4)

    def test_message_db_vector_search_handles_numpy_batches(self) -> None:
        db = MessageDB.__new__(MessageDB)
        db._conn = object()
        db.vector_enabled = True
        db._ensure_vector = lambda: None  # type: ignore[method-assign]
        db._collection = _FakeCollection(
            {
                "documents": np.asarray([["tool output"]], dtype=object),
                "metadatas": np.asarray(
                    [[{"role": "tool", "name": "search", "tool_call_id": "call_1"}]],
                    dtype=object,
                ),
                "distances": np.asarray([[0.33]], dtype=np.float32),
                "ids": np.asarray([[""]], dtype=object),
            }
        )

        rows = db._vector_search_rows("find", 1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["role"], "tool")
        self.assertEqual(rows[0]["name"], "search")
        self.assertAlmostEqual(rows[0]["distance"], 0.33, places=4)


if __name__ == "__main__":
    unittest.main()
