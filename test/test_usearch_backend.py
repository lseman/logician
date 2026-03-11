import unittest

import numpy as np

from src.db.backends.usearch import _USEARCHCollection


class _DummySearchResult:
    def __init__(self) -> None:
        self.keys = np.asarray([11, 22, 33], dtype=np.int64)
        self.distances = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)


class _DummyIndex:
    def search(self, _query, _k):
        return _DummySearchResult()


class UsearchBackendTests(unittest.TestCase):
    def test_search_usearch_handles_numpy_attrs_without_truthiness(self) -> None:
        coll = _USEARCHCollection.__new__(_USEARCHCollection)
        coll._index = _DummyIndex()

        keys, distances = coll._search_usearch(np.asarray([1.0, 2.0], dtype=np.float32), k=2)

        self.assertEqual(keys.tolist(), [11, 22])
        self.assertEqual([round(float(x), 4) for x in distances.tolist()], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
