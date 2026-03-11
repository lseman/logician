import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.db.backends.hnsw import _HNSWCollection
from src.db.embeddings import _EmbeddingRuntime


class _DummyLog:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class _DummyIndexMethod:
    def get_current_count(self):
        return 42


class _DummyIndexAttr:
    element_count = 17


class HnswConsistencyTests(unittest.TestCase):
    def test_embedding_runtime_prefer_candidate_reorders_candidates(self) -> None:
        runtime = _EmbeddingRuntime("a|b|c", _DummyLog())

        runtime.prefer_candidate("c")

        self.assertEqual(runtime._model_name_raw, "c|a|b")

    def test_embedding_runtime_prefer_candidate_ignores_unknown_candidate(self) -> None:
        runtime = _EmbeddingRuntime("a|b|c", _DummyLog())

        runtime.prefer_candidate("x")

        self.assertEqual(runtime._model_name_raw, "a|b|c")

    def test_index_count_prefers_method_then_attribute(self) -> None:
        self.assertEqual(_HNSWCollection._index_count(_DummyIndexMethod()), 42)
        self.assertEqual(_HNSWCollection._index_count(_DummyIndexAttr()), 17)

    def test_hnsw_rejects_saved_backend_mismatch(self) -> None:
        with TemporaryDirectory() as td:
            coll = _HNSWCollection.__new__(_HNSWCollection)
            coll._backend = "hnsw"
            coll._dir = Path(td)
            coll._state_file = Path(td) / "state.json"
            coll._payload_file = Path(td) / "payload.jsonl"
            coll._index_file = Path(td) / "index.bin"
            coll._state_file.write_text('{"backend":"usearch"}', encoding="utf-8")
            coll._payload_by_label = {}
            coll._id_to_label = {}
            coll._next_label = 0
            coll._dim = None
            coll._max_elements = 0
            coll._index = None
            coll._space = "cosine"
            coll._ef_search = 128
            coll._embedder = _EmbeddingRuntime("a|b", _DummyLog())

            with self.assertRaises(RuntimeError):
                coll._ensure_index()


if __name__ == "__main__":
    unittest.main()
