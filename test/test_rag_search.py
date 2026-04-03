import importlib
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class _StubDocDB:
    def __init__(self, results):
        self._results = results
        self.vector_path = "/tmp/managed-rag.vector"
        self.vector_backend = "chromadb"

    def query(self, query, n_results=5, where=None, ef_search=None):
        del query, n_results, where, ef_search
        return self._results


class RagSearchRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = importlib.import_module("skills.rag.scripts.retrieve")
        self.original_doc_db_from_agent = self.mod._doc_db_from_agent

    def tearDown(self) -> None:
        self.mod._doc_db_from_agent = self.original_doc_db_from_agent

    def test_rag_search_handles_numpy_distance_batch(self) -> None:
        self.mod._doc_db_from_agent = lambda: _StubDocDB(
            {
                "documents": [["Relevant chunk"]],
                "metadatas": [[{"source": "notes.md", "path": "/tmp/notes.md", "chunk": 2}]],
                "distances": [np.asarray([0.1234], dtype=np.float32)],
            }
        )

        raw = self.mod.rag_search("chunk")
        payload = json.loads(raw)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["source"], "notes.md")
        self.assertEqual(payload["results"][0]["chunk"], 2)
        self.assertAlmostEqual(payload["results"][0]["distance"], 0.1234, places=4)

    def test_rag_search_handles_empty_numpy_batch(self) -> None:
        self.mod._doc_db_from_agent = lambda: _StubDocDB(
            {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [np.asarray([], dtype=np.float32)],
            }
        )

        raw = self.mod.rag_search("missing")
        payload = json.loads(raw)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["results"], [])
        self.assertIn("No matches found", payload["message"])

    def test_rag_search_handles_batched_numpy_row_dicts(self) -> None:
        self.mod._doc_db_from_agent = lambda: _StubDocDB(
            np.asarray(
                [
                    [
                        {
                            "content": "Chunk from object array",
                            "metadata": {"source": "array.md", "chunk": 1},
                            "distance": np.float32(0.25),
                        }
                    ]
                ],
                dtype=object,
            )
        )

        raw = self.mod.rag_search("array")
        payload = json.loads(raw)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["source"], "array.md")
        self.assertEqual(payload["results"][0]["chunk"], 1)
        self.assertAlmostEqual(payload["results"][0]["distance"], 0.25, places=4)

    def test_rag_list_reports_managed_store_metadata(self) -> None:
        class _StubCollection:
            def count(self, where=None, include_deleted=False):
                del where, include_deleted
                return 0

            def get(self, include=None, include_deleted=False, where=None):
                del include, include_deleted, where
                return {"metadatas": []}

        stub = _StubDocDB([])
        stub.collection = _StubCollection()
        self.mod._doc_db_from_agent = lambda: stub
        self.mod.legacy_rag_vector_paths = lambda: [Path("/tmp/legacy-rag.vector")]

        payload = json.loads(self.mod.rag_list())

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["vector_path"], "/tmp/managed-rag.vector")
        self.assertEqual(payload["vector_backend"], "chromadb")


class RagRuntimeFallbackTests(unittest.TestCase):
    def test_retrieve_get_doc_db_uses_managed_settings(self) -> None:
        mod = importlib.import_module("skills.rag.scripts.retrieve")
        with (
            mock.patch.object(
                mod,
                "rag_runtime_settings",
                return_value={
                    "vector_path": "/tmp/managed-rag.vector",
                    "vector_backend": "chromadb",
                    "embedding_model_name": "demo-embedding",
                },
            ),
            mock.patch("src.db.document.DocumentDB") as document_db,
        ):
            mod._get_doc_db()

        document_db.assert_called_once_with(
            vector_path="/tmp/managed-rag.vector",
            embedding_model_name="demo-embedding",
            vector_backend="chromadb",
        )

    def test_ingest_get_doc_db_uses_managed_settings(self) -> None:
        mod = importlib.import_module("skills.rag.scripts.ingest")
        with (
            mock.patch.object(
                mod,
                "rag_runtime_settings",
                return_value={
                    "vector_path": "/tmp/managed-rag.vector",
                    "vector_backend": "chromadb",
                    "embedding_model_name": "demo-embedding",
                },
            ),
            mock.patch("src.db.document.DocumentDB") as document_db,
        ):
            mod._get_doc_db()

        document_db.assert_called_once_with(
            vector_path="/tmp/managed-rag.vector",
            embedding_model_name="demo-embedding",
            vector_backend="chromadb",
        )


if __name__ == "__main__":
    unittest.main()
