import importlib
import json
import tempfile
import unittest
from pathlib import Path


class _StubDocDB:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def add_documents(
        self,
        texts,
        metadatas,
        chunk_size_tokens,
        chunk_overlap_ratio,
    ):
        del texts, chunk_size_tokens, chunk_overlap_ratio
        self.paths.append(metadatas[0]["path"])
        return ["chunk-1"]


class MountExcludeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ingest_mod = importlib.import_module("skills.rag.scripts.ingest")
        self.docling_mod = importlib.import_module(
            "skills.qol.docling_context.docling_context"
        )
        self.explore_mod = importlib.import_module(
            "skills.coding.explore.explore"
        )
        self.bridge_mod = importlib.import_module("logician_bridge")

        self.original_doc_db_from_agent = self.ingest_mod._doc_db_from_agent
        self.original_docling_add_file = self.docling_mod.docling_add_file

    def tearDown(self) -> None:
        self.ingest_mod._doc_db_from_agent = self.original_doc_db_from_agent
        self.docling_mod.docling_add_file = self.original_docling_add_file
        self.explore_mod._PROJECT_MAP_CACHE.clear()

    def test_mount_arg_parser_accepts_exclude_flag(self) -> None:
        parsed = self.bridge_mod._parse_mount_args(
            ["repo", "**/*.py", "50", "4", "-exclude", "node_modules,dist"]
        )

        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["directory"], "repo")
        self.assertEqual(parsed["glob"], "**/*.py")
        self.assertEqual(parsed["max_files"], "50")
        self.assertEqual(parsed["map_depth"], "4")
        self.assertEqual(parsed["exclude"], "node_modules,dist")
        self.assertEqual(parsed["exclude_display"], ["node_modules", "dist"])

    def test_rag_add_dir_skips_excluded_subfolders(self) -> None:
        stub_doc_db = _StubDocDB()
        self.ingest_mod._doc_db_from_agent = lambda: stub_doc_db

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep_dir = root / "keep"
            skip_dir = root / "skip"
            keep_dir.mkdir()
            skip_dir.mkdir()
            (keep_dir / "a.py").write_text("print('keep')\n", encoding="utf-8")
            (skip_dir / "b.py").write_text("print('skip')\n", encoding="utf-8")

            payload = json.loads(
                self.ingest_mod.rag_add_dir(
                    directory=str(root),
                    glob="**/*.py",
                    max_files=10,
                    exclude="skip",
                )
            )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["files_processed"], 1)
        self.assertEqual([item["file"] for item in payload["results"]], ["keep/a.py"])
        self.assertEqual(len(stub_doc_db.paths), 1)
        self.assertTrue(stub_doc_db.paths[0].endswith("keep/a.py"))

    def test_docling_add_dir_skips_excluded_subfolders(self) -> None:
        calls: list[str] = []

        def _fake_docling_add_file(path, source_label="", chunk_size=400, overlap=0.2):
            del source_label, chunk_size, overlap
            calls.append(path)
            return json.dumps({"status": "ok", "chunks_added": 1})

        self.docling_mod.docling_add_file = _fake_docling_add_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep_dir = root / "keep"
            skip_dir = root / "generated"
            keep_dir.mkdir()
            skip_dir.mkdir()
            (keep_dir / "guide.md").write_text("keep\n", encoding="utf-8")
            (skip_dir / "guide.md").write_text("skip\n", encoding="utf-8")

            payload = json.loads(
                self.docling_mod.docling_add_dir(
                    directory=str(root),
                    glob="**/*.md",
                    max_files=10,
                    exclude="generated",
                )
            )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["files_processed"], 1)
        self.assertEqual([item["file"] for item in payload["results"]], ["keep/guide.md"])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("keep/guide.md"))

    def test_get_project_map_skips_excluded_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep_dir = root / "pkg"
            skip_dir = root / "vendor"
            keep_dir.mkdir()
            skip_dir.mkdir()
            (keep_dir / "main.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
            (skip_dir / "vendored.py").write_text(
                "def skip():\n    return 2\n", encoding="utf-8"
            )

            payload = json.loads(
                self.explore_mod.get_project_map(
                    directory=str(root),
                    max_depth=3,
                    exclude="vendor",
                )
            )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["file_count"], 1)
        self.assertEqual([item["path"] for item in payload["files"]], ["pkg/main.py"])

    def test_get_project_map_includes_config_and_doc_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / "pkg"
            web_dir = root / "web"
            pkg_dir.mkdir()
            web_dir.mkdir()
            (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
            (root / "README.md").write_text("# Demo App\n", encoding="utf-8")
            (pkg_dir / "main.py").write_text(
                "class Worker:\n    pass\n\n\ndef run():\n    return 1\n",
                encoding="utf-8",
            )
            (web_dir / "app.ts").write_text(
                "export function renderApp() {\n  return 'ok'\n}\n",
                encoding="utf-8",
            )

            payload = json.loads(
                self.explore_mod.get_project_map(
                    directory=str(root),
                    max_depth=3,
                )
            )

        self.assertEqual(payload["status"], "ok")
        files = {item["path"]: item for item in payload["files"]}
        self.assertIn("pyproject.toml", files)
        self.assertIn("README.md", files)
        self.assertIn("pkg/main.py", files)
        self.assertIn("web/app.ts", files)
        self.assertEqual(files["pyproject.toml"]["kind"], "config")
        self.assertEqual(files["README.md"]["kind"], "doc")
        self.assertEqual(files["pkg/main.py"]["kind"], "source")
        self.assertEqual(files["web/app.ts"]["language"], "ts")
        self.assertTrue(files["pkg/main.py"]["summary"])
        self.assertTrue(files["pyproject.toml"]["summary"])
        self.assertTrue(files["README.md"]["summary"])
        self.assertIn("Worker", files["pkg/main.py"]["symbols"])
        self.assertEqual(payload["by_kind"]["source"], 2)

    def test_get_project_map_uses_symbol_summary_when_no_docstring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg_dir = root / "pkg"
            pkg_dir.mkdir()
            (pkg_dir / "helpers.py").write_text(
                "class Loader:\n    pass\n\n\ndef build_index():\n    return {}\n",
                encoding="utf-8",
            )

            payload = json.loads(
                self.explore_mod.get_project_map(
                    directory=str(root),
                    max_depth=3,
                )
            )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["file_count"], 1)
        entry = payload["files"][0]
        self.assertEqual(entry["path"], "pkg/helpers.py")
        self.assertIn("Defines", entry["summary"])
        self.assertIn("Loader", entry["summary"])
        self.assertIn("build_index", entry["summary"])


if __name__ == "__main__":
    unittest.main()
