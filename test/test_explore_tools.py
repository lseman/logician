import importlib
import json
import tempfile
import unittest
from pathlib import Path


class ExploreToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.explore_mod = importlib.import_module(
            "skills.coding.explore.explore"
        )

    def test_get_file_outline_for_typescript_includes_imports_and_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "app.ts"
            target.write_text(
                "import { createRoot } from 'react-dom/client'\n"
                "export interface AppProps { title: string }\n"
                "export class AppShell {}\n"
                "export function renderApp(props: AppProps) {\n"
                "  return props.title\n"
                "}\n"
                "const bootstrap = async () => {\n"
                "  return renderApp({ title: 'ok' })\n"
                "}\n",
                encoding="utf-8",
            )

            payload = json.loads(self.explore_mod.get_file_outline(str(target)))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["language"], "ts")
        self.assertEqual(payload["imports"][0]["module"], "react-dom/client")
        function_names = {item["name"] for item in payload["functions"]}
        class_names = {item["name"] for item in payload["classes"]}
        self.assertIn("renderApp", function_names)
        self.assertIn("bootstrap", function_names)
        self.assertIn("AppShell", class_names)
        self.assertIn("AppProps", class_names)

    def test_get_file_outline_for_rust_includes_structs_and_functions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "lib.rs"
            target.write_text(
                "use std::collections::HashMap;\n\n"
                "pub struct Server;\n\n"
                "pub async fn run() -> Result<(), String> {\n"
                "    Ok(())\n"
                "}\n",
                encoding="utf-8",
            )

            payload = json.loads(self.explore_mod.get_file_outline(str(target)))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["language"], "rs")
        self.assertEqual(payload["imports"][0]["module"], "std::collections::HashMap")
        self.assertEqual(payload["classes"][0]["name"], "Server")
        self.assertEqual(payload["functions"][0]["name"], "run")

    def test_find_symbol_searches_supported_source_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            web_dir = root / "web"
            rust_dir = root / "rust"
            web_dir.mkdir()
            rust_dir.mkdir()
            (web_dir / "app.ts").write_text(
                "export function renderApp() {\n"
                "  return 'ok'\n"
                "}\n",
                encoding="utf-8",
            )
            (rust_dir / "lib.rs").write_text(
                "pub struct Server;\n\n"
                "pub fn run() {}\n",
                encoding="utf-8",
            )

            ts_payload = json.loads(
                self.explore_mod.find_symbol("renderApp", directory=str(root))
            )
            rust_payload = json.loads(
                self.explore_mod.find_symbol("Server", directory=str(root))
            )

        self.assertEqual(ts_payload["status"], "ok")
        self.assertEqual(ts_payload["count"], 1)
        self.assertEqual(ts_payload["matches"][0]["file"], "web/app.ts")
        self.assertEqual(ts_payload["matches"][0]["language"], "ts")
        self.assertEqual(rust_payload["status"], "ok")
        self.assertEqual(rust_payload["count"], 1)
        self.assertEqual(rust_payload["matches"][0]["file"], "rust/lib.rs")
        self.assertEqual(rust_payload["matches"][0]["kind"], "struct")

    def test_find_symbol_ranks_definition_files_before_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "src"
            test_dir = root / "tests"
            src_dir.mkdir()
            test_dir.mkdir()
            (src_dir / "server.py").write_text(
                "class Server:\n    pass\n",
                encoding="utf-8",
            )
            (test_dir / "test_server.py").write_text(
                "class Server:\n    pass\n",
                encoding="utf-8",
            )

            payload = json.loads(self.explore_mod.find_symbol("Server", directory=str(root)))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["matches"][0]["file"], "src/server.py")
        self.assertEqual(payload["top_files"][0]["file"], "src/server.py")
        self.assertIn("python", payload["by_language"])

    def test_rg_search_returns_ranked_top_files_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "src"
            doc_dir = root / "docs"
            src_dir.mkdir()
            doc_dir.mkdir()
            (src_dir / "server.py").write_text(
                "ServerConfig = {}\n"
                "def load_server_config():\n"
                "    return ServerConfig\n",
                encoding="utf-8",
            )
            (doc_dir / "notes.md").write_text(
                "ServerConfig appears in docs only once.\n",
                encoding="utf-8",
            )

            payload = json.loads(
                self.explore_mod.rg_search(
                    "ServerConfig",
                    directory=str(root),
                    fixed_string=True,
                    max_results=10,
                )
            )

        self.assertEqual(payload["status"], "ok")
        self.assertGreaterEqual(payload["count"], 2)
        self.assertIn("top_files", payload)
        self.assertEqual(payload["top_files"][0]["file"], "src/server.py")
        self.assertGreaterEqual(payload["top_files"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
