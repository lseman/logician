from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.repo.graph import build_repo_graph, related_repo_context
from src.repo.registry import register_repo, update_repo


class RepoGraphTests(unittest.TestCase):
    def test_python_graph_builds_call_and_reference_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / "pkg"
            pkg.mkdir(parents=True, exist_ok=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "service.py").write_text(
                "\n".join(
                    [
                        "def register_repo(name):",
                        "    return name.strip()",
                        "",
                        "def setup_repo(name):",
                        "    return register_repo(name)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (pkg / "orchestrator.py").write_text(
                "\n".join(
                    [
                        "from pkg.service import register_repo",
                        "",
                        "def boot(name):",
                        "    return register_repo(name)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (pkg / "consumer.py").write_text(
                "\n".join(
                    [
                        "from pkg.service import setup_repo",
                        "",
                        "RESULT = setup_repo('demo')",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            repo = register_repo(str(root), name="repo-graph-test", base_dir=root)
            repo = update_repo(str(repo["id"]), base_dir=root) or repo

            payload = build_repo_graph(
                repo,
                glob_pattern="pkg/**/*.py",
                max_files=20,
                base_dir=root,
            )

            self.assertEqual(payload["files_indexed"], 4)
            self.assertGreater(payload["call_edges"], 0)
            self.assertGreater(payload["reference_edges"], 0)

            repo = update_repo(str(repo["id"]), base_dir=root) or repo
            related = related_repo_context(
                repo,
                rel_paths=["pkg/service.py"],
                query="where is register_repo called",
                limit=6,
            )

            related_paths = [item["rel_path"] for item in related["related_files"]]
            self.assertIn("pkg/orchestrator.py", related_paths)
            self.assertIn("pkg/consumer.py", related_paths)

            symbol_names = [item["name"] for item in related["related_symbols"]]
            self.assertIn("register_repo", symbol_names)

    def test_typescript_graph_builds_call_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / "pkg"
            pkg.mkdir(parents=True, exist_ok=True)
            (pkg / "service.ts").write_text(
                "\n".join(
                    [
                        "export function registerRepo(name: string) {",
                        "  return name.trim();",
                        "}",
                        "",
                        "export function setupRepo(name: string) {",
                        "  return registerRepo(name);",
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (pkg / "orchestrator.ts").write_text(
                "\n".join(
                    [
                        "import { registerRepo } from './service';",
                        "",
                        "export function boot(name: string) {",
                        "  return registerRepo(name);",
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (pkg / "consumer.ts").write_text(
                "\n".join(
                    [
                        "import { setupRepo } from './service';",
                        "",
                        "export const RESULT = setupRepo('demo');",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            repo = register_repo(str(root), name="repo-graph-ts", base_dir=root)
            repo = update_repo(str(repo["id"]), base_dir=root) or repo

            payload = build_repo_graph(
                repo,
                glob_pattern="pkg/**/*.ts",
                max_files=20,
                base_dir=root,
            )

            self.assertEqual(payload["files_indexed"], 3)
            self.assertGreater(payload["call_edges"], 0)
            self.assertGreater(payload["reference_edges"], 0)

            repo = update_repo(str(repo["id"]), base_dir=root) or repo
            related = related_repo_context(
                repo,
                rel_paths=["pkg/service.ts"],
                query="where is registerRepo called",
                limit=6,
            )

            related_paths = [item["rel_path"] for item in related["related_files"]]
            self.assertIn("pkg/orchestrator.ts", related_paths)
            self.assertIn("pkg/consumer.ts", related_paths)

    def test_typescript_graph_resolves_export_and_import_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / "pkg"
            pkg.mkdir(parents=True, exist_ok=True)
            (pkg / "service.ts").write_text(
                "\n".join(
                    [
                        "export function registerRepo(name: string) {",
                        "  return name.trim();",
                        "}",
                        "",
                        "export { registerRepo as saveRepo };",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (pkg / "export_alias_consumer.ts").write_text(
                "\n".join(
                    [
                        "import { saveRepo } from './service';",
                        "",
                        "export const RESULT = saveRepo('demo');",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (pkg / "local_alias_consumer.ts").write_text(
                "\n".join(
                    [
                        "import { registerRepo as renamedRepo } from './service';",
                        "",
                        "export const RESULT = renamedRepo('demo');",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            repo = register_repo(str(root), name="repo-graph-ts-alias", base_dir=root)
            repo = update_repo(str(repo["id"]), base_dir=root) or repo

            payload = build_repo_graph(
                repo,
                glob_pattern="pkg/**/*.ts",
                max_files=20,
                base_dir=root,
            )

            self.assertEqual(payload["files_indexed"], 3)
            self.assertGreater(payload["call_edges"], 0)

            repo = update_repo(str(repo["id"]), base_dir=root) or repo
            related = related_repo_context(
                repo,
                rel_paths=["pkg/service.ts"],
                query="where is saveRepo called",
                limit=8,
            )

            related_paths = [item["rel_path"] for item in related["related_files"]]
            self.assertIn("pkg/export_alias_consumer.ts", related_paths)
            self.assertIn("pkg/local_alias_consumer.ts", related_paths)

    def test_rust_graph_builds_call_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir(parents=True, exist_ok=True)
            (src / "service.rs").write_text(
                "\n".join(
                    [
                        "pub fn register_repo(name: &str) -> String {",
                        "    name.trim().to_string()",
                        "}",
                        "",
                        "pub fn setup_repo(name: &str) -> String {",
                        "    register_repo(name)",
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (src / "orchestrator.rs").write_text(
                "\n".join(
                    [
                        "use crate::service::register_repo;",
                        "",
                        "pub fn boot(name: &str) -> String {",
                        "    register_repo(name)",
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (src / "consumer.rs").write_text(
                "\n".join(
                    [
                        "use crate::service::setup_repo;",
                        "",
                        "pub fn run() -> String {",
                        '    setup_repo("demo")',
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            repo = register_repo(str(root), name="repo-graph-rs", base_dir=root)
            repo = update_repo(str(repo["id"]), base_dir=root) or repo

            payload = build_repo_graph(
                repo,
                glob_pattern="src/**/*.rs",
                max_files=20,
                base_dir=root,
            )

            self.assertEqual(payload["files_indexed"], 3)
            self.assertGreater(payload["call_edges"], 0)
            self.assertGreater(payload["reference_edges"], 0)

            repo = update_repo(str(repo["id"]), base_dir=root) or repo
            related = related_repo_context(
                repo,
                rel_paths=["src/service.rs"],
                query="where is register_repo called",
                limit=6,
            )

            related_paths = [item["rel_path"] for item in related["related_files"]]
            self.assertIn("src/orchestrator.rs", related_paths)
            self.assertIn("src/consumer.rs", related_paths)

    def test_rust_graph_resolves_trait_and_alias_qualified_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir(parents=True, exist_ok=True)
            (src / "service.rs").write_text(
                "\n".join(
                    [
                        "pub trait RepoStore {",
                        "    fn save(&self, name: &str) -> String;",
                        "}",
                        "",
                        "pub struct Repository;",
                        "",
                        "impl RepoStore for Repository {",
                        "    fn save(&self, name: &str) -> String {",
                        "        name.trim().to_string()",
                        "    }",
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (src / "trait_caller.rs").write_text(
                "\n".join(
                    [
                        "use crate::service::{RepoStore, Repository};",
                        "",
                        "pub fn run(repo: &Repository) -> String {",
                        '    RepoStore::save(repo, "demo")',
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (src / "alias_caller.rs").write_text(
                "\n".join(
                    [
                        "use crate::service::Repository as Repo;",
                        "",
                        "pub fn run(repo: &Repo) -> String {",
                        '    Repo::save(repo, "demo")',
                        "}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            repo = register_repo(str(root), name="repo-graph-rs-alias", base_dir=root)
            repo = update_repo(str(repo["id"]), base_dir=root) or repo

            payload = build_repo_graph(
                repo,
                glob_pattern="src/**/*.rs",
                max_files=20,
                base_dir=root,
            )

            self.assertEqual(payload["files_indexed"], 3)
            self.assertGreater(payload["call_edges"], 0)

            repo = update_repo(str(repo["id"]), base_dir=root) or repo
            related = related_repo_context(
                repo,
                rel_paths=["src/service.rs"],
                query="where is Repository::save called",
                limit=8,
            )

            related_paths = [item["rel_path"] for item in related["related_files"]]
            self.assertIn("src/trait_caller.rs", related_paths)
            self.assertIn("src/alias_caller.rs", related_paths)


if __name__ == "__main__":
    unittest.main()
