from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.repo_ingest import ingest_repo


def _make_repo(root: Path) -> Path:
    repo = root / "demo-repo"
    pkg = repo / "pkg"
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
    (repo / "README.md").write_text("# Demo Repo\n", encoding="utf-8")
    return repo


def test_ingest_repo_populates_repo_artifacts_without_tui(
    tmp_path: Path,
) -> None:
    repo_path = _make_repo(tmp_path)

    payload = ingest_repo(repo_path, name="Demo Repo")

    assert payload["status"] == "ok"

    repo = payload["repo"]
    artifacts = dict(repo["artifacts"])
    repo_dir = Path(artifacts["repo_dir"])
    manifest = json.loads(Path(artifacts["manifest_path"]).read_text(encoding="utf-8"))
    summary_text = Path(artifacts["summary_path"]).read_text(encoding="utf-8")
    graph_lines = Path(artifacts["graph_path"]).read_text(encoding="utf-8").splitlines()

    assert repo_dir.exists()
    assert Path(payload["workspace_root"]) == repo_path.resolve()
    assert Path(payload["vector_path"]).name == "rag_docs.vector"
    assert manifest["files_processed"] >= 2
    assert manifest["chunks_added"] > 0
    assert manifest["graph_nodes"] > 0
    assert manifest["graph_edges"] > 0
    assert "graph_status: ready" in summary_text
    assert graph_lines
    assert (repo_path / ".logician" / "repos" / "index.json").exists()


def test_ingest_repo_still_builds_graph_when_rag_ingest_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_path = _make_repo(tmp_path)
    monkeypatch.setattr(
        "src.repo_ingest._ingest_repo_documents",
        lambda **_kwargs: {
            "status": "error",
            "error": "boom",
            "files_processed": 0,
            "total_chunks_added": 0,
        },
    )

    payload = ingest_repo(repo_path, name="Demo Repo")

    assert payload["status"] == "partial"
    assert payload["errors"]

    manifest = json.loads(
        Path(payload["repo"]["artifacts"]["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["files_processed"] == 0
    assert manifest["chunks_added"] == 0
    assert manifest["graph_nodes"] > 0
    assert manifest["last_graph_built_at"]


def test_ingest_repo_clones_git_url_into_managed_checkout(
    tmp_path: Path,
) -> None:
    source_repo = _make_repo(tmp_path / "source")
    subprocess.run(["git", "init"], cwd=source_repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=source_repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "initial",
        ],
        cwd=source_repo,
        check=True,
        capture_output=True,
    )

    payload = ingest_repo(f"file://{source_repo}", base_dir=tmp_path, name="Source Repo")

    assert payload["status"] == "ok"
    assert payload["source_url"] == f"file://{source_repo}"
    repo_path = Path(payload["repo"]["path"])
    assert repo_path.exists()
    assert repo_path.parent == tmp_path / ".logician" / "repos" / "_checkouts"
    assert (repo_path / ".git").exists()


def test_ingest_repo_uses_workspace_agent_config_for_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_path = _make_repo(tmp_path)
    (tmp_path / "agent_config.json").write_text(
        json.dumps(
            {
                "embedding_model": "demo-embedding-model",
                "rag_vector_backend": "chromadb",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    def fake_delete_existing_repo_chunks(**kwargs):
        captured["delete_embedding_model_name"] = kwargs["embedding_model_name"]
        captured["delete_vector_backend"] = kwargs["vector_backend"]
        return 0

    def fake_ingest_repo_documents(**kwargs):
        captured["ingest_embedding_model_name"] = kwargs["embedding_model_name"]
        captured["ingest_vector_backend"] = kwargs["vector_backend"]
        return {
            "status": "ok",
            "files_processed": 2,
            "total_chunks_added": 5,
            "results": [],
        }

    monkeypatch.setattr(
        "src.repo_ingest._delete_existing_repo_chunks",
        fake_delete_existing_repo_chunks,
    )
    monkeypatch.setattr(
        "src.repo_ingest._ingest_repo_documents",
        fake_ingest_repo_documents,
    )

    payload = ingest_repo(repo_path, name="Demo Repo", base_dir=tmp_path)

    assert payload["status"] == "ok"
    assert payload["vector_backend"] == "chromadb"
    assert payload["embedding_model_name"] == "demo-embedding-model"
    assert captured["delete_vector_backend"] == "chromadb"
    assert captured["ingest_vector_backend"] == "chromadb"
    assert captured["delete_embedding_model_name"] == "demo-embedding-model"
    assert captured["ingest_embedding_model_name"] == "demo-embedding-model"
