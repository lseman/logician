from __future__ import annotations

from pathlib import Path

from src.wiki_context import build_wiki_context, wiki_workspace_summary


def test_wiki_workspace_summary_reads_manifest(tmp_path: Path) -> None:
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir(parents=True)
    (wiki_root / "wiki.md").write_text(
        "\n".join(
            [
                "# Local Wiki",
                "",
                "<!-- WIKI_MANIFEST_START -->",
                "```json",
                '{"document_count": 3, "raw_artifact_count": 2, "vault_dir": "/tmp/wiki", "source_dir": "/tmp/source", "raw_dir": "/tmp/raw"}',
                "```",
                "<!-- WIKI_MANIFEST_END -->",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = wiki_workspace_summary(base_dir=tmp_path)

    assert summary["document_count"] == 3
    assert summary["raw_artifact_count"] == 2
    assert summary["vault_dir"] == "/tmp/wiki"


def test_build_wiki_context_surfaces_relevant_pages(tmp_path: Path) -> None:
    wiki_root = tmp_path / "wiki"
    (wiki_root / "dist" / "indexes").mkdir(parents=True)
    (wiki_root / "dist" / "articles").mkdir(parents=True)
    (wiki_root / "wiki.md").write_text(
        "\n".join(
            [
                "# Local Wiki",
                "",
                "<!-- WIKI_MANIFEST_START -->",
                "```json",
                '{"document_count": 4, "raw_artifact_count": 1, "vault_dir": "/tmp/wiki", "source_dir": "/tmp/source", "raw_dir": "/tmp/raw"}',
                "```",
                "<!-- WIKI_MANIFEST_END -->",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (wiki_root / "dist" / "Home.md").write_text(
        "# Home\n\nKnowledge base home.\n", encoding="utf-8"
    )
    (wiki_root / "dist" / "indexes" / "Documents.md").write_text(
        "# Documents\n\nDeployment guide is indexed here.\n",
        encoding="utf-8",
    )
    (wiki_root / "dist" / "articles" / "deploy.md").write_text(
        "# Deployment\n\nRestart service before checking health.\n",
        encoding="utf-8",
    )

    context = build_wiki_context(
        "deployment health", base_dir=tmp_path, max_chars=1200, max_results=3
    )

    assert "Workspace snapshot: 4 compiled notes, 1 raw artifacts." in context
    assert "Relevant wiki pages:" in context
    assert "dist/articles/deploy.md" in context


def test_build_wiki_context_without_query_returns_home_and_indexes(tmp_path: Path) -> None:
    wiki_root = tmp_path / "wiki"
    (wiki_root / "dist" / "indexes").mkdir(parents=True)
    (wiki_root / "wiki.md").write_text(
        "\n".join(
            [
                "# Local Wiki",
                "",
                "<!-- WIKI_MANIFEST_START -->",
                "```json",
                '{"document_count": 2, "raw_artifact_count": 0, "vault_dir": "/tmp/wiki", "source_dir": "/tmp/source", "raw_dir": "/tmp/raw"}',
                "```",
                "<!-- WIKI_MANIFEST_END -->",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (wiki_root / "dist" / "Home.md").write_text(
        "# Home\n\nKnowledge base home.\n", encoding="utf-8"
    )
    (wiki_root / "dist" / "indexes" / "Documents.md").write_text(
        "# Documents\n\nDoc index.\n", encoding="utf-8"
    )
    (wiki_root / "dist" / "indexes" / "Concepts.md").write_text(
        "# Concepts\n\nConcept index.\n", encoding="utf-8"
    )

    context = build_wiki_context("", base_dir=tmp_path, max_chars=1200, max_results=3)

    assert "Workspace snapshot: 2 compiled notes, 0 raw artifacts." in context
    assert "dist/Home.md" in context
    assert "dist/indexes/Documents.md" in context
