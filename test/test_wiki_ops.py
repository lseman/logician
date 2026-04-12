from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _load_wiki_ops():
    module_name = "test_wiki_ops_module"
    module_path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "wiki"
        / "wiki_skills"
        / "scripts"
        / "wiki_ops.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


wiki_ops = _load_wiki_ops()


def test_wiki_build_and_search_round_trip(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    raw_dir = tmp_path / "raw"
    (tmp_path / "AGENTS.md").write_text(
        "# Schema\n\nMaintain the wiki carefully.\n", encoding="utf-8"
    )
    (source_dir / "guides").mkdir(parents=True)
    (source_dir / "intro.md").write_text(
        "# Intro\n\nWelcome to the local wiki.\n", encoding="utf-8"
    )
    (source_dir / "guides" / "deploy.md").write_text(
        "# Deployment\n\nSee [[Intro]] before restarting the service.\n\nRestart service before checking health.\n",
        encoding="utf-8",
    )
    (raw_dir / "articles").mkdir(parents=True)
    (raw_dir / "articles" / "source.txt").write_text("raw source\n", encoding="utf-8")

    build = wiki_ops.wiki_build(
        source_dir=str(source_dir),
        wiki_path=str(wiki_path),
        raw_dir=str(raw_dir),
        vault_dir=str(vault_dir),
    )

    assert build["status"] == "success"
    assert build["document_count"] == 2
    assert build["raw_artifact_count"] == 1
    rendered = wiki_path.read_text(encoding="utf-8")
    assert "<!-- WIKI_MANIFEST_START -->" in rendered
    assert "Restart service before checking health." in rendered
    assert "raw/source" not in rendered
    assert (vault_dir / "Home.md").exists()
    assert (vault_dir / "index.md").exists()
    assert (vault_dir / "schema.md").exists()
    assert (vault_dir / "log.md").exists()
    assert (vault_dir / "indexes" / "Documents.md").exists()
    assert (vault_dir / "articles" / "guides-deploy-md.md").exists()
    deploy_article = (vault_dir / "articles" / "guides-deploy-md.md").read_text(encoding="utf-8")
    assert "[[articles/intro-md|Intro]]" in deploy_article
    root_index = (vault_dir / "index.md").read_text(encoding="utf-8")
    assert "[[log]]" in root_index
    schema_page = (vault_dir / "schema.md").read_text(encoding="utf-8")
    assert "AGENTS.md" in schema_page
    assert (vault_dir / "mkdocs.yml").exists()
    assert (vault_dir / "site_docs" / "Home.md").exists()
    assert (vault_dir / "MKDOCS_BUILD_STATUS.md").exists()
    converted_article = (vault_dir / "site_docs" / "articles" / "guides-deploy-md.md").read_text(
        encoding="utf-8"
    )
    assert "[Intro](articles/intro-md.md)" in converted_article
    if shutil.which("mkdocs"):
        assert (vault_dir / "site" / "index.html").exists()

    search = wiki_ops.wiki_search("restart service", wiki_path=str(wiki_path))

    assert search["status"] == "success"
    assert search["count"] == 1
    assert search["results"][0]["relative_path"] == "guides/deploy.md"
    assert search["results"][0]["matches"][0]["line_number"] == 5
    log_text = (vault_dir / "log.md").read_text(encoding="utf-8")
    assert "build | Wiki Rebuilt" in log_text
    assert "query | restart service" in log_text


def test_wiki_search_falls_back_to_dist_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    source_dir.mkdir(parents=True)
    (source_dir / "algorithms").mkdir(parents=True, exist_ok=True)
    (source_dir / "algorithms" / "branch_and_bound.md").write_text(
        "# Branch and Bound\n\nBranch and bound is a search algorithm for combinatorial optimization.\n",
        encoding="utf-8",
    )

    build = wiki_ops.wiki_build(
        source_dir=str(source_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )
    assert build["status"] == "success"
    assert (vault_dir / "index.md").exists()

    search = wiki_ops.wiki_search("branch and bound", wiki_path=str(vault_dir))
    assert search["status"] == "success"
    assert search["count"] >= 1
    assert any(
        result["relative_path"].endswith("algorithms/branch_and_bound.md")
        for result in search["results"]
    )


def test_wiki_search_falls_back_to_repo_checkouts(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    raw_dir = tmp_path / "raw"
    source_dir.mkdir(parents=True)
    repo_path = raw_dir / "repos" / "scip" / "checkout"
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "branch_and_bound.md").write_text(
        "# Branch and Bound\n\nBranch and bound is an optimization method used in combinatorial search.\n",
        encoding="utf-8",
    )

    build = wiki_ops.wiki_build(
        source_dir=str(source_dir),
        wiki_path=str(wiki_path),
        raw_dir=str(raw_dir),
        vault_dir=str(vault_dir),
    )
    assert build["status"] == "success"

    search = wiki_ops.wiki_search("branch and bound", wiki_path=str(wiki_path))
    assert search["status"] == "success"
    assert search["count"] >= 1
    assert any(
        result["relative_path"].endswith("branch_and_bound.md") for result in search["results"]
    )


def test_wiki_add_and_update_document_rebuilds_index(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"

    added = wiki_ops.wiki_add_document(
        "notes/todo.md",
        "# Tasks\n\n- Ship wiki\n",
        source_dir=str(source_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )

    assert added["status"] == "success"
    assert added["build"]["document_count"] == 1

    updated = wiki_ops.wiki_update_document(
        "notes/todo.md",
        find_text="- Ship wiki",
        replace_text="- Ship wiki\n- Verify health",
        source_dir=str(source_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )

    assert updated["status"] == "success"
    document = wiki_ops.wiki_get_document("notes/todo.md", wiki_path=str(wiki_path))
    assert document["status"] == "success"
    assert "- Verify health" in document["document"]["content"]

    search = wiki_ops.wiki_search("verify health", wiki_path=str(wiki_path))
    assert search["status"] == "success"
    assert search["results"][0]["relative_path"] == "notes/todo.md"


def test_wiki_health_detects_stale_index(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    source_dir.mkdir(parents=True)
    document_path = source_dir / "runbook.md"
    document_path.write_text("# Runbook\n\nVersion one.\n", encoding="utf-8")

    build = wiki_ops.wiki_build(
        source_dir=str(source_dir), wiki_path=str(wiki_path), vault_dir=str(vault_dir)
    )
    assert build["status"] == "success"

    document_path.write_text("# Runbook\n\nVersion two.\n", encoding="utf-8")

    health = wiki_ops.wiki_health(
        source_dir=str(source_dir), wiki_path=str(wiki_path), vault_dir=str(vault_dir)
    )

    assert health["status"] == "warning"
    assert health["healthy"] is False
    assert health["needs_rebuild"] is True
    assert health["stale_documents"][0]["relative_path"] == "runbook.md"


def test_wiki_list_sources_returns_manifest_only(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    source_dir.mkdir(parents=True)
    (source_dir / "alpha.md").write_text("# Alpha\n\nA body.\n", encoding="utf-8")

    build = wiki_ops.wiki_build(source_dir=str(source_dir), wiki_path=str(wiki_path))
    assert build["status"] == "success"

    listed = wiki_ops.wiki_list_sources(wiki_path=str(wiki_path))

    assert listed["status"] == "success"
    assert listed["count"] == 1
    assert listed["documents"][0]["relative_path"] == "alpha.md"
    assert "content" not in listed["documents"][0]


def test_wiki_read_source_note_reads_direct_source_content(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    source_dir.mkdir(parents=True)
    (source_dir / "repos").mkdir(parents=True)
    note_path = source_dir / "repos" / "scip.md"
    note_path.write_text("# SCIP\n\nConstraint integer programming details.\n", encoding="utf-8")

    result = wiki_ops.wiki_read_source_note("repos/scip.md", source_dir=str(source_dir))

    assert result["status"] == "success"
    assert result["relative_path"] == "repos/scip.md"
    assert result["document"]["title"] == "SCIP"
    assert "Constraint integer programming details." in result["document"]["content"]


def test_wiki_write_output_updates_outputs_index(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    source_dir.mkdir(parents=True)
    (source_dir / "alpha.md").write_text("# Alpha\n\nA body.\n", encoding="utf-8")

    build = wiki_ops.wiki_build(
        source_dir=str(source_dir), wiki_path=str(wiki_path), vault_dir=str(vault_dir)
    )
    assert build["status"] == "success"

    output = wiki_ops.wiki_write_output(
        "briefs/summary.md", "# Summary\n\nFiled output.\n", vault_dir=str(vault_dir)
    )

    assert output["status"] == "success"
    assert (vault_dir / "outputs" / "briefs" / "summary.md").exists()
    outputs_index = (vault_dir / "outputs" / "README.md").read_text(encoding="utf-8")
    assert "[[outputs/briefs/summary]]" in outputs_index
    log_text = (vault_dir / "log.md").read_text(encoding="utf-8")
    assert "query | summary.md" in log_text


def test_wiki_lint_writes_report_and_flags_short_documents(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    vault_dir = tmp_path / "wiki"
    source_dir.mkdir(parents=True)
    (source_dir / "tiny.md").write_text("# Tiny\n\nshort note\n", encoding="utf-8")

    report = wiki_ops.wiki_lint(
        source_dir=str(source_dir), vault_dir=str(vault_dir), write_report=True
    )

    assert report["status"] == "warning"
    assert any(finding["kind"] == "short_document" for finding in report["findings"])
    assert (vault_dir / "reports" / "Lint.md").exists()


def test_wiki_lint_flags_possible_contradictions(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    vault_dir = tmp_path / "wiki"
    source_dir.mkdir(parents=True)
    (source_dir / "a.md").write_text(
        "# Policy A\n\nThe deployment process should use feature flags for production rollouts.\n",
        encoding="utf-8",
    )
    (source_dir / "b.md").write_text(
        "# Policy B\n\nThe deployment process should not use feature flags for production rollouts.\n",
        encoding="utf-8",
    )

    report = wiki_ops.wiki_lint(
        source_dir=str(source_dir), vault_dir=str(vault_dir), write_report=True
    )

    assert report["status"] == "warning"
    assert report["contradiction_count"] >= 1
    assert any(finding["kind"] == "possible_contradiction" for finding in report["findings"])


def test_wiki_ingest_raw_promotes_artifact_into_source_note(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nTest schema.\n", encoding="utf-8")
    (source_dir / "guides").mkdir(parents=True)
    related_note = source_dir / "guides" / "feature-flags.md"
    related_note.write_text(
        "# Feature Flags\n\nFeature flags improve safety during production rollouts.\n",
        encoding="utf-8",
    )
    (raw_dir / "research").mkdir(parents=True)
    raw_path = raw_dir / "research" / "notes.txt"
    raw_path.write_text(
        "Feature flags improve safety during production rollouts.\nUse them selectively.\n",
        encoding="utf-8",
    )

    result = wiki_ops.wiki_ingest_raw(
        "research/notes.txt",
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )

    assert result["status"] == "success"
    assert result["source_note_path"] == "sources/research/notes.md"
    created = source_dir / "sources" / "research" / "notes.md"
    assert created.exists()
    created_text = created.read_text(encoding="utf-8")
    assert "Derived from raw artifact `research/notes.txt`" in created_text
    assert "## Source Excerpt" in created_text
    assert "## Related Existing Notes" in created_text
    assert "## Suggested Follow-Up Pages" in created_text
    assert "Feature Flags" in created_text
    related_text = related_note.read_text(encoding="utf-8")
    assert "## Source Updates" in related_text
    assert "sources/research/notes" in related_text
    assert result["build"]["document_count"] == 2
    assert result["updated_related_paths"] == ["guides/feature-flags.md"]
    assert any(
        suggestion["action"] == "update"
        and suggestion["relative_path"] == "guides/feature-flags.md"
        for suggestion in result["suggested_pages"]
    )
    assert (vault_dir / "log.md").exists()


def test_wiki_ingest_raw_suggests_new_follow_up_page_when_topic_is_missing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nSuggestion schema.\n", encoding="utf-8")
    raw_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    raw_path = raw_dir / "semantic-retrieval.txt"
    raw_path.write_text(
        "Vector databases enable semantic retrieval for notebooks and personal knowledge bases.\n",
        encoding="utf-8",
    )

    result = wiki_ops.wiki_ingest_raw(
        "semantic-retrieval.txt",
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
        update_related=False,
    )

    assert result["status"] == "success"
    assert any(
        suggestion["action"] == "create" and suggestion["relative_path"].startswith("concepts/")
        for suggestion in result["suggested_pages"]
    )


def test_wiki_promote_suggestion_creates_follow_up_note(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nPromote schema.\n", encoding="utf-8")
    raw_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    (raw_dir / "semantic-retrieval.txt").write_text(
        "Vector databases enable semantic retrieval for notebooks and personal knowledge bases.\n",
        encoding="utf-8",
    )

    ingested = wiki_ops.wiki_ingest_raw(
        "semantic-retrieval.txt",
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
        update_related=False,
    )
    create_suggestion = next(
        suggestion for suggestion in ingested["suggested_pages"] if suggestion["action"] == "create"
    )

    promoted = wiki_ops.wiki_promote_suggestion(
        source_note_path=ingested["source_note_path"],
        suggestion_path=create_suggestion["relative_path"],
        source_dir=str(source_dir),
        wiki_path=str(wiki_path),
        raw_dir=str(raw_dir),
        vault_dir=str(vault_dir),
    )

    assert promoted["status"] == "success"
    created_path = source_dir / create_suggestion["relative_path"]
    assert created_path.exists()
    created_text = created_path.read_text(encoding="utf-8")
    assert "Suggested follow-up page promoted from" in created_text
    assert "Why This Page Exists" in created_text


def test_wiki_promote_suggestion_updates_existing_note(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nUpdate promote schema.\n", encoding="utf-8")
    (source_dir / "guides").mkdir(parents=True)
    related_note = source_dir / "guides" / "feature-flags.md"
    related_note.write_text("# Feature Flags\n\nExisting note.\n", encoding="utf-8")
    (raw_dir / "research").mkdir(parents=True)
    (raw_dir / "research" / "notes.txt").write_text(
        "Feature flags improve safety during production rollouts.\n",
        encoding="utf-8",
    )

    ingested = wiki_ops.wiki_ingest_raw(
        "research/notes.txt",
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
        update_related=False,
    )
    listed = wiki_ops.wiki_list_suggestions(
        source_note_path=ingested["source_note_path"],
        source_dir=str(source_dir),
    )
    update_suggestion = next(
        suggestion for suggestion in listed["suggestions"] if suggestion["action"] == "update"
    )

    promoted = wiki_ops.wiki_promote_suggestion(
        source_note_path=ingested["source_note_path"],
        suggestion_path=update_suggestion["relative_path"],
        source_dir=str(source_dir),
        wiki_path=str(wiki_path),
        raw_dir=str(raw_dir),
        vault_dir=str(vault_dir),
    )

    assert promoted["status"] == "success"
    related_text = related_note.read_text(encoding="utf-8")
    assert "## Suggested Follow-Up Work" in related_text
    assert ingested["source_note_path"].removesuffix(".md") in related_text


def test_wiki_recreate_alias_and_cli_work_standalone(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    source_dir.mkdir(parents=True)
    (source_dir / "guide.md").write_text("# Guide\n\nRecreate the wiki.\n", encoding="utf-8")

    recreated = wiki_ops.wiki_recreate(
        source_dir=str(source_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )

    assert getattr(wiki_ops.wiki_recreate, "__tool__", False) is True
    assert recreated["status"] == "success"
    assert recreated["document_count"] == 1

    cli_path = Path(__file__).resolve().parents[1] / "skills" / "wiki" / "wiki_cli.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "search",
            "recreate",
            "--wiki-path",
            str(wiki_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "success"
    assert payload["count"] == 1


def test_top_level_wiki_launchers_work(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    source_dir.mkdir(parents=True)
    (source_dir / "alpha.md").write_text("# Alpha\n\nLauncher path.\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]

    module_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "wiki",
            "recreate",
            "--source-dir",
            str(source_dir),
            "--wiki-path",
            str(wiki_path),
            "--vault-dir",
            str(vault_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    module_payload = json.loads(module_proc.stdout)
    assert module_payload["status"] == "success"

    launcher_path = repo_root / "wiki"
    launcher_proc = subprocess.run(
        [
            str(launcher_path),
            "search",
            "launcher",
            "--wiki-path",
            str(wiki_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    launcher_payload = json.loads(launcher_proc.stdout)
    assert launcher_payload["status"] == "success"
    assert launcher_payload["count"] == 1


def test_wiki_cli_ingest_raw_command(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nCLI schema.\n", encoding="utf-8")
    (raw_dir / "research").mkdir(parents=True)
    (raw_dir / "research" / "article.txt").write_text(
        "A durable claim from raw input.\n", encoding="utf-8"
    )

    cli_path = Path(__file__).resolve().parents[1] / "skills" / "wiki" / "wiki_cli.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "ingest-raw",
            "research/article.txt",
            "--source-dir",
            str(source_dir),
            "--raw-dir",
            str(raw_dir),
            "--wiki-path",
            str(wiki_path),
            "--vault-dir",
            str(vault_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "success"
    assert (source_dir / "sources" / "research" / "article.md").exists()


def test_wiki_ingest_repo_snapshots_codebase_into_raw_and_source_note(tmp_path: Path) -> None:
    repo_dir = tmp_path / "sample_repo"
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nRepo schema.\n", encoding="utf-8")
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "tests").mkdir(parents=True)
    (repo_dir / ".git").mkdir(parents=True)
    (repo_dir / "README.md").write_text("# Sample Repo\n\nA small service.\n", encoding="utf-8")
    (repo_dir / "pyproject.toml").write_text("[project]\nname = 'sample-repo'\n", encoding="utf-8")
    (repo_dir / "src" / "app.py").write_text(
        "def main() -> None:\n    print('hello from repo ingest')\n",
        encoding="utf-8",
    )
    (repo_dir / "tests" / "test_app.py").write_text(
        "from src.app import main\n\n\ndef test_main() -> None:\n    main()\n",
        encoding="utf-8",
    )
    (repo_dir / ".git" / "config").write_text(
        "[core]\nrepositoryformatversion = 0\n", encoding="utf-8"
    )

    result = wiki_ops.wiki_ingest_repo(
        repo_path=str(repo_dir),
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )

    assert result["status"] == "success"
    assert result["source_note_path"] == "repos/sample-repo.md"
    source_note = source_dir / "repos" / "sample-repo.md"
    assert source_note.exists()
    source_text = source_note.read_text(encoding="utf-8")
    assert "Derived from local repository" in source_text
    assert "## Snapshot Excerpt" in source_text
    raw_snapshot = raw_dir / "repos" / "sample-repo" / "snapshot.md"
    raw_manifest = raw_dir / "repos" / "sample-repo" / "manifest.json"
    assert raw_snapshot.exists()
    assert raw_manifest.exists()
    snapshot_text = raw_snapshot.read_text(encoding="utf-8")
    assert "src/app.py" in snapshot_text
    assert "hello from repo ingest" in snapshot_text
    assert ".git/config" not in snapshot_text
    assert any(
        suggestion["relative_path"] == "concepts/sample-repo-architecture.md"
        for suggestion in result["suggested_pages"]
    )
    assert result["repo_snapshot"]["test_file_count"] == 1
    assert result["build"]["document_count"] == 1


def test_wiki_build_refreshes_changed_repo_snapshot(tmp_path: Path) -> None:
    repo_dir = tmp_path / "sample_repo"
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nRepo schema.\n", encoding="utf-8")
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "tests").mkdir(parents=True)
    (repo_dir / "README.md").write_text("# Sample Repo\n\nA small service.\n", encoding="utf-8")
    (repo_dir / "pyproject.toml").write_text("[project]\nname = 'sample-repo'\n", encoding="utf-8")
    (repo_dir / "src" / "app.py").write_text(
        "def main() -> None:\n    print('hello from repo ingest')\n",
        encoding="utf-8",
    )

    initial_result = wiki_ops.wiki_ingest_repo(
        repo_path=str(repo_dir),
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )
    assert initial_result["status"] == "success"

    (repo_dir / "src" / "app.py").write_text(
        "def main() -> None:\n    print('hello from repo ingest (updated)')\n",
        encoding="utf-8",
    )

    build_result = wiki_ops.wiki_build(
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )
    assert build_result["status"] == "success"

    raw_snapshot = raw_dir / "repos" / "sample-repo" / "snapshot.md"
    assert "hello from repo ingest (updated)" in raw_snapshot.read_text(encoding="utf-8")

    source_note = source_dir / "sources" / "repos" / "sample-repo.md"
    assert source_note.exists()
    assert "hello from repo ingest (updated)" in source_note.read_text(encoding="utf-8")


def test_wiki_ingest_repo_clones_remote_reference(tmp_path: Path) -> None:
    origin_dir = tmp_path / "origin_repo"
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nRemote repo schema.\n", encoding="utf-8")
    (origin_dir / "src").mkdir(parents=True)
    (origin_dir / "README.md").write_text(
        "# Remote Repo\n\nRemote clone fixture.\n", encoding="utf-8"
    )
    (origin_dir / "src" / "main.py").write_text("print('remote repo')\n", encoding="utf-8")
    subprocess.run(["git", "init"], check=True, cwd=origin_dir, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        check=True,
        cwd=origin_dir,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Wiki Tests"],
        check=True,
        cwd=origin_dir,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], check=True, cwd=origin_dir, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        check=True,
        cwd=origin_dir,
        capture_output=True,
        text=True,
    )

    result = wiki_ops.wiki_ingest_repo(
        repo_path=origin_dir.as_uri(),
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )

    assert result["status"] == "success"
    assert result["repo_materialization"]["mode"] == "remote_clone"
    checkout_dir = raw_dir / "repos" / "origin-repo" / "checkout"
    assert checkout_dir.exists()
    assert (checkout_dir / "README.md").exists()
    assert (source_dir / "sources" / "repos" / "origin-repo.md").exists()


def test_wiki_cli_ingest_repo_command(tmp_path: Path) -> None:
    repo_dir = tmp_path / "cli_repo"
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nCLI repo schema.\n", encoding="utf-8")
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "README.md").write_text("# CLI Repo\n\nRepo ingest via CLI.\n", encoding="utf-8")
    (repo_dir / "src" / "main.py").write_text("print('cli repo')\n", encoding="utf-8")

    cli_path = Path(__file__).resolve().parents[1] / "skills" / "wiki" / "wiki_cli.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "ingest-repo",
            str(repo_dir),
            "--source-dir",
            str(source_dir),
            "--raw-dir",
            str(raw_dir),
            "--wiki-path",
            str(wiki_path),
            "--vault-dir",
            str(vault_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "success"
    assert (source_dir / "sources" / "repos" / "cli-repo.md").exists()


def test_wiki_search_repo_finds_matches_in_ingested_checkout(tmp_path: Path) -> None:
    repo_dir = tmp_path / "search_repo"
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nRepo search schema.\n", encoding="utf-8")
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "solver.cpp").write_text(
        "void run_branch_and_bound() {}\n",
        encoding="utf-8",
    )
    wiki_ops.wiki_ingest_repo(
        repo_path=str(repo_dir),
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )

    result = wiki_ops.wiki_search_repo(
        repo="search_repo",
        pattern="branch and bound",
        raw_dir=str(raw_dir),
    )

    assert result["status"] == "success"
    assert result["count"] >= 1
    assert result["results"][0]["relative_path"] == "src/solver.cpp"


def test_wiki_cli_search_repo_command(tmp_path: Path) -> None:
    repo_dir = tmp_path / "cli_search_repo"
    raw_dir = tmp_path / "raw"
    source_dir = tmp_path / "wiki_source"
    wiki_path = tmp_path / "wiki.md"
    vault_dir = tmp_path / "wiki"
    (tmp_path / "AGENTS.md").write_text("# Schema\n\nCLI repo search schema.\n", encoding="utf-8")
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "search.py").write_text(
        "def branch_and_bound():\n    return 'branch and bound'\n",
        encoding="utf-8",
    )
    wiki_ops.wiki_ingest_repo(
        repo_path=str(repo_dir),
        source_dir=str(source_dir),
        raw_dir=str(raw_dir),
        wiki_path=str(wiki_path),
        vault_dir=str(vault_dir),
    )

    cli_path = Path(__file__).resolve().parents[1] / "skills" / "wiki" / "wiki_cli.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "search-repo",
            "cli_search_repo",
            "branch and bound",
            "--raw-dir",
            str(raw_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "success"
    assert payload["count"] >= 1


def test_wiki_cli_read_source_command(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki_source"
    source_dir.mkdir(parents=True)
    (source_dir / "sources").mkdir(parents=True)
    (source_dir / "sources" / "note.md").write_text(
        "# Note\n\nDirect source read.\n", encoding="utf-8"
    )

    cli_path = Path(__file__).resolve().parents[1] / "skills" / "wiki" / "wiki_cli.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "read-source",
            "sources/note.md",
            "--source-dir",
            str(source_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "success"
    assert payload["document"]["title"] == "Note"
