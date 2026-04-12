from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skills.wiki.wiki_skills.scripts import wiki_ops


class _WikiHelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Help formatter that preserves examples and shows defaults."""

    pass


def _print_result(result: dict[str, Any]) -> int:
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    status = str(result.get("status", "")).lower()
    return 0 if status in {"ok", "success", "warning"} else 1


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", help="Path to the maintained wiki source tree (`source/`).")
    parser.add_argument(
        "--wiki-path", help="Path to the generated monolithic wiki export (`wiki.md`)."
    )
    parser.add_argument("--raw-dir", help="Path to the raw artifact workspace (`raw/`).")
    parser.add_argument("--vault-dir", help="Path to the compiled browsing workspace (`dist/`).")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wiki",
        description=(
            "Maintain and query the local markdown wiki knowledge base.\n\n"
            "The CLI works across three layers:\n"
            "- raw artifacts under `raw/`\n"
            "- maintained notes under `source/`\n"
            "- generated outputs in `wiki.md` and `dist/`\n"
        ),
        epilog=textwrap.dedent(
            """\
            Common workflows:
              wiki recreate --source-dir wiki/source --raw-dir wiki/raw --wiki-path wiki/wiki.md --vault-dir wiki/dist
              wiki ingest-repo https://github.com/ERGO-Code/HiGHS --source-dir wiki/source --raw-dir wiki/raw --wiki-path wiki/wiki.md --vault-dir wiki/dist
              wiki search-repo highs "branch and bound" --raw-dir wiki/raw
              wiki read-source repos/scip.md --source-dir wiki/source
              wiki promote-suggestion repos/highs.md concepts/highs-architecture.md --source-dir wiki/source --raw-dir wiki/raw --wiki-path wiki/wiki.md --vault-dir wiki/dist
            """
        ),
        formatter_class=_WikiHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="command",
        description="Use one of the commands below to build, inspect, ingest, search, or maintain the wiki.",
    )

    list_parser = subparsers.add_parser(
        "list",
        help="List maintained source notes in source/.",
        description="List markdown source notes that the wiki compiler can build into wiki.md and dist/.",
        formatter_class=_WikiHelpFormatter,
    )
    list_parser.add_argument("--path", help="Override the wiki source directory to scan.")

    list_raw_parser = subparsers.add_parser(
        "list-raw",
        help="List collected raw artifacts under raw/.",
        description="Inspect the raw artifact inventory before promoting files or repositories into maintained notes.",
        formatter_class=_WikiHelpFormatter,
    )
    list_raw_parser.add_argument("--raw-dir", help="Raw artifact directory to inspect.")

    build_parser = subparsers.add_parser(
        "build",
        help="Build wiki.md and the compiled dist workspace.",
        description=(
            "Compile source/ and raw/ into the monolithic wiki.md export and the browsable dist/ workspace. "
            "A MkDocs website under dist/site/ is generated when MkDocs is installed."
        ),
        formatter_class=_WikiHelpFormatter,
    )
    _add_build_args(build_parser)

    recreate_parser = subparsers.add_parser(
        "recreate",
        help="Alias for build.",
        description="Recreate wiki.md and the compiled dist/ workspace from the current source/ and raw/ trees.",
        formatter_class=_WikiHelpFormatter,
    )
    _add_build_args(recreate_parser)

    load_parser = subparsers.add_parser(
        "load",
        help="Load wiki.md plus its manifest metadata.",
        description="Read the generated wiki export, including the embedded manifest used for indexed retrieval.",
        formatter_class=_WikiHelpFormatter,
    )
    load_parser.add_argument("--wiki-path", help="Path to the generated wiki.md file.")

    list_sources_parser = subparsers.add_parser(
        "list-sources",
        help="List documents currently indexed in wiki.md.",
        description="Inspect what is already present in the generated wiki index without loading full document bodies.",
        formatter_class=_WikiHelpFormatter,
    )
    list_sources_parser.add_argument("--wiki-path", help="Path to the generated wiki.md file.")

    read_source_parser = subparsers.add_parser(
        "read-source",
        help="Read a source note directly from source/.",
        description=(
            "Fetch a maintained note directly from source/.\n"
            "Use this when the agent needs the current source text and should not wait for the user to paste it."
        ),
        formatter_class=_WikiHelpFormatter,
    )
    read_source_parser.add_argument(
        "relative_path", help="Relative path inside source/, such as `repos/scip.md`."
    )
    read_source_parser.add_argument("--source-dir", help="Source note root to read from.")

    search_parser = subparsers.add_parser(
        "search",
        help="Search the generated wiki index.",
        description="Search compiled wiki document content inside wiki.md. Use `search-repo` for code-level repository questions.",
        formatter_class=_WikiHelpFormatter,
    )
    search_parser.add_argument("pattern", help="Literal text or regex pattern to search for.")
    search_parser.add_argument("--wiki-path", help="Path to the generated wiki.md file.")
    search_parser.add_argument("--regex", action="store_true", help="Treat the pattern as regex.")
    search_parser.add_argument(
        "--case-sensitive", action="store_true", help="Match case exactly instead of ignoring case."
    )
    search_parser.add_argument(
        "--max-matches", type=int, default=10, help="Maximum matching documents to return."
    )
    search_parser.add_argument(
        "--context-lines", type=int, default=1, help="Context lines to include around each hit."
    )

    search_repo_parser = subparsers.add_parser(
        "search-repo",
        help="Search an ingested repository checkout.",
        description=(
            "Search the raw checkout of an ingested repository.\n"
            "This is the right command for code-level questions like symbols, algorithms, options, and comments."
        ),
        formatter_class=_WikiHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              wiki search-repo highs "branch and bound" --raw-dir wiki/raw
              wiki search-repo scip "constraint integer programming" --raw-dir wiki/raw
            """
        ),
    )
    search_repo_parser.add_argument(
        "repo", help="Repository identifier: slug, remote URL, or checkout path."
    )
    search_repo_parser.add_argument("pattern", help="Literal text or regex pattern to search for.")
    search_repo_parser.add_argument(
        "--raw-dir", help="Raw artifact directory containing `repos/<slug>/` snapshots."
    )
    search_repo_parser.add_argument(
        "--regex", action="store_true", help="Treat the pattern as regex."
    )
    search_repo_parser.add_argument(
        "--case-sensitive", action="store_true", help="Match case exactly instead of ignoring case."
    )
    search_repo_parser.add_argument(
        "--max-matches", type=int, default=10, help="Maximum matching files to return."
    )
    search_repo_parser.add_argument(
        "--max-hits-per-file", type=int, default=3, help="Maximum matching lines to keep per file."
    )
    search_repo_parser.add_argument(
        "--context-lines", type=int, default=1, help="Context lines to include around each hit."
    )

    get_parser = subparsers.add_parser(
        "get",
        help="Fetch one indexed document from wiki.md.",
        description="Read one compiled/indexed wiki document by id, relative path, or stem.",
        formatter_class=_WikiHelpFormatter,
    )
    get_parser.add_argument("identifier", help="Document id, relative path, or stem to fetch.")
    get_parser.add_argument("--wiki-path", help="Path to the generated wiki.md file.")

    add_doc_parser = subparsers.add_parser(
        "add-document",
        help="Create a maintained source note in source/.",
        description="Write a markdown source note into source/ and optionally rebuild the wiki immediately.",
        formatter_class=_WikiHelpFormatter,
    )
    add_doc_parser.add_argument(
        "relative_path", help="Relative markdown path to create under source/."
    )
    add_doc_parser.add_argument("content", help="Full document content to write.")
    _add_build_args(add_doc_parser)
    add_doc_parser.add_argument(
        "--no-rebuild", action="store_true", help="Skip rebuilding wiki.md and dist/ after writing."
    )

    add_raw_doc_parser = subparsers.add_parser(
        "add-raw-document",
        help="Create a raw text artifact under raw/.",
        description="Write text directly into the raw artifact workspace for later ingest.",
        formatter_class=_WikiHelpFormatter,
    )
    add_raw_doc_parser.add_argument("relative_path", help="Relative path to create under raw/.")
    add_raw_doc_parser.add_argument("content", help="Full raw text content to write.")
    add_raw_doc_parser.add_argument("--raw-dir", help="Raw artifact directory to write into.")

    add_file_parser = subparsers.add_parser(
        "add-file",
        help="Copy a local file into source/.",
        description="Promote an existing local file directly into the maintained source-note tree.",
        formatter_class=_WikiHelpFormatter,
    )
    add_file_parser.add_argument("path", help="Filesystem path of the file to copy into source/.")
    add_file_parser.add_argument(
        "--output-dir", help="Compatibility alias for the generated wiki path."
    )
    add_file_parser.add_argument(
        "--source-label", help="Override the destination filename inside source/."
    )
    add_file_parser.add_argument("--source-dir", help="Source note directory to write into.")
    add_file_parser.add_argument(
        "--raw-dir", help="Raw artifact directory for any follow-on rebuild."
    )
    add_file_parser.add_argument(
        "--vault-dir", help="Compiled dist workspace for any follow-on rebuild."
    )
    add_file_parser.add_argument(
        "--no-rebuild", action="store_true", help="Skip rebuilding wiki.md and dist/ after writing."
    )

    add_raw_file_parser = subparsers.add_parser(
        "add-raw-file",
        help="Copy a local file into raw/.",
        description="Store an arbitrary source artifact under raw/ so it can be ingested later.",
        formatter_class=_WikiHelpFormatter,
    )
    add_raw_file_parser.add_argument("path", help="Filesystem path of the file to copy into raw/.")
    add_raw_file_parser.add_argument("--raw-dir", help="Raw artifact directory to write into.")
    add_raw_file_parser.add_argument(
        "--target-name", help="Override the destination filename inside raw/."
    )

    add_raw_paper_parser = subparsers.add_parser(
        "add-raw-paper",
        help="Download or copy a paper PDF into raw/papers/ and ingest it.",
        description=(
            "Materialize a paper PDF under raw/papers/, extract text/metadata when possible, "
            "create a maintained source note, suggest follow-up pages, and optionally rebuild."
        ),
        formatter_class=_WikiHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              wiki add-raw-paper https://arxiv.org/pdf/2604.04921 --source-dir wiki/source --raw-dir wiki/raw --wiki-path wiki/wiki.md --vault-dir wiki/dist
              wiki add-raw-paper ./paper.pdf --paper-title "My Paper" --source-dir wiki/source --raw-dir wiki/raw
            """
        ),
    )
    add_raw_paper_parser.add_argument("paper", help="Paper PDF URL, arXiv URL/id, or local PDF path.")
    add_raw_paper_parser.add_argument("--paper-title", help="Override the generated paper note title.")
    add_raw_paper_parser.add_argument(
        "--source-note-path", help="Override the destination note path under source/."
    )
    add_raw_paper_parser.add_argument(
        "--target-name", help="Override the filename stem stored under raw/papers/."
    )
    add_raw_paper_parser.add_argument("--source-dir", help="Source note root to write into.")
    add_raw_paper_parser.add_argument("--raw-dir", help="Raw artifact root to write into.")
    add_raw_paper_parser.add_argument("--wiki-path", help="Path to the generated wiki.md file.")
    add_raw_paper_parser.add_argument("--vault-dir", help="Path to the compiled dist workspace.")
    add_raw_paper_parser.add_argument(
        "--no-rebuild", action="store_true", help="Skip rebuilding wiki.md and dist/ after ingest."
    )
    add_raw_paper_parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Maximum extracted PDF text length to keep in raw/papers/*.txt and the note excerpt.",
    )
    add_raw_paper_parser.add_argument(
        "--max-pages",
        type=int,
        default=12,
        help="Maximum PDF pages to inspect for text extraction.",
    )
    add_raw_paper_parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="Download timeout in seconds.",
    )
    add_raw_paper_parser.add_argument(
        "--max-bytes",
        type=int,
        default=100 * 1024 * 1024,
        help="Maximum PDF download/copy size in bytes.",
    )
    add_raw_paper_parser.add_argument(
        "--no-update-related",
        action="store_true",
        help="Do not append backlink updates to related existing notes.",
    )
    add_raw_paper_parser.add_argument(
        "--max-related-notes",
        type=int,
        default=3,
        help="Maximum related notes to link or update during ingest.",
    )
    add_raw_paper_parser.add_argument(
        "--max-suggested-pages",
        type=int,
        default=6,
        help="Maximum follow-up page suggestions to generate.",
    )

    ingest_raw_parser = subparsers.add_parser(
        "ingest-raw",
        help="Promote a raw artifact into a source-note scaffold.",
        description="Create a maintained source/ note from an existing raw artifact, suggest follow-up pages, and optionally rebuild.",
        formatter_class=_WikiHelpFormatter,
    )
    ingest_raw_parser.add_argument(
        "raw_relative_path",
        help="Relative artifact path inside raw/, such as `research/article.txt`.",
    )
    ingest_raw_parser.add_argument(
        "--source-note-path", help="Override the destination note path under source/."
    )
    ingest_raw_parser.add_argument("--title", help="Override the generated note title.")
    ingest_raw_parser.add_argument("--source-dir", help="Source note root to write into.")
    ingest_raw_parser.add_argument("--raw-dir", help="Raw artifact root to read from.")
    ingest_raw_parser.add_argument("--wiki-path", help="Path to the generated wiki.md file.")
    ingest_raw_parser.add_argument("--vault-dir", help="Path to the compiled dist workspace.")
    ingest_raw_parser.add_argument(
        "--no-rebuild", action="store_true", help="Skip rebuilding wiki.md and dist/ after ingest."
    )
    ingest_raw_parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Maximum raw excerpt length to embed in the scaffold note.",
    )
    ingest_raw_parser.add_argument(
        "--no-update-related",
        action="store_true",
        help="Do not append backlink updates to related existing notes.",
    )
    ingest_raw_parser.add_argument(
        "--max-related-notes",
        type=int,
        default=3,
        help="Maximum related notes to link or update during ingest.",
    )
    ingest_raw_parser.add_argument(
        "--max-suggested-pages",
        type=int,
        default=5,
        help="Maximum follow-up page suggestions to generate.",
    )

    ingest_repo_parser = subparsers.add_parser(
        "ingest-repo",
        help="Ingest a local repo or git remote into the wiki.",
        description=(
            "Snapshot a local repository, or clone a remote git repository into raw/ first,\n"
            "then create a maintained repo overview note in source/ and suggest follow-up pages."
        ),
        formatter_class=_WikiHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              wiki ingest-repo /path/to/repo --source-dir wiki/source --raw-dir wiki/raw --wiki-path wiki/wiki.md --vault-dir wiki/dist
              wiki ingest-repo https://github.com/ERGO-Code/HiGHS --source-dir wiki/source --raw-dir wiki/raw --wiki-path wiki/wiki.md --vault-dir wiki/dist
            """
        ),
    )
    ingest_repo_parser.add_argument(
        "repo_path", help="Local repository path or remote git URL to ingest."
    )
    ingest_repo_parser.add_argument(
        "--repo-name", help="Stable display name and slug override for the ingested repository."
    )
    ingest_repo_parser.add_argument(
        "--source-note-path", help="Override the destination repo note path under source/."
    )
    ingest_repo_parser.add_argument("--source-dir", help="Source note root to write into.")
    ingest_repo_parser.add_argument(
        "--raw-dir", help="Raw artifact root where repo snapshots and checkouts are stored."
    )
    ingest_repo_parser.add_argument("--wiki-path", help="Path to the generated wiki.md file.")
    ingest_repo_parser.add_argument("--vault-dir", help="Path to the compiled dist workspace.")
    ingest_repo_parser.add_argument(
        "--no-rebuild", action="store_true", help="Skip rebuilding wiki.md and dist/ after ingest."
    )
    ingest_repo_parser.add_argument(
        "--max-files",
        type=int,
        default=200,
        help="Maximum files to include in the catalog preview.",
    )
    ingest_repo_parser.add_argument(
        "--max-tree-entries",
        type=int,
        default=120,
        help="Maximum file tree entries to include in the raw snapshot.",
    )
    ingest_repo_parser.add_argument(
        "--max-excerpt-files",
        type=int,
        default=8,
        help="Maximum representative files to excerpt into the raw snapshot.",
    )
    ingest_repo_parser.add_argument(
        "--max-excerpt-chars",
        type=int,
        default=1200,
        help="Maximum characters per representative file excerpt.",
    )
    ingest_repo_parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Maximum raw snapshot excerpt length to embed in the repo note.",
    )
    ingest_repo_parser.add_argument(
        "--no-update-related",
        action="store_true",
        help="Do not append backlink updates to related existing notes.",
    )
    ingest_repo_parser.add_argument(
        "--max-related-notes",
        type=int,
        default=3,
        help="Maximum related notes to link or update during ingest.",
    )
    ingest_repo_parser.add_argument(
        "--max-suggested-pages",
        type=int,
        default=6,
        help="Maximum follow-up page suggestions to generate.",
    )

    list_suggestions_parser = subparsers.add_parser(
        "list-suggestions",
        help="List follow-up pages suggested by a source note.",
        description="Read the `Suggested Follow-Up Pages` section from a maintained source note.",
        formatter_class=_WikiHelpFormatter,
    )
    list_suggestions_parser.add_argument(
        "source_note_path", help="Relative path of the source note under source/."
    )
    list_suggestions_parser.add_argument("--source-dir", help="Source note root to read from.")

    promote_suggestion_parser = subparsers.add_parser(
        "promote-suggestion",
        help="Promote one suggested follow-up page.",
        description="Create or update one suggested follow-up page from a maintained source note, then optionally rebuild the wiki.",
        formatter_class=_WikiHelpFormatter,
    )
    promote_suggestion_parser.add_argument(
        "source_note_path", help="Relative path of the source note under source/."
    )
    promote_suggestion_parser.add_argument(
        "suggestion_path", help="Relative path of the suggested page to promote."
    )
    promote_suggestion_parser.add_argument(
        "--source-dir", help="Source note root to read from and write to."
    )
    promote_suggestion_parser.add_argument(
        "--wiki-path", help="Path to the generated wiki.md file."
    )
    promote_suggestion_parser.add_argument(
        "--raw-dir", help="Raw artifact directory for any follow-on rebuild."
    )
    promote_suggestion_parser.add_argument(
        "--vault-dir", help="Compiled dist workspace for any follow-on rebuild."
    )
    promote_suggestion_parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip rebuilding wiki.md and dist/ after promotion.",
    )

    update_doc_parser = subparsers.add_parser(
        "update-document",
        help="Modify an existing source note.",
        description="Replace, append, prepend, or fully overwrite a maintained source note in source/.",
        formatter_class=_WikiHelpFormatter,
    )
    update_doc_parser.add_argument(
        "relative_path", help="Relative path of the source note under source/."
    )
    update_doc_parser.add_argument("--content", help="Full replacement document content.")
    update_doc_parser.add_argument("--find-text", help="Existing text to find before replacing.")
    update_doc_parser.add_argument("--replace-text", help="Replacement text used with --find-text.")
    update_doc_parser.add_argument(
        "--replace-all", action="store_true", help="Replace all matches instead of only the first."
    )
    update_doc_parser.add_argument("--append-text", help="Append text to the end of the document.")
    update_doc_parser.add_argument(
        "--prepend-text", help="Prepend text to the beginning of the document."
    )
    _add_build_args(update_doc_parser)
    update_doc_parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip rebuilding wiki.md and dist/ after the edit.",
    )

    write_output_parser = subparsers.add_parser(
        "write-output",
        help="Write a derived output into dist/outputs/.",
        description="Store a reusable generated answer or artifact in the compiled workspace so it compounds over time.",
        formatter_class=_WikiHelpFormatter,
    )
    write_output_parser.add_argument("relative_path", help="Relative path under dist/outputs/.")
    write_output_parser.add_argument("content", help="Full output content to write.")
    write_output_parser.add_argument("--vault-dir", help="Compiled dist workspace to write into.")

    health_parser = subparsers.add_parser(
        "health",
        help="Check whether generated wiki artifacts are stale.",
        description="Compare source/, raw/, wiki.md, and dist/ to see whether a rebuild is needed.",
        formatter_class=_WikiHelpFormatter,
    )
    _add_build_args(health_parser)

    lint_parser = subparsers.add_parser(
        "lint",
        help="Run structural lint checks on the wiki.",
        description="Analyze source/ for structural issues such as short notes, contradictions, and weak linking.",
        formatter_class=_WikiHelpFormatter,
    )
    lint_parser.add_argument("--source-dir", help="Source note directory to lint.")
    lint_parser.add_argument(
        "--vault-dir", help="Compiled workspace where the lint report should be written."
    )
    lint_parser.add_argument(
        "--no-write-report", action="store_true", help="Skip writing `dist/reports/Lint.md`."
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Run health, lint, and optional search checks.",
        description="Convenience command for a quick wiki maintenance pass: health check, lint pass, and optional sample query.",
        formatter_class=_WikiHelpFormatter,
    )
    verify_parser.add_argument(
        "--query", help="Optional sample query to run after health and lint checks."
    )
    verify_parser.add_argument(
        "--output-dir", help="Compatibility alias for the generated wiki path."
    )
    verify_parser.add_argument("--source-dir", help="Source note directory to inspect.")
    verify_parser.add_argument("--raw-dir", help="Raw artifact directory to inspect.")
    verify_parser.add_argument("--vault-dir", help="Compiled workspace to inspect.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _print_result(wiki_ops.wiki_list(path=args.path))
    if args.command == "list-raw":
        return _print_result(wiki_ops.wiki_list_raw(raw_dir=args.raw_dir))
    if args.command == "build":
        return _print_result(
            wiki_ops.wiki_build(
                source_dir=args.source_dir,
                wiki_path=args.wiki_path,
                raw_dir=args.raw_dir,
                vault_dir=args.vault_dir,
            )
        )
    if args.command == "recreate":
        return _print_result(
            wiki_ops.wiki_recreate(
                source_dir=args.source_dir,
                wiki_path=args.wiki_path,
                raw_dir=args.raw_dir,
                vault_dir=args.vault_dir,
            )
        )
    if args.command == "load":
        return _print_result(wiki_ops.wiki_load_index(wiki_path=args.wiki_path))
    if args.command == "list-sources":
        return _print_result(wiki_ops.wiki_list_sources(wiki_path=args.wiki_path))
    if args.command == "read-source":
        return _print_result(
            wiki_ops.wiki_read_source_note(
                relative_path=args.relative_path,
                source_dir=args.source_dir,
            )
        )
    if args.command == "search":
        return _print_result(
            wiki_ops.wiki_search(
                pattern=args.pattern,
                literal=not args.regex,
                case_sensitive=args.case_sensitive,
                max_matches=args.max_matches,
                context_lines=args.context_lines,
                wiki_path=args.wiki_path,
            )
        )
    if args.command == "search-repo":
        return _print_result(
            wiki_ops.wiki_search_repo(
                repo=args.repo,
                pattern=args.pattern,
                raw_dir=args.raw_dir,
                literal=not args.regex,
                case_sensitive=args.case_sensitive,
                max_matches=args.max_matches,
                max_hits_per_file=args.max_hits_per_file,
                context_lines=args.context_lines,
            )
        )
    if args.command == "get":
        return _print_result(
            wiki_ops.wiki_get_document(identifier=args.identifier, wiki_path=args.wiki_path)
        )
    if args.command == "add-document":
        return _print_result(
            wiki_ops.wiki_add_document(
                relative_path=args.relative_path,
                content=args.content,
                source_dir=args.source_dir,
                wiki_path=args.wiki_path,
                raw_dir=args.raw_dir,
                vault_dir=args.vault_dir,
                rebuild=not args.no_rebuild,
            )
        )
    if args.command == "add-raw-document":
        return _print_result(
            wiki_ops.wiki_add_raw_document(
                relative_path=args.relative_path,
                content=args.content,
                raw_dir=args.raw_dir,
            )
        )
    if args.command == "add-file":
        return _print_result(
            wiki_ops.wiki_add_file(
                path=args.path,
                output_dir=args.output_dir,
                source_label=args.source_label,
                source_dir=args.source_dir,
                raw_dir=args.raw_dir,
                vault_dir=args.vault_dir,
                rebuild=not args.no_rebuild,
            )
        )
    if args.command == "add-raw-file":
        return _print_result(
            wiki_ops.wiki_add_raw_file(
                path=args.path,
                raw_dir=args.raw_dir,
                target_name=args.target_name,
            )
        )
    if args.command == "add-raw-paper":
        return _print_result(
            wiki_ops.wiki_add_raw_paper(
                paper=args.paper,
                paper_title=args.paper_title,
                source_note_path=args.source_note_path,
                target_name=args.target_name,
                source_dir=args.source_dir,
                raw_dir=args.raw_dir,
                wiki_path=args.wiki_path,
                vault_dir=args.vault_dir,
                rebuild=not args.no_rebuild,
                max_chars=args.max_chars,
                max_pages=args.max_pages,
                timeout_sec=args.timeout_sec,
                max_bytes=args.max_bytes,
                update_related=not args.no_update_related,
                max_related_notes=args.max_related_notes,
                max_suggested_pages=args.max_suggested_pages,
            )
        )
    if args.command == "ingest-raw":
        return _print_result(
            wiki_ops.wiki_ingest_raw(
                raw_relative_path=args.raw_relative_path,
                source_note_path=args.source_note_path,
                title=args.title,
                source_dir=args.source_dir,
                raw_dir=args.raw_dir,
                wiki_path=args.wiki_path,
                vault_dir=args.vault_dir,
                rebuild=not args.no_rebuild,
                max_chars=args.max_chars,
                update_related=not args.no_update_related,
                max_related_notes=args.max_related_notes,
                max_suggested_pages=args.max_suggested_pages,
            )
        )
    if args.command == "ingest-repo":
        return _print_result(
            wiki_ops.wiki_ingest_repo(
                repo_path=args.repo_path,
                repo_name=args.repo_name,
                source_note_path=args.source_note_path,
                source_dir=args.source_dir,
                raw_dir=args.raw_dir,
                wiki_path=args.wiki_path,
                vault_dir=args.vault_dir,
                rebuild=not args.no_rebuild,
                max_files=args.max_files,
                max_tree_entries=args.max_tree_entries,
                max_excerpt_files=args.max_excerpt_files,
                max_excerpt_chars=args.max_excerpt_chars,
                max_chars=args.max_chars,
                update_related=not args.no_update_related,
                max_related_notes=args.max_related_notes,
                max_suggested_pages=args.max_suggested_pages,
            )
        )
    if args.command == "list-suggestions":
        return _print_result(
            wiki_ops.wiki_list_suggestions(
                source_note_path=args.source_note_path,
                source_dir=args.source_dir,
            )
        )
    if args.command == "promote-suggestion":
        return _print_result(
            wiki_ops.wiki_promote_suggestion(
                source_note_path=args.source_note_path,
                suggestion_path=args.suggestion_path,
                source_dir=args.source_dir,
                wiki_path=args.wiki_path,
                raw_dir=args.raw_dir,
                vault_dir=args.vault_dir,
                rebuild=not args.no_rebuild,
            )
        )
    if args.command == "update-document":
        return _print_result(
            wiki_ops.wiki_update_document(
                relative_path=args.relative_path,
                content=args.content,
                find_text=args.find_text,
                replace_text=args.replace_text,
                replace_all=args.replace_all,
                append_text=args.append_text,
                prepend_text=args.prepend_text,
                source_dir=args.source_dir,
                wiki_path=args.wiki_path,
                raw_dir=args.raw_dir,
                vault_dir=args.vault_dir,
                rebuild=not args.no_rebuild,
            )
        )
    if args.command == "write-output":
        return _print_result(
            wiki_ops.wiki_write_output(
                relative_path=args.relative_path,
                content=args.content,
                vault_dir=args.vault_dir,
            )
        )
    if args.command == "health":
        return _print_result(
            wiki_ops.wiki_health(
                source_dir=args.source_dir,
                wiki_path=args.wiki_path,
                raw_dir=args.raw_dir,
                vault_dir=args.vault_dir,
            )
        )
    if args.command == "lint":
        return _print_result(
            wiki_ops.wiki_lint(
                source_dir=args.source_dir,
                vault_dir=args.vault_dir,
                write_report=not args.no_write_report,
            )
        )
    if args.command == "verify":
        return _print_result(
            wiki_ops.wiki_verify(
                query=args.query,
                output_dir=args.output_dir,
                source_dir=args.source_dir,
                raw_dir=args.raw_dir,
                vault_dir=args.vault_dir,
            )
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
