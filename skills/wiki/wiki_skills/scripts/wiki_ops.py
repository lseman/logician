"""Markdown-backed wiki operations for local, fast knowledge bases.

This module keeps a small-scale knowledge base entirely on disk:

- ``source/`` holds normalized markdown source notes
- ``raw/`` holds original artifacts and collected material
- ``wiki.md`` is the monolithic searchable index
- ``dist/`` is an Obsidian-friendly compiled workspace with articles,
  concepts, indexes, reports, and outputs

The goal is to preserve the speed and inspectability of plain markdown while
giving LLM agents a richer workspace to maintain over time.

By default the local wiki workspace lives under ``./wiki`` and repository
snapshots are stored under ``./wiki/raw/repos/``.
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from src.repo import query_repo_context

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

try:
    from rapidfuzz import fuzz as _rf_fuzz

    HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _rf_fuzz = None  # type: ignore
    HAS_RAPIDFUZZ = False

try:
    from tree_sitter import Parser  # type: ignore
except ImportError:  # pragma: no cover
    Parser = None

DEFAULT_ROOT_DIR = Path("wiki")
DEFAULT_SOURCE_DIR = DEFAULT_ROOT_DIR / "source"
DEFAULT_RAW_DIR = DEFAULT_ROOT_DIR / "raw"
DEFAULT_WIKI_PATH = DEFAULT_ROOT_DIR / "wiki.md"
DEFAULT_VAULT_DIR = DEFAULT_ROOT_DIR / "dist"
DEFAULT_SCHEMA_PATH = DEFAULT_ROOT_DIR / "AGENTS.md"
SUPPORTED_SUFFIXES = {".md", ".mdx", ".rst", ".txt"}
RAW_TEXT_SUFFIXES = SUPPORTED_SUFFIXES | {".json", ".csv", ".yaml", ".yml"}
MANIFEST_START = "<!-- WIKI_MANIFEST_START -->"
MANIFEST_END = "<!-- WIKI_MANIFEST_END -->"
DOC_START_TOKEN = "<!-- WIKI_DOC_START"
DOC_END_TOKEN = "<!-- WIKI_DOC_END"
MANAGED_VAULT_SUBDIRS = ("articles", "concepts", "indexes", "reports")
REPO_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".mypy_cache",
    ".next",
    ".nuxt",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "venv",
}
REPO_CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".kt",
    ".lua",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}
REPO_TEXT_SUFFIXES = RAW_TEXT_SUFFIXES | REPO_CODE_SUFFIXES
REPO_KEY_FILENAMES = {
    ".editorconfig",
    ".env.example",
    ".gitignore",
    ".python-version",
    "cargo.toml",
    "compose.yaml",
    "compose.yml",
    "docker-compose.yaml",
    "docker-compose.yml",
    "dockerfile",
    "makefile",
    "package.json",
    "pyproject.toml",
    "readme",
    "readme.md",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
}
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "around",
    "because",
    "been",
    "being",
    "between",
    "both",
    "can",
    "data",
    "does",
    "each",
    "easily",
    "from",
    "have",
    "into",
    "just",
    "keep",
    "keeps",
    "like",
    "more",
    "much",
    "need",
    "notes",
    "onto",
    "over",
    "page",
    "pages",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "using",
    "very",
    "when",
    "where",
    "which",
    "wiki",
    "with",
    "without",
    "your",
}

_DOC_BLOCK_RE = re.compile(
    r"<!-- WIKI_DOC_START (?P<meta>\{.*?\}) -->\n"
    r"(?P<content>.*?)(?:\n)?"
    r"<!-- WIKI_DOC_END (?P<end>\{.*?\}) -->",
    re.DOTALL,
)


@dataclass(frozen=True)
class WikiDocument:
    id: str
    title: str
    relative_path: str
    source_path: str
    sha256: str
    modified_at: str
    size_bytes: int
    line_count: int
    word_count: int
    headings: list[str]
    summary: str
    content: str
    keywords: tuple[str, ...]
    top_level: str

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "relative_path": self.relative_path,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "modified_at": self.modified_at,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "word_count": self.word_count,
            "headings": self.headings,
            "summary": self.summary,
            "keywords": list(self.keywords),
            "top_level": self.top_level,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_from_timestamp(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def _titleize(value: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[-_/]+", value) if part) or "Untitled"


def _normalize_relative_path(path: Path) -> str:
    return path.as_posix()


def _resolve_root_dir(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    candidate = Path("wiki")
    if candidate.exists():
        return candidate.resolve()
    fallback = Path("wiki_ingest")
    if fallback.exists():
        return fallback.resolve()
    return candidate.expanduser().resolve()


def _resolve_source_dir(path: str | Path | None = None) -> Path:
    return Path(path or DEFAULT_SOURCE_DIR).expanduser().resolve()


def _resolve_raw_dir(path: str | Path | None = None) -> Path:
    return Path(path or DEFAULT_RAW_DIR).expanduser().resolve()


def _resolve_wiki_path(path: str | Path | None = None) -> Path:
    raw = Path(path or DEFAULT_WIKI_PATH).expanduser()
    if raw.is_dir():
        wiki_md = raw / "wiki.md"
        if wiki_md.exists():
            return wiki_md.resolve()
        index_md = raw / "index.md"
        if index_md.exists():
            return index_md.resolve()
        return (raw / "wiki.md").resolve()
    if raw.suffix.lower() == ".md":
        return raw.resolve()
    return (raw / "wiki.md").resolve()


def _resolve_vault_dir(path: str | Path | None = None) -> Path:
    return Path(path or DEFAULT_VAULT_DIR).expanduser().resolve()


def _resolve_schema_path(path: str | Path | None = None) -> Path:
    return Path(path or DEFAULT_SCHEMA_PATH).expanduser().resolve()


def _infer_layout_root(
    *,
    source_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    wiki_path: str | Path | None = None,
    vault_dir: str | Path | None = None,
) -> Path:
    if source_dir is not None:
        return _resolve_source_dir(source_dir).parent
    if raw_dir is not None:
        return _resolve_raw_dir(raw_dir).parent
    if wiki_path is not None:
        return _resolve_wiki_path(wiki_path).parent
    if vault_dir is not None:
        return _resolve_vault_dir(vault_dir).parent
    return _resolve_root_dir()


def _infer_raw_dir(
    *,
    source_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    wiki_path: str | Path | None = None,
    vault_dir: str | Path | None = None,
) -> Path:
    if raw_dir is not None:
        return _resolve_raw_dir(raw_dir)
    return (
        _infer_layout_root(
            source_dir=source_dir,
            raw_dir=raw_dir,
            wiki_path=wiki_path,
            vault_dir=vault_dir,
        )
        / "raw"
    )


def _infer_source_dir(
    *,
    source_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    wiki_path: str | Path | None = None,
    vault_dir: str | Path | None = None,
) -> Path:
    if source_dir is not None:
        return _resolve_source_dir(source_dir)
    return (
        _infer_layout_root(
            source_dir=source_dir,
            raw_dir=raw_dir,
            wiki_path=wiki_path,
            vault_dir=vault_dir,
        )
        / "source"
    )


def _infer_wiki_path(
    *,
    source_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    wiki_path: str | Path | None = None,
    vault_dir: str | Path | None = None,
) -> Path:
    if wiki_path is not None:
        return _resolve_wiki_path(wiki_path)
    return (
        _infer_layout_root(
            source_dir=source_dir,
            raw_dir=raw_dir,
            wiki_path=wiki_path,
            vault_dir=vault_dir,
        )
        / "wiki.md"
    )


def _infer_vault_dir(
    *,
    source_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    wiki_path: str | Path | None = None,
    vault_dir: str | Path | None = None,
) -> Path:
    if vault_dir is not None:
        return _resolve_vault_dir(vault_dir)
    return (
        _infer_layout_root(
            source_dir=source_dir,
            raw_dir=raw_dir,
            wiki_path=wiki_path,
            vault_dir=vault_dir,
        )
        / "dist"
    )


def _infer_schema_path(
    *,
    source_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    wiki_path: str | Path | None = None,
    vault_dir: str | Path | None = None,
) -> Path:
    return (
        _infer_layout_root(
            source_dir=source_dir,
            raw_dir=raw_dir,
            wiki_path=wiki_path,
            vault_dir=vault_dir,
        )
        / "AGENTS.md"
    )


def _ensure_safe_target(root_dir: Path, relative_path: str | Path) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        raise ValueError("relative_path must stay inside the target directory")
    target = (root_dir / rel).resolve()
    try:
        target.relative_to(root_dir)
    except ValueError as exc:
        raise ValueError("relative_path escapes the target directory") from exc
    return target


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return any(part.startswith(".") for part in relative.parts)


def _iter_files(root_dir: Path, suffixes: set[str] | None = None) -> list[Path]:
    if not root_dir.exists():
        return []

    files = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        if _is_hidden(path, root_dir):
            continue
        files.append(path)
    return sorted(files, key=lambda item: _normalize_relative_path(item.relative_to(root_dir)))


def _search_markdown_directory(
    directory: Path,
    pattern: str,
    *,
    literal: bool,
    case_sensitive: bool,
    max_matches: int | None,
    context_lines: int,
) -> dict[str, Any]:
    if not directory.exists() or not directory.is_dir():
        return _error("Wiki directory not found", wiki_path=str(directory))

    results: list[dict[str, Any]] = []
    for path in _iter_files(directory, {".md"}):
        content = _read_text(path)
        line_hits = _search_lines(
            content,
            pattern,
            literal=literal,
            case_sensitive=case_sensitive,
        )
        if not line_hits:
            continue

        title = None
        for line in content.splitlines():
            match = re.match(r"^\s{0,3}#\s+(.*?)\s*$", line)
            if match:
                title = match.group(1).strip()
                break
        title = title or path.stem

        match_count = sum(len(hit["submatches"]) for hit in line_hits)
        trimmed_hits = line_hits[: max_matches or len(line_hits)]
        lines = content.splitlines()
        for hit in trimmed_hits:
            start = max(1, hit["line_number"] - context_lines)
            end = hit["line_number"] + context_lines
            hit["context"] = lines[start - 1 : end]

        results.append(
            {
                "id": _normalize_relative_path(path.relative_to(directory)),
                "title": title,
                "relative_path": _normalize_relative_path(path.relative_to(directory)),
                "match_count": match_count,
                "matches": trimmed_hits,
            }
        )

    results.sort(key=lambda item: (-item["match_count"], item["relative_path"]))
    if max_matches is not None:
        results = results[:max_matches]

    return _ok(
        wiki_path=str(directory),
        query=pattern,
        count=len(results),
        results=results,
    )


def _iter_source_files(source_dir: Path) -> list[Path]:
    return _iter_files(source_dir, SUPPORTED_SUFFIXES)


def _iter_raw_files(raw_dir: Path) -> list[Path]:
    return _iter_files(raw_dir, None)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_markdown_headings(content: str) -> list[str]:
    headings = []
    for line in content.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*$", line)
        if match:
            headings.append(match.group(1).strip())
    return headings


def _extract_rst_title(content: str) -> str | None:
    lines = content.splitlines()
    for index in range(len(lines) - 1):
        title = lines[index].strip()
        underline = lines[index + 1].strip()
        if not title or not underline:
            continue
        if len(underline) < len(title):
            continue
        if len(set(underline)) == 1 and underline[0] in "=-~^#*":
            return title
    return None


def _extract_title(path: Path, content: str) -> str:
    headings = _extract_markdown_headings(content)
    if headings:
        return headings[0]
    rst_title = _extract_rst_title(content)
    if rst_title:
        return rst_title
    return path.stem.replace("_", " ").replace("-", " ").strip() or path.name


def _summarize(content: str) -> str:
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("```"):
            continue
        return line[:220]
    return ""


def _word_count(content: str) -> int:
    return len(re.findall(r"\b\w+\b", content))


def _extract_keywords(title: str, headings: list[str], content: str) -> tuple[str, ...]:
    text = " ".join([title, *headings, _summarize(content), " ".join(content.splitlines()[:8])])
    counts = Counter(
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
        if token not in STOPWORDS and not token.isdigit()
    )
    return tuple(token for token, _ in counts.most_common(8))


def _build_document(path: Path, source_dir: Path) -> WikiDocument:
    content = _read_text(path)
    relative_path = _normalize_relative_path(path.relative_to(source_dir))
    stat_result = path.stat()
    headings = _extract_markdown_headings(content)
    keywords = _extract_keywords(_extract_title(path, content), headings, content)
    parts = Path(relative_path).parts
    top_level = parts[0] if len(parts) > 1 else "root"

    return WikiDocument(
        id=_slugify(relative_path),
        title=_extract_title(path, content),
        relative_path=relative_path,
        source_path=str(path),
        sha256=_sha256_text(content),
        modified_at=_iso_from_timestamp(stat_result.st_mtime),
        size_bytes=stat_result.st_size,
        line_count=len(content.splitlines()),
        word_count=_word_count(content),
        headings=headings,
        summary=_summarize(content),
        content=content.rstrip(),
        keywords=keywords,
        top_level=top_level,
    )


def _scan_documents(source_dir: Path) -> list[WikiDocument]:
    return [_build_document(path, source_dir) for path in _iter_source_files(source_dir)]


def _scan_raw_artifacts(raw_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in _iter_raw_files(raw_dir):
        relative_path = _normalize_relative_path(path.relative_to(raw_dir))
        suffix = path.suffix.lower()
        kind = "text" if suffix in RAW_TEXT_SUFFIXES else "binary"
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
            kind = "image"
        elif suffix in {".zip", ".tar", ".gz", ".xz"}:
            kind = "archive"
        elif suffix in {".ipynb", ".py"}:
            kind = "code"
        stat_result = path.stat()
        artifacts.append(
            {
                "relative_path": relative_path,
                "path": str(path),
                "kind": kind,
                "size_bytes": stat_result.st_size,
                "modified_at": _iso_from_timestamp(stat_result.st_mtime),
                "sha256": _sha256_file(path),
            }
        )
    return artifacts


def _build_manifest(
    source_dir: Path,
    raw_dir: Path,
    wiki_path: Path,
    vault_dir: Path,
    documents: list[WikiDocument],
    raw_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "generated_at": _now_iso(),
        "source_dir": str(source_dir),
        "raw_dir": str(raw_dir),
        "wiki_path": str(wiki_path),
        "vault_dir": str(vault_dir),
        "document_count": len(documents),
        "raw_artifact_count": len(raw_artifacts),
        "documents": [document.manifest_entry() for document in documents],
        "raw_artifacts": raw_artifacts,
    }


def _build_wiki_markdown(
    source_dir: Path,
    raw_dir: Path,
    wiki_path: Path,
    vault_dir: Path,
    documents: list[WikiDocument],
    raw_artifacts: list[dict[str, Any]],
) -> str:
    manifest = _build_manifest(source_dir, raw_dir, wiki_path, vault_dir, documents, raw_artifacts)
    lines = [
        "# Local Wiki",
        "",
        "> Generated file. Rebuild after changing files under `source` or `raw`.",
        "",
        f"- Source directory: `{source_dir}`",
        f"- Raw directory: `{raw_dir}`",
        f"- Compiled vault: `{vault_dir}`",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Documents: `{len(documents)}`",
        f"- Raw artifacts: `{len(raw_artifacts)}`",
        "",
        "## Index",
        "",
        "| Path | Title | Words | Updated |",
        "| --- | --- | ---: | --- |",
    ]
    for document in documents:
        lines.append(
            f"| `{document.relative_path}` | {document.title} | {document.word_count} | `{document.modified_at}` |"
        )
    if not documents:
        lines.append("| _none_ | _none_ | 0 | _n/a_ |")

    lines.extend(
        [
            "",
            "## Raw Inventory",
            "",
            "| Path | Kind | Size | Updated |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for artifact in raw_artifacts:
        lines.append(
            f"| `{artifact['relative_path']}` | {artifact['kind']} | {artifact['size_bytes']} | `{artifact['modified_at']}` |"
        )
    if not raw_artifacts:
        lines.append("| _none_ | _none_ | 0 | _n/a_ |")

    lines.extend(
        [
            "",
            "## Manifest",
            "",
            MANIFEST_START,
            "```json",
            json.dumps(manifest, indent=2, sort_keys=True),
            "```",
            MANIFEST_END,
            "",
            "## Documents",
            "",
        ]
    )

    for document in documents:
        start_meta = json.dumps(
            {"id": document.id, "relative_path": document.relative_path}, sort_keys=True
        )
        end_meta = json.dumps({"id": document.id}, sort_keys=True)
        lines.extend(
            [
                f"### {document.title}",
                "",
                f"- ID: `{document.id}`",
                f"- Path: `{document.relative_path}`",
                f"- SHA256: `{document.sha256}`",
                f"- Modified: `{document.modified_at}`",
            ]
        )
        if document.summary:
            lines.append(f"- Summary: {document.summary}")
        if document.headings:
            heading_preview = ", ".join(f"`{heading}`" for heading in document.headings[:6])
            lines.append(f"- Headings: {heading_preview}")
        if document.keywords:
            keyword_preview = ", ".join(f"`{keyword}`" for keyword in document.keywords[:8])
            lines.append(f"- Keywords: {keyword_preview}")
        lines.extend(
            [
                "",
                f"{DOC_START_TOKEN} {start_meta} -->",
            ]
        )
        if document.content:
            lines.append(document.content)
        lines.extend(
            [
                f"{DOC_END_TOKEN} {end_meta} -->",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _extract_manifest_block(content: str) -> str:
    try:
        start = content.index(MANIFEST_START) + len(MANIFEST_START)
        end = content.index(MANIFEST_END, start)
    except ValueError as exc:
        raise ValueError("wiki.md is missing manifest markers") from exc
    block = content[start:end].strip()
    if block.startswith("```json"):
        block = block[len("```json") :].strip()
    if block.endswith("```"):
        block = block[:-3].rstrip()
    return block


def _load_wiki_state(wiki_path: Path) -> dict[str, Any]:
    if not wiki_path.exists():
        raise FileNotFoundError(f"Wiki file not found: {wiki_path}")

    content = _read_text(wiki_path)
    manifest = json.loads(_extract_manifest_block(content))
    document_content: dict[str, str] = {}
    for match in _DOC_BLOCK_RE.finditer(content):
        meta = json.loads(match.group("meta"))
        document_content[meta["id"]] = match.group("content").rstrip("\n")

    documents = []
    for entry in manifest.get("documents", []):
        enriched = dict(entry)
        enriched["content"] = document_content.get(entry["id"], "")
        documents.append(enriched)

    return {"content": content, "manifest": manifest, "documents": documents}


def _extract_reference_tokens(content: str) -> set[str]:
    tokens = set()
    for match in re.finditer(r"\[\[([^\]|#]+)", content):
        tokens.add(match.group(1).strip().lower())
    for match in re.finditer(r"\]\(([^)]+)\)", content):
        tokens.add(match.group(1).strip().lower())
        tokens.add(Path(match.group(1).strip()).stem.lower())
    return {token for token in tokens if token}


def _build_relationships(documents: list[WikiDocument]) -> dict[str, Any]:
    docs_by_id = {document.id: document for document in documents}
    lookup: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        lookup[document.id].add(document.id)
        lookup[document.title.lower()].add(document.id)
        lookup[document.relative_path.lower()].add(document.id)
        lookup[Path(document.relative_path).stem.lower()].add(document.id)

    explicit_links: dict[str, set[str]] = defaultdict(set)
    backlinks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    related: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for document in documents:
        content_lower = document.content.lower()
        tokens = _extract_reference_tokens(document.content)
        targets: set[str] = set()
        for token in tokens:
            targets.update(lookup.get(token, set()))
        for other in documents:
            if other.id == document.id:
                continue
            if other.title.lower() in content_lower or other.relative_path.lower() in content_lower:
                targets.add(other.id)
        targets.discard(document.id)
        explicit_links[document.id].update(targets)

    for source_id, target_ids in explicit_links.items():
        source_doc = docs_by_id[source_id]
        for target_id in sorted(target_ids):
            backlinks[target_id].append(
                {
                    "id": source_doc.id,
                    "title": source_doc.title,
                    "relative_path": source_doc.relative_path,
                    "reason": "explicit reference",
                }
            )

    for left_index, left in enumerate(documents):
        for right in documents[left_index + 1 :]:
            shared = sorted(set(left.keywords) & set(right.keywords))
            score = 0
            reasons = []
            if left.top_level == right.top_level and left.top_level:
                score += 1
                reasons.append(f"shared section `{left.top_level}`")
            if shared:
                score += min(4, len(shared))
                reasons.append("shared keywords: " + ", ".join(shared[:4]))
            if right.id in explicit_links[left.id]:
                score += 3
                reasons.append("explicit reference from left")
            if left.id in explicit_links[right.id]:
                score += 3
                reasons.append("explicit reference from right")
            if score < 2:
                continue

            left_entry = {
                "id": right.id,
                "title": right.title,
                "relative_path": right.relative_path,
                "score": score,
                "reasons": reasons,
            }
            right_entry = {
                "id": left.id,
                "title": left.title,
                "relative_path": left.relative_path,
                "score": score,
                "reasons": reasons,
            }
            related[left.id].append(left_entry)
            related[right.id].append(right_entry)

    for document in documents:
        related[document.id] = sorted(
            related[document.id],
            key=lambda item: (-item["score"], item["relative_path"]),
        )[:6]
        backlinks[document.id] = sorted(
            backlinks[document.id], key=lambda item: item["relative_path"]
        )

    return {
        "explicit_links": {key: sorted(value) for key, value in explicit_links.items()},
        "backlinks": dict(backlinks),
        "related": dict(related),
    }


def _build_concepts(documents: list[WikiDocument]) -> list[dict[str, Any]]:
    concepts = []
    folder_map: dict[str, list[WikiDocument]] = defaultdict(list)
    for document in documents:
        folder_map[document.top_level].append(document)

    for folder, grouped_documents in sorted(folder_map.items()):
        concept_id = f"folder-{_slugify(folder)}"
        title = "Root Notes" if folder == "root" else _titleize(folder)
        concepts.append(
            {
                "id": concept_id,
                "title": title,
                "kind": "folder",
                "token": folder,
                "document_ids": [document.id for document in grouped_documents],
                "document_paths": [document.relative_path for document in grouped_documents],
                "summary": f"Documents grouped under the `{folder}` section.",
            }
        )

    keyword_docs: dict[str, list[WikiDocument]] = defaultdict(list)
    for document in documents:
        for keyword in document.keywords:
            keyword_docs[keyword].append(document)

    keyword_concepts = []
    for keyword, grouped_documents in keyword_docs.items():
        unique_documents = {document.id: document for document in grouped_documents}
        if len(unique_documents) < 2:
            continue
        keyword_concepts.append(
            {
                "id": f"keyword-{_slugify(keyword)}",
                "title": _titleize(keyword),
                "kind": "keyword",
                "token": keyword,
                "document_ids": [document.id for document in unique_documents.values()],
                "document_paths": [
                    document.relative_path for document in unique_documents.values()
                ],
                "summary": f"Cross-cutting topic inferred from repeated keyword `{keyword}`.",
                "weight": len(unique_documents),
            }
        )

    for concept in sorted(keyword_concepts, key=lambda item: (-item["weight"], item["title"]))[:12]:
        concept.pop("weight", None)
        concepts.append(concept)

    return sorted(concepts, key=lambda item: (item["kind"], item["title"]))


_NEGATION_TOKENS = {"not", "no", "never", "cannot", "can't", "wont", "won't", "without"}
_CLAIM_HINTS = {
    " is ",
    " are ",
    " was ",
    " were ",
    " should ",
    " must ",
    " can ",
    " cannot ",
    " can't ",
    " never ",
    " always ",
    " better ",
    " worse ",
}


def _sentence_candidates(content: str) -> list[str]:
    text = re.sub(r"`[^`]+`", " ", content)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    text = re.sub(r"\[\[[^\]]+\]\]", " ", text)
    candidates = []
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = chunk.strip(" -\t")
        if len(sentence) < 25:
            continue
        lowered = f" {sentence.lower()} "
        if not any(hint in lowered for hint in _CLAIM_HINTS):
            continue
        candidates.append(sentence)
    return candidates


def _normalize_claim(sentence: str) -> tuple[str, bool]:
    lowered = sentence.lower()
    negated = any(token in lowered.split() for token in _NEGATION_TOKENS)
    normalized = re.sub(r"[^a-z0-9\s]", " ", lowered)
    normalized = re.sub(r"\b(?:not|no|never|cannot|cant|won't|wont|without)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized, negated


def _find_contradiction_candidates(
    documents: list[WikiDocument],
    relationships: dict[str, Any],
) -> list[dict[str, Any]]:
    docs_by_id = {document.id: document for document in documents}
    findings: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    for document in documents:
        related_docs = relationships["related"].get(document.id, [])
        sentence_map: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)
        for sentence in _sentence_candidates(document.content):
            key, negated = _normalize_claim(sentence)
            if len(key) < 20:
                continue
            sentence_map[key].append((document.id, negated, sentence))

        for related in related_docs:
            other = docs_by_id.get(related["id"])
            if other is None:
                continue
            for sentence in _sentence_candidates(other.content):
                key, negated = _normalize_claim(sentence)
                if len(key) < 20 or key not in sentence_map:
                    continue
                for source_id, source_negated, source_sentence in sentence_map[key]:
                    if source_id == other.id or source_negated == negated:
                        continue
                    left_id, right_id = sorted((source_id, other.id))
                    pair_key = (left_id, right_id, key)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    findings.append(
                        {
                            "level": "warning",
                            "kind": "possible_contradiction",
                            "document_id": source_id,
                            "relative_path": docs_by_id[source_id].relative_path,
                            "other_document_id": other.id,
                            "other_relative_path": other.relative_path,
                            "message": (
                                "Related notes may disagree about the same claim. "
                                f"`{docs_by_id[source_id].relative_path}` says: {source_sentence[:140]} "
                                f"while `{other.relative_path}` says: {sentence[:140]}"
                            ),
                        }
                    )
    return findings


def _build_lint_report(
    documents: list[WikiDocument],
    relationships: dict[str, Any],
    concepts: list[dict[str, Any]],
) -> dict[str, Any]:
    concept_membership: dict[str, int] = defaultdict(int)
    for concept in concepts:
        for document_id in concept["document_ids"]:
            concept_membership[document_id] += 1

    duplicate_titles = [
        title
        for title, count in Counter(document.title for document in documents).items()
        if count > 1
    ]
    findings = []
    for document in documents:
        if document.word_count < 40:
            findings.append(
                {
                    "level": "warning",
                    "document_id": document.id,
                    "relative_path": document.relative_path,
                    "kind": "short_document",
                    "message": "Document is short and may need more context for future queries.",
                }
            )
        if len(document.headings) <= 1:
            findings.append(
                {
                    "level": "info",
                    "document_id": document.id,
                    "relative_path": document.relative_path,
                    "kind": "light_structure",
                    "message": "Document has limited heading structure.",
                }
            )
        if not relationships["backlinks"].get(document.id) and not relationships["related"].get(
            document.id
        ):
            findings.append(
                {
                    "level": "warning",
                    "document_id": document.id,
                    "relative_path": document.relative_path,
                    "kind": "isolated_document",
                    "message": "Document has no backlinks or inferred related notes.",
                }
            )
        if concept_membership.get(document.id, 0) == 0:
            findings.append(
                {
                    "level": "info",
                    "document_id": document.id,
                    "relative_path": document.relative_path,
                    "kind": "no_concepts",
                    "message": "Document does not belong to any generated concepts.",
                }
            )

    for title in duplicate_titles:
        findings.append(
            {
                "level": "warning",
                "kind": "duplicate_title",
                "message": f"Multiple documents share the title `{title}`.",
            }
        )

    contradiction_candidates = _find_contradiction_candidates(documents, relationships)
    findings.extend(contradiction_candidates)

    return {
        "status": "success" if not findings else "warning",
        "document_count": len(documents),
        "finding_count": len(findings),
        "duplicate_titles": duplicate_titles,
        "contradiction_count": len(contradiction_candidates),
        "findings": findings,
    }


def _article_wikilink(document: WikiDocument | dict[str, Any]) -> str:
    return f"[[articles/{document['id'] if isinstance(document, dict) else document.id}|{document['title'] if isinstance(document, dict) else document.title}]]"


def _concept_wikilink(concept: dict[str, Any]) -> str:
    return f"[[concepts/{concept['id']}|{concept['title']}]]"


def _markdown_link(from_path: Path, to_path: Path, label: str) -> str:
    return f"[{label}]({Path(os.path.relpath(to_path, start=from_path.parent)).as_posix()})"


MKDOCS_CONFIG_NAME = "mkdocs.yml"
MKDOCS_SITE_DIR_NAME = "site"
MKDOCS_DOCS_DIR_NAME = "site_docs"
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]")


def _html_safe_wikilink(match: re.Match[str]) -> str:
    target = match.group(1).strip()
    label = (match.group(2) or target).strip()
    target_part, _, anchor = target.partition("#")
    href = target_part if target_part.lower().endswith(".md") else f"{target_part}.md"
    href = href.replace(" ", "%20")
    if anchor:
        href = f"{href}#{quote(anchor, safe='')}"
    return f"[{label}]({href})"


def _convert_wikilinks_for_html(content: str) -> str:
    return _WIKILINK_RE.sub(_html_safe_wikilink, content)


def _clear_generated_markdown(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.rglob("*.md"):
        path.unlink()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _append_log_entry(
    vault_dir: Path, action: str, title: str, details: list[str] | None = None
) -> Path:
    log_path = vault_dir / "log.md"
    header = "\n".join(
        [
            "# Log",
            "",
            "> Append-only timeline of wiki ingests, queries, rebuilds, lint passes, and filed outputs.",
            "",
        ]
    )
    entry_lines = [f"## [{_now_iso()}] {action} | {title}", ""]
    for detail in details or ["No details recorded."]:
        entry_lines.append(f"- {detail}")
    entry_text = "\n".join(entry_lines).rstrip() + "\n"

    if log_path.exists():
        existing = _read_text(log_path).rstrip()
        combined = f"{existing}\n\n{entry_text}"
    else:
        combined = f"{header}{entry_text}"
    _write_text(log_path, combined)
    return log_path


def _build_home_page(
    documents: list[WikiDocument],
    raw_artifacts: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    lint_report: dict[str, Any],
) -> str:
    lines = [
        "# Knowledge Base Home",
        "",
        "> This workspace is maintained by the wiki compiler as a persistent, compounding artifact for LLM-assisted knowledge work.",
        "",
        "## Navigation",
        "",
        "- [[index]]",
        "- [[schema]]",
        "- [[log]]",
        "- [[indexes/Documents]]",
        "- [[indexes/Concepts]]",
        "- [[indexes/Backlinks]]",
        "- [[indexes/Raw Sources]]",
        "- [[reports/Health]]",
        "- [[reports/Lint]]",
        "- [[outputs/README]]",
        "",
        "## Snapshot",
        "",
        f"- Compiled notes: `{len(documents)}`",
        f"- Raw artifacts: `{len(raw_artifacts)}`",
        f"- Concepts: `{len(concepts)}`",
        f"- Lint findings: `{lint_report['finding_count']}`",
        "",
        "## Workflow",
        "",
        "1. Collect material into `raw/` or normalize markdown into `source/`.",
        "2. Rebuild the workspace so summaries, indexes, and backlinks stay fresh.",
        "3. Ask questions against the wiki and file useful outputs back into `outputs/`.",
        "4. Run health and lint passes to keep the graph coherent over time.",
    ]
    return "\n".join(lines)


def _build_root_index(
    documents: list[WikiDocument],
    raw_artifacts: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    health_report: dict[str, Any],
    lint_report: dict[str, Any],
) -> str:
    sections: dict[str, list[WikiDocument]] = defaultdict(list)
    for document in documents:
        sections[document.top_level].append(document)

    lines = [
        "# Index",
        "",
        "> Content-oriented catalog for the maintained wiki. Read this first, then drill into the linked pages.",
        "",
        "## Core Pages",
        "",
        "- [[Home]]: workspace overview, counts, and quick navigation.",
        "- [[schema]]: operating rules for how the LLM should maintain this wiki.",
        "- [[log]]: append-only timeline of ingests, queries, and maintenance work.",
        f"- [[reports/Health]]: current build health; healthy=`{health_report['healthy']}`.",
        f"- [[reports/Lint]]: structural findings; count=`{lint_report['finding_count']}`.",
        "- [[outputs/README]]: filed analyses and reusable derived artifacts.",
        "",
        "## Source Note Catalog",
        "",
    ]
    if not documents:
        lines.append("- No wiki source notes yet.")
    for section_name in sorted(sections):
        title = "Root" if section_name == "root" else _titleize(section_name)
        lines.extend([f"### {title}", ""])
        for document in sorted(sections[section_name], key=lambda item: item.relative_path):
            summary = document.summary or "No summary yet."
            lines.append(
                f"- {_article_wikilink(document)}: {summary} (`{document.relative_path}`, {document.word_count} words)"
            )
        lines.append("")

    lines.extend(["## Concepts", ""])
    if concepts:
        for concept in concepts:
            lines.append(
                f"- {_concept_wikilink(concept)}: {concept['summary']} ({len(concept['document_ids'])} documents)"
            )
    else:
        lines.append("- No generated concepts yet.")

    lines.extend(["", "## Raw Sources", ""])
    if raw_artifacts:
        for artifact in raw_artifacts:
            lines.append(
                f"- `{artifact['relative_path']}`: {artifact['kind']} source updated `{artifact['modified_at']}`"
            )
    else:
        lines.append("- No raw sources collected yet.")
    return "\n".join(lines)


def _build_documents_index(documents: list[WikiDocument]) -> str:
    lines = [
        "# Documents",
        "",
        "| Article | Source Path | Words | Keywords |",
        "| --- | --- | ---: | --- |",
    ]
    for document in documents:
        keywords = ", ".join(document.keywords[:4]) if document.keywords else "_none_"
        lines.append(
            f"| {_article_wikilink(document)} | `{document.relative_path}` | {document.word_count} | {keywords} |"
        )
    if not documents:
        lines.append("| _none_ | _none_ | 0 | _none_ |")
    return "\n".join(lines)


def _build_backlinks_index(documents: list[WikiDocument], relationships: dict[str, Any]) -> str:
    lines = ["# Backlinks", ""]
    for document in documents:
        lines.append(f"## {document.title}")
        lines.append("")
        backlinks = relationships["backlinks"].get(document.id, [])
        related = relationships["related"].get(document.id, [])
        if backlinks:
            lines.append("### Explicit Backlinks")
            lines.append("")
            for backlink in backlinks:
                lines.append(f"- [[articles/{backlink['id']}|{backlink['title']}]]")
            lines.append("")
        if related:
            lines.append("### Related Notes")
            lines.append("")
            for item in related:
                reason = "; ".join(item["reasons"][:2])
                lines.append(f"- [[articles/{item['id']}|{item['title']}]]: {reason}")
            lines.append("")
        if not backlinks and not related:
            lines.append("- No backlinks or related notes yet.")
            lines.append("")
    return "\n".join(lines).rstrip()


def _build_concepts_index(concepts: list[dict[str, Any]]) -> str:
    lines = [
        "# Concepts",
        "",
        "| Concept | Kind | Documents |",
        "| --- | --- | ---: |",
    ]
    for concept in concepts:
        lines.append(
            f"| {_concept_wikilink(concept)} | `{concept['kind']}` | {len(concept['document_ids'])} |"
        )
    if not concepts:
        lines.append("| _none_ | _none_ | 0 |")
    return "\n".join(lines)


def _build_raw_index(raw_artifacts: list[dict[str, Any]]) -> str:
    lines = [
        "# Raw Sources",
        "",
        "| Path | Kind | Size | Updated |",
        "| --- | --- | ---: | --- |",
    ]
    for artifact in raw_artifacts:
        lines.append(
            f"| `{artifact['relative_path']}` | {artifact['kind']} | {artifact['size_bytes']} | `{artifact['modified_at']}` |"
        )
    if not raw_artifacts:
        lines.append("| _none_ | _none_ | 0 | _n/a_ |")
    return "\n".join(lines)


def _build_health_report_page(health_report: dict[str, Any]) -> str:
    lines = [
        "# Health",
        "",
        f"- Healthy: `{health_report['healthy']}`",
        f"- Needs rebuild: `{health_report['needs_rebuild']}`",
        f"- Indexed documents: `{health_report['indexed_count']}`",
        f"- Source documents: `{health_report['source_count']}`",
        f"- Raw artifacts: `{health_report['raw_artifact_count']}`",
        "",
        "## Missing From Index",
        "",
    ]
    if health_report["missing_from_index"]:
        for item in health_report["missing_from_index"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Missing From Source", ""])
    if health_report["missing_from_source"]:
        for item in health_report["missing_from_source"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Stale Documents", ""])
    if health_report["stale_documents"]:
        for item in health_report["stale_documents"]:
            lines.append(f"- `{item['relative_path']}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Missing Workspace Pages", ""])
    if health_report["missing_workspace_pages"]:
        for item in health_report["missing_workspace_pages"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")
    return "\n".join(lines)


def _build_lint_report_page(lint_report: dict[str, Any]) -> str:
    lines = [
        "# Lint",
        "",
        f"- Status: `{lint_report['status']}`",
        f"- Documents scanned: `{lint_report['document_count']}`",
        f"- Findings: `{lint_report['finding_count']}`",
        f"- Possible contradictions: `{lint_report.get('contradiction_count', 0)}`",
        "",
        "## Findings",
        "",
    ]
    if lint_report["findings"]:
        for finding in lint_report["findings"]:
            if "relative_path" in finding:
                lines.append(
                    f"- `{finding['level']}` `{finding['kind']}` in `{finding['relative_path']}`: {finding['message']}"
                )
            else:
                lines.append(f"- `{finding['level']}` `{finding['kind']}`: {finding['message']}")
    else:
        lines.append("- No issues found.")
    return "\n".join(lines)


def _build_outputs_index(outputs_dir: Path, vault_dir: Path) -> str:
    lines = [
        "# Outputs",
        "",
        "> Place generated answers, briefs, slide decks, plots, and other derived artifacts here.",
        "",
    ]
    output_files = sorted(
        path
        for path in outputs_dir.rglob("*")
        if path.is_file() and not _is_hidden(path, outputs_dir)
    )
    if not output_files:
        lines.append("- No outputs filed yet.")
        return "\n".join(lines)

    for path in output_files:
        relative = _normalize_relative_path(path.relative_to(vault_dir))
        lines.append(f"- [[{relative.removesuffix('.md')}]]")
    return "\n".join(lines)


def _build_schema_page(schema_path: Path, vault_dir: Path) -> str:
    page_path = vault_dir / "schema.md"
    schema_link = _markdown_link(page_path, schema_path, "AGENTS.md")
    lines = [
        "# Schema",
        "",
        "> The canonical maintainer schema lives outside the compiled vault and tells the LLM how to ingest, query, file outputs, and lint the wiki.",
        "",
        f"- Canonical schema: {schema_link}",
        "",
        "## Operating Model",
        "",
        "- Raw sources are immutable inputs under `raw/`.",
        "- `source/` holds LLM-maintained source notes and summaries.",
        "- `dist/` is the compiled browsing workspace for Obsidian and query-time navigation.",
        "- Useful answers should be filed back into `dist/outputs/` so they compound over time.",
        "- `index.md` is the content-oriented table of contents; `log.md` is the chronological record.",
    ]
    if not schema_path.exists():
        lines.extend(
            [
                "",
                "## Status",
                "",
                f"- Expected schema file is missing at `{schema_path}`.",
            ]
        )
    return "\n".join(lines)


def _build_mkdocs_config(vault_dir: Path) -> str:
    return "\n".join(
        [
            "site_name: Logician Wiki",
            "docs_dir: site_docs",
            "site_dir: site",
            "theme:",
            "  name: readthedocs",
            "plugins:",
            "  - search",
            "nav:",
            "  - Home: Home.md",
            "  - Index: index.md",
            "  - Schema: schema.md",
            "  - Log: log.md",
            "  - Indexes:",
            "      - Documents: indexes/Documents.md",
            "      - Concepts: indexes/Concepts.md",
            "      - Backlinks: indexes/Backlinks.md",
            "      - Raw Sources: indexes/Raw Sources.md",
            "  - Reports:",
            "      - Health: reports/Health.md",
            "      - Lint: reports/Lint.md",
            "  - Outputs: outputs/README.md",
        ]
    )


def _build_site_markdown_files(vault_dir: Path, site_docs_dir: Path) -> None:
    if site_docs_dir.exists():
        shutil.rmtree(site_docs_dir)
    site_docs_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(vault_dir.rglob("*.md")):
        if source_path.is_dir():
            continue
        if source_path == vault_dir / MKDOCS_CONFIG_NAME:
            continue
        if source_path.is_relative_to(site_docs_dir):
            continue
        if source_path.is_relative_to(vault_dir / MKDOCS_SITE_DIR_NAME):
            continue
        relative_path = source_path.relative_to(vault_dir)
        target_path = site_docs_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        content = _read_text(source_path)
        html_safe = _convert_wikilinks_for_html(content)
        _write_text(target_path, html_safe)


def _build_mkdocs_site(vault_dir: Path) -> tuple[bool, str]:
    config_path = vault_dir / MKDOCS_CONFIG_NAME
    site_dir = vault_dir / MKDOCS_SITE_DIR_NAME
    mkdocs_executable = shutil.which("mkdocs")
    if not mkdocs_executable:
        return False, "MkDocs not installed"

    try:
        result = subprocess.run(
            [mkdocs_executable, "build", "-f", str(config_path), "-d", str(site_dir)],
            cwd=str(vault_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout.strip() or "MkDocs site generated successfully"
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or "MkDocs build failed"
    except FileNotFoundError:
        return False, "MkDocs executable not found"


def _build_article_page(
    document: WikiDocument,
    source_dir: Path,
    article_path: Path,
    relationships: dict[str, Any],
    concepts: list[dict[str, Any]],
) -> str:
    source_link = _markdown_link(article_path, source_dir / document.relative_path, "source note")
    document_concepts = [concept for concept in concepts if document.id in concept["document_ids"]]
    backlinks = relationships["backlinks"].get(document.id, [])
    related = relationships["related"].get(document.id, [])

    lines = [
        f"# {document.title}",
        "",
        "> Compiled article maintained by the wiki builder.",
        "",
        f"- ID: `{document.id}`",
        f"- Source: {source_link}",
        f"- Source path: `{document.relative_path}`",
        f"- Updated: `{document.modified_at}`",
        f"- Words: `{document.word_count}`",
        f"- Summary: {document.summary or 'No summary available.'}",
        "",
        "## Concepts",
        "",
    ]
    if document_concepts:
        for concept in document_concepts:
            lines.append(f"- {_concept_wikilink(concept)}")
    else:
        lines.append("- None yet.")

    lines.extend(["", "## Backlinks", ""])
    if backlinks:
        for backlink in backlinks:
            lines.append(f"- [[articles/{backlink['id']}|{backlink['title']}]]")
    else:
        lines.append("- No explicit backlinks yet.")

    lines.extend(["", "## Related Notes", ""])
    if related:
        for item in related:
            reason = "; ".join(item["reasons"][:2])
            lines.append(f"- [[articles/{item['id']}|{item['title']}]]: {reason}")
    else:
        lines.append("- No related notes inferred yet.")

    lines.extend(["", "## Headings", ""])
    if document.headings:
        for heading in document.headings:
            lines.append(f"- {heading}")
    else:
        lines.append("- No headings found.")

    lines.extend(["", "## Source Material", ""])
    lines.append(document.content or "_Empty source document._")
    return "\n".join(lines)


def _build_concept_page(concept: dict[str, Any], documents: dict[str, WikiDocument]) -> str:
    lines = [
        f"# {concept['title']}",
        "",
        f"- Kind: `{concept['kind']}`",
        f"- Token: `{concept['token']}`",
        f"- Summary: {concept['summary']}",
        "",
        "## Documents",
        "",
    ]
    for document_id in concept["document_ids"]:
        document = documents[document_id]
        lines.append(f"- {_article_wikilink(document)}")
    return "\n".join(lines)


def _build_vault_workspace(
    source_dir: Path,
    raw_dir: Path,
    vault_dir: Path,
    documents: list[WikiDocument],
    raw_artifacts: list[dict[str, Any]],
    relationships: dict[str, Any],
    concepts: list[dict[str, Any]],
    health_report: dict[str, Any],
    lint_report: dict[str, Any],
) -> None:
    vault_dir.mkdir(parents=True, exist_ok=True)
    for subdir_name in MANAGED_VAULT_SUBDIRS:
        subdir = vault_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        _clear_generated_markdown(subdir)

    outputs_dir = vault_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    schema_path = _infer_schema_path(source_dir=source_dir, raw_dir=raw_dir, vault_dir=vault_dir)
    documents_by_id = {document.id: document for document in documents}

    _write_text(
        vault_dir / "Home.md", _build_home_page(documents, raw_artifacts, concepts, lint_report)
    )
    _write_text(
        vault_dir / "index.md",
        _build_root_index(documents, raw_artifacts, concepts, health_report, lint_report),
    )
    _write_text(vault_dir / "schema.md", _build_schema_page(schema_path, vault_dir))
    _write_text(vault_dir / "indexes" / "Documents.md", _build_documents_index(documents))
    _write_text(
        vault_dir / "indexes" / "Backlinks.md", _build_backlinks_index(documents, relationships)
    )
    _write_text(vault_dir / "indexes" / "Concepts.md", _build_concepts_index(concepts))
    _write_text(vault_dir / "indexes" / "Raw Sources.md", _build_raw_index(raw_artifacts))
    _write_text(vault_dir / "reports" / "Health.md", _build_health_report_page(health_report))
    _write_text(vault_dir / "reports" / "Lint.md", _build_lint_report_page(lint_report))
    _write_text(outputs_dir / "README.md", _build_outputs_index(outputs_dir, vault_dir))

    for document in documents:
        article_path = vault_dir / "articles" / f"{document.id}.md"
        _write_text(
            article_path,
            _build_article_page(document, source_dir, article_path, relationships, concepts),
        )

    for concept in concepts:
        _write_text(
            vault_dir / "concepts" / f"{concept['id']}.md",
            _build_concept_page(concept, documents_by_id),
        )

    # Keep a tiny pointer in the workspace to the raw directory for easy browsing.
    raw_readme_path = vault_dir / "indexes" / "Raw Workspace.md"
    _write_text(
        raw_readme_path,
        "\n".join(
            [
                "# Raw Workspace",
                "",
                f"- Raw directory: `{raw_dir}`",
                "- Use [[indexes/Raw Sources]] for the generated inventory.",
                "- Store original articles, datasets, images, repos, or snapshots in `raw/`.",
            ]
        ),
    )

    _write_text(vault_dir / MKDOCS_CONFIG_NAME, _build_mkdocs_config(vault_dir))
    _build_site_markdown_files(vault_dir, vault_dir / MKDOCS_DOCS_DIR_NAME)
    mkdocs_built, mkdocs_message = _build_mkdocs_site(vault_dir)
    _write_text(
        vault_dir / "MKDOCS_BUILD_STATUS.md",
        "\n".join(
            [
                "# MkDocs Build Status",
                "",
                f"- MkDocs available: `{bool(shutil.which('mkdocs'))}`",
                f"- Site generated: `{mkdocs_built}`",
                "",
                "## Details",
                "",
                mkdocs_message,
            ]
        ),
    )


def _compute_health_report(
    source_dir: Path,
    raw_dir: Path,
    wiki_path: Path,
    vault_dir: Path,
    current_documents: list[WikiDocument],
) -> dict[str, Any]:
    if not wiki_path.exists():
        return {
            "status": "error",
            "healthy": False,
            "needs_rebuild": True,
            "source_dir": str(source_dir),
            "raw_dir": str(raw_dir),
            "wiki_path": str(wiki_path),
            "vault_dir": str(vault_dir),
            "indexed_count": 0,
            "source_count": len(current_documents),
            "raw_artifact_count": len(_scan_raw_artifacts(raw_dir)),
            "missing_from_index": [document.relative_path for document in current_documents],
            "missing_from_source": [],
            "stale_documents": [],
            "missing_workspace_pages": [],
        }

    try:
        state = _load_wiki_state(wiki_path)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "healthy": False,
            "needs_rebuild": True,
            "message": str(exc),
            "source_dir": str(source_dir),
            "raw_dir": str(raw_dir),
            "wiki_path": str(wiki_path),
            "vault_dir": str(vault_dir),
            "indexed_count": 0,
            "source_count": len(current_documents),
            "raw_artifact_count": len(_scan_raw_artifacts(raw_dir)),
            "missing_from_index": [],
            "missing_from_source": [],
            "stale_documents": [],
            "missing_workspace_pages": [],
        }

    current_map = {document.relative_path: document for document in current_documents}
    indexed_map = {document["relative_path"]: document for document in state["documents"]}
    missing_from_index = sorted(set(current_map) - set(indexed_map))
    missing_from_source = sorted(set(indexed_map) - set(current_map))
    stale_documents = []
    for relative_path in sorted(set(current_map) & set(indexed_map)):
        if current_map[relative_path].sha256 != indexed_map[relative_path]["sha256"]:
            stale_documents.append(
                {
                    "relative_path": relative_path,
                    "current_sha256": current_map[relative_path].sha256,
                    "indexed_sha256": indexed_map[relative_path]["sha256"],
                }
            )

    expected_pages = [
        vault_dir / "Home.md",
        vault_dir / "index.md",
        vault_dir / "schema.md",
        vault_dir / "log.md",
        vault_dir / "indexes" / "Documents.md",
        vault_dir / "indexes" / "Concepts.md",
        vault_dir / "indexes" / "Backlinks.md",
        vault_dir / "indexes" / "Raw Sources.md",
        vault_dir / "reports" / "Health.md",
        vault_dir / "reports" / "Lint.md",
        vault_dir / "outputs" / "README.md",
    ]
    missing_workspace_pages = [str(path) for path in expected_pages if not path.exists()]
    healthy = (
        not missing_from_index
        and not missing_from_source
        and not stale_documents
        and not missing_workspace_pages
    )

    return {
        "status": "success" if healthy else "warning",
        "healthy": healthy,
        "needs_rebuild": not healthy,
        "source_dir": str(source_dir),
        "raw_dir": str(raw_dir),
        "wiki_path": str(wiki_path),
        "vault_dir": str(vault_dir),
        "indexed_count": len(indexed_map),
        "source_count": len(current_map),
        "raw_artifact_count": len(_scan_raw_artifacts(raw_dir)),
        "missing_from_index": missing_from_index,
        "missing_from_source": missing_from_source,
        "stale_documents": stale_documents,
        "missing_workspace_pages": missing_workspace_pages,
    }


def _ok(**payload: Any) -> dict[str, Any]:
    return {"status": "success", **payload}


def _error(message: str, **payload: Any) -> dict[str, Any]:
    return {"status": "error", "message": message, **payload}


def wiki_list(path: str | None = None) -> dict[str, Any]:
    """List wiki source documents available for compilation."""
    source_dir = _resolve_source_dir(path)
    documents = _scan_documents(source_dir)
    return _ok(
        source_dir=str(source_dir),
        count=len(documents),
        documents=[document.manifest_entry() for document in documents],
    )


def wiki_list_raw(raw_dir: str | None = None) -> dict[str, Any]:
    """List artifacts collected in the raw directory."""
    resolved_raw_dir = _resolve_raw_dir(raw_dir)
    artifacts = _scan_raw_artifacts(resolved_raw_dir)
    return _ok(raw_dir=str(resolved_raw_dir), count=len(artifacts), artifacts=artifacts)


def wiki_build(
    source_dir: str | None = None,
    wiki_path: str | None = None,
    raw_dir: str | None = None,
    vault_dir: str | None = None,
) -> dict[str, Any]:
    """Build the monolithic wiki and the structured Obsidian-style workspace.

    When MkDocs is installed, the build also produces a static website under
    `dist/site/` with a generated `mkdocs.yml` configuration.
    """
    resolved_source_dir = _infer_source_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_raw_dir = _infer_raw_dir(
        source_dir=resolved_source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_wiki_path = _infer_wiki_path(
        source_dir=resolved_source_dir,
        raw_dir=resolved_raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_vault_dir = _infer_vault_dir(
        source_dir=resolved_source_dir,
        raw_dir=resolved_raw_dir,
        wiki_path=resolved_wiki_path,
        vault_dir=vault_dir,
    )

    resolved_source_dir.mkdir(parents=True, exist_ok=True)
    resolved_raw_dir.mkdir(parents=True, exist_ok=True)
    resolved_wiki_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_vault_dir.mkdir(parents=True, exist_ok=True)

    _refresh_ingested_repos_if_changed(
        resolved_raw_dir, resolved_source_dir, resolved_wiki_path, resolved_vault_dir
    )
    documents = _scan_documents(resolved_source_dir)
    raw_artifacts = _scan_raw_artifacts(resolved_raw_dir)
    _build_repo_search_engine_artifacts(resolved_raw_dir)
    relationships = _build_relationships(documents)
    concepts = _build_concepts(documents)
    lint_report = _build_lint_report(documents, relationships, concepts)

    rendered = _build_wiki_markdown(
        resolved_source_dir,
        resolved_raw_dir,
        resolved_wiki_path,
        resolved_vault_dir,
        documents,
        raw_artifacts,
    )
    resolved_wiki_path.write_text(rendered, encoding="utf-8")

    health_report = _compute_health_report(
        resolved_source_dir,
        resolved_raw_dir,
        resolved_wiki_path,
        resolved_vault_dir,
        documents,
    )

    _build_vault_workspace(
        resolved_source_dir,
        resolved_raw_dir,
        resolved_vault_dir,
        documents,
        raw_artifacts,
        relationships,
        concepts,
        health_report,
        lint_report,
    )

    health_report = _compute_health_report(
        resolved_source_dir,
        resolved_raw_dir,
        resolved_wiki_path,
        resolved_vault_dir,
        documents,
    )

    log_path = _append_log_entry(
        resolved_vault_dir,
        "build",
        "Wiki Rebuilt",
        [
            f"documents={len(documents)}",
            f"raw_artifacts={len(raw_artifacts)}",
            f"concepts={len(concepts)}",
            f"lint_findings={lint_report['finding_count']}",
        ],
    )

    health_report = _compute_health_report(
        resolved_source_dir,
        resolved_raw_dir,
        resolved_wiki_path,
        resolved_vault_dir,
        documents,
    )
    _write_text(
        resolved_vault_dir / "reports" / "Health.md", _build_health_report_page(health_report)
    )
    _write_text(
        resolved_vault_dir / "index.md",
        _build_root_index(documents, raw_artifacts, concepts, health_report, lint_report),
    )

    return _ok(
        source_dir=str(resolved_source_dir),
        raw_dir=str(resolved_raw_dir),
        wiki_path=str(resolved_wiki_path),
        vault_dir=str(resolved_vault_dir),
        mkdocs_config=str(resolved_vault_dir / MKDOCS_CONFIG_NAME),
        site_dir=str(resolved_vault_dir / MKDOCS_SITE_DIR_NAME),
        mkdocs_site_status=str(resolved_vault_dir / "MKDOCS_BUILD_STATUS.md"),
        log_path=str(log_path),
        document_count=len(documents),
        raw_artifact_count=len(raw_artifacts),
        concept_count=len(concepts),
        lint_finding_count=lint_report["finding_count"],
        documents=[document.manifest_entry() for document in documents],
        health=health_report,
    )


def wiki_recreate(
    source_dir: str | None = None,
    wiki_path: str | None = None,
    raw_dir: str | None = None,
    vault_dir: str | None = None,
) -> dict[str, Any]:
    """Use when: Recreate wiki.md and the compiled wiki workspace from the current source tree."""
    return wiki_build(
        source_dir=source_dir,
        wiki_path=wiki_path,
        raw_dir=raw_dir,
        vault_dir=vault_dir,
    )


def wiki_add_dir(
    directory: str,
    output_dir: str | None = None,
    glob: str = "**/*.{md,mdx,rst,txt}",
    chunk_size: int | None = None,
    overlap: float | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper that rebuilds the wiki from a source directory."""
    del glob, chunk_size, overlap
    return wiki_build(source_dir=directory, wiki_path=output_dir)


def wiki_load_index(wiki_path: str | None = None) -> dict[str, Any]:
    """Load the rendered wiki plus its manifest."""
    resolved_wiki_path = _resolve_wiki_path(wiki_path)
    try:
        state = _load_wiki_state(resolved_wiki_path)
    except FileNotFoundError:
        return _error("No wiki file found", wiki_path=str(resolved_wiki_path))
    except (ValueError, json.JSONDecodeError) as exc:
        return _error(str(exc), wiki_path=str(resolved_wiki_path))

    return _ok(
        wiki_path=str(resolved_wiki_path),
        content=state["content"],
        manifest=state["manifest"],
    )


def wiki_list_sources(wiki_path: str | None = None) -> dict[str, Any]:
    """List the documents recorded in the current wiki index."""
    resolved_wiki_path = _resolve_wiki_path(wiki_path)
    try:
        state = _load_wiki_state(resolved_wiki_path)
    except FileNotFoundError:
        return _error("No wiki file found", wiki_path=str(resolved_wiki_path))
    except (ValueError, json.JSONDecodeError) as exc:
        return _error(str(exc), wiki_path=str(resolved_wiki_path))

    return _ok(
        wiki_path=str(resolved_wiki_path),
        count=len(state["documents"]),
        documents=[
            {key: value for key, value in document.items() if key != "content"}
            for document in state["documents"]
        ],
    )


def wiki_read_source_note(
    relative_path: str,
    *,
    source_dir: str | None = None,
) -> dict[str, Any]:
    """Read one source note directly from source without relying on wiki.md."""
    resolved_source_dir = _infer_source_dir(source_dir=source_dir)
    target = _ensure_safe_target(resolved_source_dir, relative_path)
    if not target.exists() or not target.is_file():
        return _error("Source note not found", source_path=str(target), relative_path=relative_path)

    document = _build_document(target, resolved_source_dir)
    return _ok(
        source_dir=str(resolved_source_dir),
        source_path=str(target),
        relative_path=document.relative_path,
        document=document.manifest_entry() | {"content": document.content},
    )


def _search_lines(
    content: str,
    query: str,
    *,
    literal: bool,
    case_sensitive: bool,
    code_aware_literal: bool = False,
) -> list[dict[str, Any]]:
    matcher = _compile_search_matcher(
        query,
        literal=literal,
        case_sensitive=case_sensitive,
        code_aware_literal=code_aware_literal,
    )

    hits = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        found = list(matcher.finditer(line))
        if not found:
            continue
        hits.append(
            {
                "line_number": line_number,
                "line": line,
                "submatches": [
                    {"match": match.group(0), "start": match.start(), "end": match.end()}
                    for match in found
                ],
            }
        )
    if hits:
        return hits

    if not literal:
        for line_number, line in enumerate(content.splitlines(), start=1):
            if _fuzzy_contains(line, query, case_sensitive=case_sensitive):
                start = (line.lower() if not case_sensitive else line).find(
                    query.lower() if not case_sensitive else query
                )
                if start < 0:
                    start = 0
                end = start + len(query)
                hits.append(
                    {
                        "line_number": line_number,
                        "line": line,
                        "submatches": [{"match": query, "start": start, "end": end}],
                    }
                )
        return hits

    return hits


def _compile_search_matcher(
    query: str,
    *,
    literal: bool,
    case_sensitive: bool,
    code_aware_literal: bool = False,
) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    if literal and code_aware_literal:
        token_parts = [re.escape(part) for part in re.findall(r"[A-Za-z0-9]+", query)]
        if len(token_parts) >= 2:
            return re.compile(r"(?:[\W_]+)".join(token_parts), flags)
    pattern = re.escape(query) if literal else query
    return re.compile(pattern, flags)


def _fuzzy_contains(text: str, query: str, *, case_sensitive: bool) -> bool:
    if not case_sensitive:
        text = text.lower()
        query = query.lower()
    if not query or not text:
        return False
    if query in text:
        return True
    tokens = re.findall(r"\w+", query)
    if tokens and all(token in text for token in tokens):
        return True
    score = 0.0
    if HAS_RAPIDFUZZ and _rf_fuzz is not None:
        if hasattr(_rf_fuzz, "WRatio"):
            score = float(_rf_fuzz.WRatio(query, text)) / 100.0
        else:
            score = (
                max(
                    float(_rf_fuzz.token_set_ratio(query, text)),
                    float(_rf_fuzz.partial_ratio(query, text)),
                )
                / 100.0
            )
    else:
        score = SequenceMatcher(None, query, text).ratio()
    return score >= 0.65


def _find_line_for_query(
    content: str, query: str, *, case_sensitive: bool
) -> tuple[int, str, int] | None:
    needle = query if case_sensitive else query.lower()
    for line_number, line in enumerate(content.splitlines(), start=1):
        hay = line if case_sensitive else line.lower()
        if needle in hay:
            return line_number, line, hay.index(needle)
    return None


def _search_code_structure_hits(
    path: Path,
    content: str,
    query: str,
    *,
    literal: bool,
    case_sensitive: bool,
) -> list[dict[str, Any]]:
    structure = _extract_code_structure(path)
    if not structure:
        return []

    hits: list[dict[str, Any]] = []
    query_tokens = re.findall(r"\w+", query if case_sensitive else query.lower())

    def matches_text(value: str) -> bool:
        if not value:
            return False
        if literal:
            hay = value if case_sensitive else value.lower()
            needle = query if case_sensitive else query.lower()
            return needle in hay
        return _fuzzy_contains(value, query, case_sensitive=case_sensitive)

    def make_hit(
        line_number: int, line: str, match_text: str, start: int, end: int
    ) -> dict[str, Any]:
        return {
            "line_number": line_number,
            "line": line,
            "submatches": [{"match": match_text, "start": start, "end": end}],
        }

    def scan_field(field_text: str, match_text: str) -> None:
        if not field_text:
            return
        if not matches_text(field_text):
            return
        line_info = _find_line_for_query(
            content, match_text if literal else query, case_sensitive=case_sensitive
        )
        if line_info is not None:
            line_number, line, start = line_info
            hits.append(make_hit(line_number, line, match_text, start, start + len(match_text)))
        else:
            for line_number, line in enumerate(content.splitlines(), start=1):
                hay = line if case_sensitive else line.lower()
                if all(token in hay for token in query_tokens):
                    hits.append(make_hit(line_number, line, query, 0, len(query)))
                    break

    scan_field(structure.get("docstring", ""), query)
    for comment in structure.get("rationale_comments", []):
        scan_field(comment, query)
    for entry in structure.get("calls", []):
        scan_field(entry, entry)
    for entry in structure.get("functions", []):
        if isinstance(entry, dict):
            scan_field(entry.get("name", ""), entry.get("name", ""))
    for entry in structure.get("classes", []):
        if isinstance(entry, dict):
            scan_field(entry.get("name", ""), entry.get("name", ""))

    return hits


def wiki_search(
    pattern: str,
    path: str | None = None,
    literal: bool = True,
    case_sensitive: bool = False,
    max_matches: int | None = 10,
    context_lines: int = 1,
    wiki_path: str | None = None,
) -> dict[str, Any]:
    """Search the generated wiki for matching document content."""
    del path
    if not pattern:
        return _error("Search pattern cannot be empty")

    resolved_wiki_path = _resolve_wiki_path(wiki_path)
    try:
        state = _load_wiki_state(resolved_wiki_path)
        re.compile(
            re.escape(pattern) if literal else pattern, 0 if case_sensitive else re.IGNORECASE
        )
    except FileNotFoundError:
        if resolved_wiki_path.is_dir():
            return _search_markdown_directory(
                resolved_wiki_path,
                pattern,
                literal=literal,
                case_sensitive=case_sensitive,
                max_matches=max_matches,
                context_lines=context_lines,
            )
        return _error("No wiki file found", wiki_path=str(resolved_wiki_path))
    except (ValueError, json.JSONDecodeError, re.error) as exc:
        if resolved_wiki_path.is_dir():
            return _search_markdown_directory(
                resolved_wiki_path,
                pattern,
                literal=literal,
                case_sensitive=case_sensitive,
                max_matches=max_matches,
                context_lines=context_lines,
            )
        if resolved_wiki_path.name == "index.md":
            return _search_markdown_directory(
                resolved_wiki_path.parent,
                pattern,
                literal=literal,
                case_sensitive=case_sensitive,
                max_matches=max_matches,
                context_lines=context_lines,
            )
        return _error(str(exc), wiki_path=str(resolved_wiki_path))

    results = []
    for document in state["documents"]:
        lines = document["content"].splitlines()
        line_hits = _search_lines(
            document["content"], pattern, literal=literal, case_sensitive=case_sensitive
        )
        if not line_hits:
            continue
        match_count = sum(len(hit["submatches"]) for hit in line_hits)
        trimmed_hits = line_hits[: max_matches or len(line_hits)]
        for hit in trimmed_hits:
            start = max(1, hit["line_number"] - context_lines)
            end = hit["line_number"] + context_lines
            hit["context"] = lines[start - 1 : end]
        results.append(
            {
                "id": document["id"],
                "title": document["title"],
                "relative_path": document["relative_path"],
                "match_count": match_count,
                "matches": trimmed_hits,
            }
        )

    results.sort(key=lambda item: (-item["match_count"], item["relative_path"]))
    if max_matches is not None:
        results = results[:max_matches]

    if not results:
        repo_fallback = _wiki_search_repo_fallback(
            pattern,
            literal=literal,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
            context_lines=context_lines,
            max_hits_per_file=max_matches or 3,
        )
        if repo_fallback is not None:
            return repo_fallback

    result = _ok(
        wiki_path=str(resolved_wiki_path),
        query=pattern,
        count=len(results),
        results=results,
    )
    result["log_path"] = str(
        _append_log_entry(
            _infer_vault_dir(wiki_path=resolved_wiki_path),
            "query",
            pattern[:80],
            [f"matches={len(results)}", f"literal={literal}", f"case_sensitive={case_sensitive}"],
        )
    )
    return result


def wiki_search_repo(
    repo: str,
    pattern: str,
    *,
    raw_dir: str | None = None,
    literal: bool = True,
    case_sensitive: bool = False,
    max_matches: int | None = 10,
    max_hits_per_file: int = 3,
    context_lines: int = 1,
) -> dict[str, Any]:
    """Search an ingested repository checkout for matching code or text."""
    if not repo:
        return _error("Repository identifier cannot be empty")
    if not pattern:
        return _error("Search pattern cannot be empty")

    resolved_raw_dir = _infer_raw_dir(raw_dir=raw_dir)
    checkout_dir, repo_slug = _resolve_ingested_repo_checkout(repo, raw_dir=resolved_raw_dir)
    if checkout_dir is None:
        return _error(
            "Ingested repository checkout not found",
            repo=repo,
            expected_checkout=str(
                _ensure_safe_target(resolved_raw_dir, Path("repos") / repo_slug / "checkout")
            ),
        )

    try:
        _compile_search_matcher(
            pattern,
            literal=literal,
            case_sensitive=case_sensitive,
            code_aware_literal=True,
        )
    except re.error as exc:
        return _error(str(exc), repo=repo, checkout_path=str(checkout_dir))

    graph_search_used = False
    graph_candidates: list[str] = []
    graph_path = checkout_dir.parent / "search_graph.json"
    if graph_path.exists() and not literal:
        graph_results = query_repo_context(
            {"artifacts": {"graph_path": str(graph_path)}},
            query=pattern,
            limit=max(1, int(max_matches or 10) * 2),
        )
        graph_candidates = [path for path in graph_results.get("candidate_paths", []) if path]
        if graph_candidates:
            graph_search_used = True

    search_paths = []
    if graph_search_used:
        max_graph_candidates = max(4, min(20, int(max_matches or 10) * 2))
        search_paths = [
            checkout_dir / Path(path) for path in graph_candidates[:max_graph_candidates]
        ]
        search_paths = [path for path in search_paths if path.is_file()]
    if not search_paths:
        search_paths = list(_iter_repo_search_files(checkout_dir))

    results = []
    files_scanned = 0
    total_hits = 0
    for path in search_paths:
        try:
            content = _read_text(path)
        except OSError:
            continue
        files_scanned += 1
        line_hits = _search_lines(
            content,
            pattern,
            literal=literal,
            case_sensitive=case_sensitive,
            code_aware_literal=True,
        )
        if not line_hits and path.suffix.lower() in REPO_CODE_SUFFIXES:
            line_hits = _search_code_structure_hits(
                path,
                content,
                pattern,
                literal=literal,
                case_sensitive=case_sensitive,
            )
        if not line_hits:
            continue
        lines = content.splitlines()
        match_count = sum(len(hit["submatches"]) for hit in line_hits)
        total_hits += match_count
        trimmed_hits = line_hits[:max_hits_per_file]
        for hit in trimmed_hits:
            start = max(1, hit["line_number"] - context_lines)
            end = hit["line_number"] + context_lines
            hit["context"] = lines[start - 1 : end]
        results.append(
            {
                "relative_path": _normalize_relative_path(path.relative_to(checkout_dir)),
                "match_count": match_count,
                "matches": trimmed_hits,
            }
        )

    results.sort(key=lambda item: (-item["match_count"], item["relative_path"]))
    if max_matches is not None:
        results = results[:max_matches]

    result = _ok(
        repo=repo,
        repo_slug=repo_slug,
        raw_dir=str(resolved_raw_dir),
        checkout_path=str(checkout_dir),
        query=pattern,
        count=len(results),
        total_hits=total_hits,
        files_scanned=files_scanned,
        graph_search_used=graph_search_used,
        graph_candidates=graph_candidates[: max(1, int(max_matches or 10))],
        results=results,
    )
    result["log_path"] = str(
        _append_log_entry(
            _infer_vault_dir(raw_dir=resolved_raw_dir),
            "query",
            f"repo:{repo_slug} {pattern[:60]}",
            [
                f"repo={repo}",
                f"checkout={checkout_dir}",
                f"matches={len(results)}",
                f"total_hits={total_hits}",
                f"literal={literal}",
                f"case_sensitive={case_sensitive}",
            ],
        )
    )
    return result


def wiki_read_content(query: str, wiki_path: str | None = None) -> dict[str, Any]:
    """Compatibility alias for search."""
    return wiki_search(pattern=query, wiki_path=wiki_path)


def wiki_get_document(identifier: str, wiki_path: str | None = None) -> dict[str, Any]:
    """Fetch a document from the wiki by relative path or document id."""
    resolved_wiki_path = _resolve_wiki_path(wiki_path)
    try:
        state = _load_wiki_state(resolved_wiki_path)
    except FileNotFoundError:
        return _error("No wiki file found", wiki_path=str(resolved_wiki_path))
    except (ValueError, json.JSONDecodeError) as exc:
        return _error(str(exc), wiki_path=str(resolved_wiki_path))

    for document in state["documents"]:
        if identifier in {
            document["id"],
            document["relative_path"],
            Path(document["relative_path"]).stem,
        }:
            return _ok(wiki_path=str(resolved_wiki_path), document=document)
    return _error(
        "Document not found in wiki", wiki_path=str(resolved_wiki_path), identifier=identifier
    )


def _default_source_note_path_for_raw(raw_relative_path: str) -> str:
    raw_path = Path(raw_relative_path)
    stemmed = raw_path.with_suffix(".md")
    return _normalize_relative_path(Path("sources") / stemmed)


def _source_note_wikilink(relative_path: str, title: str | None = None) -> str:
    label = (
        title
        or Path(relative_path).stem.replace("-", " ").replace("_", " ").strip()
        or relative_path
    )
    return f"[[{Path(relative_path).with_suffix('').as_posix()}|{label}]]"


_GENERIC_SUGGESTION_TOKENS = {
    "article",
    "articles",
    "document",
    "documents",
    "note",
    "notes",
    "page",
    "pages",
    "research",
    "source",
    "sources",
    "summary",
    "repository",
    "repo",
    "snapshot",
    "codebase",
    "should",
}


def _raw_artifact_text_preview(path: Path, *, max_chars: int) -> str:
    suffix = path.suffix.lower()
    if suffix not in RAW_TEXT_SUFFIXES and suffix not in {".py", ".ipynb"}:
        return ""
    content = _read_text(path).strip()
    if len(content) > max_chars:
        return content[:max_chars].rstrip() + "\n\n[truncated]"
    return content


def _default_source_note_path_for_repo(repo_name: str) -> str:
    return _normalize_relative_path(Path("repos") / f"{_slugify(repo_name)}.md")


def _repo_name_from_local_path(path: Path) -> str:
    if path.name.lower() == "checkout" and path.parent.name:
        return path.parent.name
    return path.name


def _looks_like_repo_remote(value: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9+.-]*://", value)) or bool(
        re.match(r"^[^/\s@]+@[^:\s]+:.+$", value)
    )


def _repo_name_from_reference(reference: str) -> str:
    cleaned = reference.rstrip("/").split("/")[-1]
    if ":" in cleaned and "/" not in cleaned:
        cleaned = cleaned.split(":")[-1]
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.strip()
    return cleaned or "repository"


def _materialize_repo_for_ingest(
    repo_reference: str,
    *,
    raw_dir: Path,
    repo_name: str | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    if _looks_like_repo_remote(repo_reference):
        resolved_repo_name = repo_name or _repo_name_from_reference(repo_reference)
        repo_slug = _slugify(resolved_repo_name)
        managed_root = _ensure_safe_target(raw_dir, Path("repos") / repo_slug)
        checkout_path = managed_root / "checkout"
        if not checkout_path.exists():
            checkout_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                clone = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_reference, str(checkout_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                return (
                    checkout_path,
                    resolved_repo_name,
                    {
                        "status": "error",
                        "message": "Git is required to clone repository remotes",
                        "repo_reference": repo_reference,
                        "checkout_path": str(checkout_path),
                    },
                )
            if clone.returncode != 0:
                stderr = (clone.stderr or "").strip()
                stdout = (clone.stdout or "").strip()
                return (
                    checkout_path,
                    resolved_repo_name,
                    {
                        "status": "error",
                        "message": "Failed to clone repository remote",
                        "repo_reference": repo_reference,
                        "checkout_path": str(checkout_path),
                        "git_stdout": stdout,
                        "git_stderr": stderr,
                    },
                )
        return (
            checkout_path,
            resolved_repo_name,
            {
                "status": "success",
                "mode": "remote_clone",
                "repo_reference": repo_reference,
                "checkout_path": str(checkout_path),
            },
        )

    resolved_repo_path = Path(repo_reference).expanduser().resolve()
    resolved_repo_name = repo_name or _repo_name_from_local_path(resolved_repo_path)
    if not resolved_repo_path.exists() or not resolved_repo_path.is_dir():
        return (
            resolved_repo_path,
            resolved_repo_name,
            {
                "status": "error",
                "message": "Repository directory not found",
                "repo_path": str(resolved_repo_path),
            },
        )
    return (
        resolved_repo_path,
        resolved_repo_name,
        {
            "status": "success",
            "mode": "local_directory",
            "repo_path": str(resolved_repo_path),
        },
    )


def _resolve_ingested_repo_checkout(repo: str, *, raw_dir: Path) -> tuple[Path | None, str]:
    candidate = Path(repo).expanduser()
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve(), candidate.name

    normalized = repo
    if _looks_like_repo_remote(repo):
        normalized = _repo_name_from_reference(repo)
    repo_slug = _slugify(normalized)
    checkout_path = _ensure_safe_target(raw_dir, Path("repos") / repo_slug / "checkout")
    if checkout_path.exists() and checkout_path.is_dir():
        return checkout_path, repo_slug
    manifest_path = _ensure_safe_target(raw_dir, Path("repos") / repo_slug / "manifest.json")
    if manifest_path.exists() and manifest_path.is_file():
        try:
            manifest = json.loads(_read_text(manifest_path))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        repo_root = manifest.get("repo_root")
        if isinstance(repo_root, str):
            repo_root_path = Path(repo_root).expanduser()
            if repo_root_path.exists() and repo_root_path.is_dir():
                return repo_root_path.resolve(), repo_slug
    return None, repo_slug


def _repo_language_hint(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".css": "css",
        ".go": "go",
        ".h": "c",
        ".hpp": "cpp",
        ".html": "html",
        ".java": "java",
        ".js": "javascript",
        ".json": "json",
        ".md": "markdown",
        ".py": "python",
        ".rs": "rust",
        ".sh": "bash",
        ".sql": "sql",
        ".toml": "toml",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".zsh": "bash",
    }.get(suffix, "text")


def _tree_sitter_language_for_suffix(suffix: str) -> str | None:
    return {
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".c++": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".hh": "cpp",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".py": "python",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "c_sharp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
    }.get(suffix.lower())


def _get_tree_sitter_language(language: str) -> Any | None:
    if Parser is None:
        return None
    try:
        import tree_sitter_languages  # type: ignore

        if hasattr(tree_sitter_languages, "get_language"):
            return tree_sitter_languages.get_language(language)
    except ImportError:
        pass

    module_name = f"tree_sitter_{language}"
    try:
        module = __import__(module_name)
        if hasattr(module, "LANGUAGE"):
            return getattr(module, "LANGUAGE")
    except ImportError:
        pass

    return None


def _get_tree_sitter_parser(language: str) -> Any | None:
    ts_language = _get_tree_sitter_language(language)
    if ts_language is None or Parser is None:
        return None
    parser = Parser()
    parser.set_language(ts_language)
    return parser


def _tree_sitter_code_structure(path: Path) -> dict[str, Any] | None:
    language = _tree_sitter_language_for_suffix(path.suffix)
    parser = _get_tree_sitter_parser(language) if language else None
    if parser is None:
        return None

    text = _read_text(path)
    text_bytes = text.encode("utf-8")
    try:
        tree = parser.parse(text_bytes)
    except Exception:
        return None

    structure: dict[str, Any] = {
        "language": language or _repo_language_hint(path),
        "imports": [],
        "classes": [],
        "functions": [],
        "calls": [],
        "docstring": "",
        "rationale_comments": [],
    }

    def node_text(node: Any) -> str:
        try:
            return text_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace").strip()
        except Exception:
            return ""

    def node_identifier(node: Any) -> str | None:
        if node is None:
            return None
        if node.type == "identifier":
            return node_text(node)
        for child in node.children:
            found = node_identifier(child)
            if found:
                return found
        return None

    import_types = {
        "import_statement",
        "import_clause",
        "import_declaration",
        "using_directive",
        "namespace_import",
        "require_call",
    }
    class_types = {
        "class_definition",
        "class_declaration",
        "struct_specifier",
        "type_definition",
        "interface_declaration",
        "enum_declaration",
        "trait_definition",
    }
    function_types = {
        "function_definition",
        "function_declaration",
        "method_definition",
        "method_declarator",
        "arrow_function",
        "generator_function",
        "function_item",
    }
    call_types = {
        "call_expression",
        "call",
        "method_call_expression",
        "scoped_call_expression",
    }

    def add_import(node: Any) -> None:
        text_value = node_text(node)
        if text_value:
            structure["imports"].append(text_value.replace("\n", " "))

    def add_type_node(node: Any, target: str) -> None:
        name = node_identifier(
            node.child_by_field_name("name") or node.child_by_field_name("identifier") or node
        )
        if name:
            structure[target].append({"name": name, "docstring": ""})

    def add_call(node: Any) -> None:
        func = node.child_by_field_name("function") or node.child_by_field_name("callee")
        if func is None:
            for child in node.children:
                if child.type in {
                    "identifier",
                    "field_expression",
                    "member_expression",
                    "scoped_identifier",
                }:
                    func = child
                    break
        if func is not None:
            call_name = node_text(func)
            if call_name:
                structure["calls"].append(call_name)

    def walk(node: Any) -> list[Any]:
        nodes = [node]
        for child in node.children:
            nodes.extend(walk(child))
        return nodes

    for node in walk(tree.root_node):
        if node.type in import_types:
            add_import(node)
        elif node.type in class_types:
            add_type_node(node, "classes")
        elif node.type in function_types:
            add_type_node(node, "functions")
        elif node.type in call_types:
            add_call(node)

    for line in text.splitlines():
        comment = None
        if "#" in line:
            comment = line.split("#", 1)[1].strip()
        elif "//" in line:
            comment = line.split("//", 1)[1].strip()
        if not comment:
            continue
        lower = comment.lower()
        if any(
            token in lower
            for token in ("because", "rationale", "reason", "note", "todo", "fixme", "why")
        ):
            structure["rationale_comments"].append(comment)

    return structure


def _repo_file_priority(relative_path: Path) -> tuple[int, str]:
    name = relative_path.name.lower()
    suffix = relative_path.suffix.lower()
    stem = relative_path.stem.lower()
    parts = {part.lower() for part in relative_path.parts[:-1]}
    score = 0
    if name in REPO_KEY_FILENAMES or name.startswith("dockerfile"):
        score += 120
    if len(relative_path.parts) == 1:
        score += 30
    if suffix in REPO_CODE_SUFFIXES:
        score += 25
    if parts & {"src", "app", "cmd", "lib", "skills", "services", "server", "client"}:
        score += 10
    if stem in {"main", "app", "cli", "server", "index", "api", "core", "config"}:
        score += 10
    if "test" in relative_path.as_posix().lower():
        score += 5
    return score, relative_path.as_posix()


def _repo_excerpt_text(path: Path, *, max_chars: int) -> str:
    text = _read_text(path).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    line_break = clipped.rfind("\n")
    if line_break >= max_chars // 2:
        clipped = clipped[:line_break].rstrip()
    return clipped + "\n\n[truncated]"


def _get_ast_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _get_ast_call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _extract_python_code_structure(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    structure: dict[str, Any] = {
        "language": "python",
        "imports": [],
        "classes": [],
        "functions": [],
        "calls": [],
        "docstring": "",
        "rationale_comments": [],
    }

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        structure["parse_error"] = str(exc)
        return structure

    structure["docstring"] = ast.get_docstring(tree) or ""

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            structure["functions"].append(
                {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "args": [arg.arg for arg in node.args.args],
                }
            )
        elif isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    methods.append(
                        {
                            "name": child.name,
                            "docstring": ast.get_docstring(child) or "",
                            "args": [arg.arg for arg in child.args.args],
                        }
                    )
            structure["classes"].append(
                {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "methods": methods,
                }
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _get_ast_call_name(node.func)
            if call_name:
                structure["calls"].append(call_name)

    for line in text.splitlines():
        if "#" not in line:
            continue
        comment = line.split("#", 1)[1].strip()
        if not comment:
            continue
        lower = comment.lower()
        if any(
            token in lower
            for token in ("because", "rationale", "reason", "note", "todo", "fixme", "why")
        ):
            structure["rationale_comments"].append(comment)

    return structure


def _extract_code_structure(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _extract_python_code_structure(path)

    tree_sitter_structure = _tree_sitter_code_structure(path)
    if tree_sitter_structure is not None:
        return tree_sitter_structure

    text = _read_text(path)
    structure = {
        "language": _repo_language_hint(path),
        "imports": [],
        "classes": [],
        "functions": [],
        "calls": [],
        "docstring": "",
        "rationale_comments": [],
    }
    if suffix in {".js", ".ts", ".jsx", ".tsx"}:
        for match in re.finditer(r"^\s*(?:import|from)\s+([\w@./-]+)", text, re.M):
            structure["imports"].append(match.group(1))
        for match in re.finditer(r"^\s*(?:function|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M):
            name = match.group(1)
            if text[match.start()] == "c":
                structure["functions"].append({"name": name, "docstring": ""})
            else:
                structure["classes"].append({"name": name, "docstring": "", "methods": []})
    elif suffix in {".java", ".go", ".rs", ".cpp", ".c", ".cs"}:
        for match in re.finditer(
            r"^\s*(?:class|fn|func|def|interface)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M
        ):
            structure["functions"].append({"name": match.group(1), "docstring": ""})
    for line in text.splitlines():
        if "#" in line or "//" in line or "/*" in line:
            comment = line.split("#", 1)[-1].split("//", 1)[-1].strip()
            if not comment:
                continue
            lower = comment.lower()
            if any(
                token in lower
                for token in ("because", "rationale", "reason", "note", "todo", "fixme", "why")
            ):
                structure["rationale_comments"].append(comment)
    return structure


def _detect_graph_communities(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[list[str]]:
    if nx is None:
        return []

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["id"])
    for edge in edges:
        if edge["source"] != edge["target"]:
            graph.add_edge(edge["source"], edge["target"])

    if graph.number_of_nodes() == 0:
        return []

    try:
        communities = nx.community.greedy_modularity_communities(graph)
    except Exception:
        communities = list(nx.connected_components(graph))

    return [sorted(list(c)) for c in communities]


def _build_repo_search_graph(repo_root: Path) -> dict[str, Any]:
    files = _iter_repo_search_files(repo_root)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    file_nodes: dict[str, dict[str, Any]] = {}

    for path in files:
        relative_path = path.relative_to(repo_root).as_posix()
        file_id = f"file:{relative_path}"
        file_node = {
            "id": file_id,
            "label": relative_path,
            "type": "file",
            "language": _repo_language_hint(path),
            "size_bytes": path.stat().st_size,
        }
        nodes.append(file_node)
        file_nodes[relative_path] = file_node

        if path.suffix.lower() in REPO_CODE_SUFFIXES:
            structure = _extract_code_structure(path)
            for imp in sorted(set(structure.get("imports", []))):
                imp_id = f"import:{imp}"
                nodes.append({"id": imp_id, "label": imp, "type": "import"})
                edges.append({"source": file_id, "target": imp_id, "type": "imports"})

            for cls in structure.get("classes", []):
                cls_id = f"class:{relative_path}:{cls['name']}"
                nodes.append(
                    {
                        "id": cls_id,
                        "label": cls["name"],
                        "type": "class",
                        "file": relative_path,
                        "docstring": cls.get("docstring", ""),
                    }
                )
                edges.append({"source": file_id, "target": cls_id, "type": "contains"})
                for method in cls.get("methods", []):
                    method_id = f"method:{relative_path}:{cls['name']}.{method['name']}"
                    nodes.append(
                        {
                            "id": method_id,
                            "label": method["name"],
                            "type": "method",
                            "class": cls["name"],
                            "file": relative_path,
                            "docstring": method.get("docstring", ""),
                        }
                    )
                    edges.append({"source": cls_id, "target": method_id, "type": "contains"})

            for fn in structure.get("functions", []):
                fn_id = f"function:{relative_path}:{fn['name']}"
                nodes.append(
                    {
                        "id": fn_id,
                        "label": fn["name"],
                        "type": "function",
                        "file": relative_path,
                        "docstring": fn.get("docstring", ""),
                    }
                )
                edges.append({"source": file_id, "target": fn_id, "type": "contains"})

            for call in sorted(set(structure.get("calls", []))):
                call_id = f"call:{relative_path}:{call}"
                nodes.append(
                    {
                        "id": call_id,
                        "label": call,
                        "type": "call",
                        "file": relative_path,
                    }
                )
                edges.append({"source": file_id, "target": call_id, "type": "calls"})

            for comment in structure.get("rationale_comments", []):
                comment_id = f"comment:{relative_path}:{len(comment)}"
                nodes.append(
                    {
                        "id": comment_id,
                        "label": comment[:60] + ("..." if len(comment) > 60 else ""),
                        "type": "rationale_comment",
                        "file": relative_path,
                        "text": comment,
                    }
                )
                edges.append({"source": file_id, "target": comment_id, "type": "comments"})

    communities = _detect_graph_communities(nodes, edges)
    graph_data = {
        "repo_root": str(repo_root),
        "node_count": len({node["id"] for node in nodes}),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "communities": [
            {
                "id": index + 1,
                "members": community,
                "size": len(community),
            }
            for index, community in enumerate(communities)
        ],
    }
    return graph_data


def _build_graph_report(graph_data: dict[str, Any]) -> str:
    lines = [
        "# Repository Search Graph Audit Report",
        "",
        f"- Files indexed: {graph_data['node_count']}",
        f"- Relationships captured: {graph_data['edge_count']}",
        f"- Communities detected: {len(graph_data['communities'])}",
        "",
    ]
    for community in graph_data["communities"][:8]:
        lines.append(f"## Community {community['id']} ({community['size']} nodes)")
        lines.extend([f"- {member}" for member in community["members"][:8]])
        if community["size"] > 8:
            lines.append(f"- ... {community['size'] - 8} additional nodes")
        lines.append("")
    if not graph_data["communities"]:
        lines.append("No communities were detected. The graph is small or disconnected.")
    return "\n".join(lines)


def _build_graph_html(graph_data: dict[str, Any]) -> str:
    json_text = json.dumps(graph_data, indent=2)
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Repository Search Graph</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 20px; }}
    textarea {{ width: 100%; height: 300px; font-family: monospace; }}
    input {{ width: 100%; padding: 8px; margin-bottom: 10px; }}
    .result {{ margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>Repository Search Graph</h1>
  <p>Search file, symbol, import, and rationale nodes extracted from the repository snapshot.</p>
  <input id=\"search\" placeholder=\"Search nodes...\" />
  <div id=\"results\"></div>
  <h2>Raw graph JSON</h2>
  <textarea readonly>{html.escape(json_text)}</textarea>
  <script>
    const graph = {json.dumps(graph_data)};
    const nodes = graph.nodes;
    const results = document.getElementById('results');
    const input = document.getElementById('search');
    function render(filter) {{
      const query = filter.trim().toLowerCase();
      const hits = nodes.filter(node => node.label.toLowerCase().includes(query) || (node.type || '').toLowerCase().includes(query));
      results.innerHTML = `<p>${{hits.length}} matching nodes</p>` + hits.slice(0, 50).map(node => `<div class=\"result\"><strong>${{node.label}}</strong> <code>${{node.type}}</code><br/><small>${{node.file || node.id}}</small></div>`).join('');
    }}
    input.addEventListener('input', event => render(event.target.value));
    render('');
  </script>
</body>
</html>"""


def _write_repo_search_artifacts(
    raw_snapshot_dir: Path, graph_data: dict[str, Any]
) -> dict[str, str]:
    json_path = raw_snapshot_dir / "search_graph.json"
    html_path = raw_snapshot_dir / "search_graph.html"
    report_path = raw_snapshot_dir / "search_graph_audit.md"
    _write_text(json_path, json.dumps(graph_data, indent=2, sort_keys=True))
    _write_text(html_path, _build_graph_html(graph_data))
    _write_text(report_path, _build_graph_report(graph_data))
    return {
        "graph_json": _normalize_relative_path(json_path),
        "graph_html": _normalize_relative_path(html_path),
        "graph_audit": _normalize_relative_path(report_path),
    }


def _repo_signature_for_root(repo_root: Path) -> str:
    git_dir = repo_root / ".git"
    if git_dir.exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_root,
                check=False,
            )
            if result.returncode == 0:
                return hashlib.sha256(f"git:{result.stdout.strip()}".encode("utf-8")).hexdigest()
        except FileNotFoundError:
            pass

    digest = hashlib.sha256()
    for path in sorted(_iter_repo_search_files(repo_root), key=lambda path: path.as_posix()):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        stat_result = path.stat()
        digest.update(relative)
        digest.update(str(stat_result.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat_result.st_size).encode("utf-8"))
    return digest.hexdigest()


def _resolve_manifest_repo_root(repo_dir: Path, manifest: dict[str, Any]) -> Path | None:
    candidate = manifest.get("repo_root")
    if isinstance(candidate, str):
        resolved = Path(candidate).expanduser()
        if resolved.exists() and resolved.is_dir():
            return resolved.resolve()

    fallback = repo_dir / "checkout"
    if fallback.exists() and fallback.is_dir():
        return fallback.resolve()

    if repo_dir.exists() and repo_dir.is_dir():
        return repo_dir.resolve()

    return None


def _repo_snapshot_needs_refresh(repo_dir: Path) -> bool:
    manifest_path = repo_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    repo_root_path = _resolve_manifest_repo_root(repo_dir, manifest)
    if repo_root_path is None:
        return False

    stored_signature = manifest.get("repo_signature")
    current_signature = _repo_signature_for_root(repo_root_path)
    return stored_signature != current_signature


def _refresh_ingested_repos_if_changed(
    raw_dir: Path,
    source_dir: Path,
    wiki_path: Path,
    vault_dir: Path,
) -> None:
    repos_dir = raw_dir / "repos"
    if not repos_dir.is_dir():
        return

    for repo_dir in sorted(repos_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        if not _repo_snapshot_needs_refresh(repo_dir):
            continue

        try:
            manifest = json.loads((repo_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        repo_root_path = _resolve_manifest_repo_root(repo_dir, manifest)
        if repo_root_path is None:
            continue

        if str(repo_root_path) != manifest.get("repo_root"):
            manifest["repo_root"] = str(repo_root_path)
            (repo_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )

        repo_name = manifest.get("repo_name")
        if not repo_name or repo_name == "checkout":
            repo_name = None

        wiki_ingest_repo(
            repo_path=str(repo_root_path),
            repo_name=repo_name,
            source_dir=str(source_dir),
            raw_dir=str(raw_dir),
            wiki_path=str(wiki_path),
            vault_dir=str(vault_dir),
            rebuild=False,
            update_related=False,
        )


def _refresh_repo_search_graph(
    repo_dir: Path, repo_root: Path | None = None
) -> dict[str, str] | None:
    manifest_path = repo_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    repo_root_path = repo_root
    if repo_root_path is None:
        repo_root_path = _resolve_manifest_repo_root(repo_dir, manifest)
    if repo_root_path is None:
        return None

    if str(repo_root_path) != manifest.get("repo_root"):
        manifest["repo_root"] = str(repo_root_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    graph_data = _build_repo_search_graph(repo_root_path)
    graph_artifacts = _write_repo_search_artifacts(repo_dir, graph_data)
    _hydrate_repo_snapshot_with_graph_artifacts(repo_dir, graph_artifacts)
    return graph_artifacts


def _hydrate_repo_snapshot_with_graph_artifacts(
    repo_dir: Path, graph_artifacts: dict[str, str]
) -> None:
    manifest_path = repo_dir / "manifest.json"
    snapshot_path = repo_dir / "snapshot.md"
    if not manifest_path.exists():
        return
    try:
        snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot["graph_artifacts"] = graph_artifacts
        manifest_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        _write_text(snapshot_path, _build_repo_snapshot_markdown(snapshot))
    except Exception:
        return


def _build_repo_search_engine_artifacts(raw_dir: Path) -> None:
    repos_dir = raw_dir / "repos"
    if not repos_dir.is_dir():
        return
    for repo_dir in sorted(repos_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        manifest_path = repo_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        graph_json = repo_dir / "search_graph.json"
        if not graph_json.exists() or graph_json.stat().st_mtime < manifest_path.stat().st_mtime:
            # Determine repo root via manifest or snapshot metadata if available
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                repo_root = Path(manifest.get("repo_root", str(repo_dir)))
            except Exception:
                repo_root = repo_dir
            _refresh_repo_search_graph(repo_dir, repo_root)


def _iter_repo_search_files(checkout_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in _iter_files(checkout_dir, None):
        relative_path = path.relative_to(checkout_dir)
        name = relative_path.name.lower()
        suffix = relative_path.suffix.lower()
        if any(part in REPO_IGNORED_DIRS for part in relative_path.parts[:-1]):
            continue
        if (
            suffix in REPO_TEXT_SUFFIXES
            or name in REPO_KEY_FILENAMES
            or name.startswith("dockerfile")
        ):
            files.append(path)
    return files


def _iter_ingested_repo_slugs(raw_dir: Path) -> list[str]:
    repos_dir = raw_dir / "repos"
    if not repos_dir.exists() or not repos_dir.is_dir():
        return []

    slugs: list[str] = []
    for repo_dir in sorted(repos_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        if (repo_dir / "checkout").is_dir() or (repo_dir / "manifest.json").is_file():
            slugs.append(repo_dir.name)
    return slugs


def _wiki_search_repo_fallback(
    pattern: str,
    *,
    literal: bool,
    case_sensitive: bool,
    max_matches: int | None,
    context_lines: int,
    max_hits_per_file: int,
) -> dict[str, Any] | None:
    raw_dir = _infer_raw_dir()
    repo_slugs = _iter_ingested_repo_slugs(raw_dir)
    if not repo_slugs:
        return None

    for repo_slug in repo_slugs:
        result = wiki_search_repo(
            repo_slug,
            pattern,
            raw_dir=str(raw_dir),
            literal=literal,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
            max_hits_per_file=max_hits_per_file,
            context_lines=context_lines,
        )
        if result.get("status") == "success" and int(result.get("count", 0)) > 0:
            result["search_fallback"] = True
            return result
    return None


def _collect_repo_snapshot(
    repo_root: Path,
    *,
    max_files: int,
    max_tree_entries: int,
    max_excerpt_files: int,
    max_excerpt_chars: int,
) -> dict[str, Any]:
    eligible_files: list[Path] = []
    tree_preview: list[str] = []
    extension_counts: Counter[str] = Counter()
    top_level_counts: Counter[str] = Counter()
    total_files = 0

    for current_root, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(
            name for name in dirnames if name not in REPO_IGNORED_DIRS and not name.startswith(".")
        )
        current_dir = Path(current_root)
        for filename in sorted(filenames):
            total_files += 1
            absolute_path = current_dir / filename
            relative_path = absolute_path.relative_to(repo_root)
            if len(tree_preview) < max_tree_entries:
                tree_preview.append(relative_path.as_posix())

            suffix = relative_path.suffix.lower()
            name = relative_path.name.lower()
            if (
                suffix not in REPO_TEXT_SUFFIXES
                and name not in REPO_KEY_FILENAMES
                and not name.startswith("dockerfile")
            ):
                continue

            eligible_files.append(absolute_path)
            extension_counts[suffix or "<none>"] += 1
            top_level = relative_path.parts[0] if len(relative_path.parts) > 1 else "<root>"
            top_level_counts[top_level] += 1

    ranked_files = sorted(
        eligible_files,
        key=lambda path: (-_repo_file_priority(path.relative_to(repo_root))[0], path.as_posix()),
    )
    excerpt_paths = ranked_files[:max_excerpt_files]
    excerpt_entries = []
    for path in excerpt_paths:
        relative_path = path.relative_to(repo_root)
        excerpt_entries.append(
            {
                "relative_path": relative_path.as_posix(),
                "size_bytes": path.stat().st_size,
                "language": _repo_language_hint(relative_path),
                "excerpt": _repo_excerpt_text(path, max_chars=max_excerpt_chars),
            }
        )

    catalog_entries = []
    for path in sorted(eligible_files, key=lambda item: item.relative_to(repo_root).as_posix())[
        :max_files
    ]:
        relative_path = path.relative_to(repo_root)
        catalog_entries.append(
            {
                "relative_path": relative_path.as_posix(),
                "size_bytes": path.stat().st_size,
                "language": _repo_language_hint(relative_path),
            }
        )

    key_files = [
        path.relative_to(repo_root).as_posix()
        for path in ranked_files
        if path.relative_to(repo_root).name.lower() in REPO_KEY_FILENAMES
        or path.relative_to(repo_root).name.lower().startswith("dockerfile")
    ][:8]
    top_level_areas = [name for name, _ in top_level_counts.most_common(8)]
    summary = (
        f"Repository snapshot for `{repo_root.name}` with {len(eligible_files)} text/code files "
        f"across {len(top_level_counts)} top-level areas."
    )
    test_file_count = sum(
        1 for path in eligible_files if "test" in path.relative_to(repo_root).as_posix().lower()
    )

    return {
        "generated_at": _now_iso(),
        "repo_root": str(repo_root),
        "repo_name": repo_root.name,
        "repo_signature": _repo_signature_for_root(repo_root),
        "summary": summary,
        "total_file_count": total_files,
        "included_file_count": len(eligible_files),
        "catalog_truncated": len(eligible_files) > max_files,
        "tree_truncated": total_files > max_tree_entries,
        "top_level_areas": top_level_areas,
        "top_level_counts": dict(top_level_counts.most_common(12)),
        "extension_counts": dict(extension_counts.most_common(12)),
        "key_files": key_files,
        "test_file_count": test_file_count,
        "tree_preview": tree_preview,
        "catalog_entries": catalog_entries,
        "excerpt_entries": excerpt_entries,
    }


def _build_repo_snapshot_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        f"# {snapshot['repo_name']} Repository Snapshot",
        "",
        f"> Generated from local repository `{snapshot['repo_root']}` for wiki ingest.",
        "",
        "## Summary",
        "",
        snapshot["summary"],
        "",
        "## Snapshot Metadata",
        "",
        f"- Captured at: `{snapshot['generated_at']}`",
        f"- Repository path: `{snapshot['repo_root']}`",
        f"- Files scanned: `{snapshot['total_file_count']}`",
        f"- Text/code files included: `{snapshot['included_file_count']}`",
        f"- Test-like files detected: `{snapshot['test_file_count']}`",
        "",
        "## Top-Level Areas",
        "",
    ]
    if snapshot["top_level_counts"]:
        for area, count in snapshot["top_level_counts"].items():
            lines.append(f"- `{area}`: {count} files")
    else:
        lines.append("- No eligible files detected.")

    lines.extend(["", "## Dominant File Types", ""])
    if snapshot["extension_counts"]:
        for suffix, count in snapshot["extension_counts"].items():
            lines.append(f"- `{suffix}`: {count} files")
    else:
        lines.append("- No file types recorded.")

    lines.extend(["", "## Key Files", ""])
    if snapshot["key_files"]:
        for relative_path in snapshot["key_files"]:
            lines.append(f"- `{relative_path}`")
    else:
        lines.append("- No conventional entrypoint or config files detected.")

    lines.extend(["", "## Representative Excerpts", ""])
    if snapshot["excerpt_entries"]:
        for entry in snapshot["excerpt_entries"]:
            lines.extend(
                [
                    f"### {entry['relative_path']}",
                    "",
                    f"- Size: `{entry['size_bytes']}` bytes",
                    "",
                    f"```{entry['language']}",
                    entry["excerpt"],
                    "```",
                    "",
                ]
            )
    else:
        lines.append("- No representative excerpts available.")
        lines.append("")

    lines.extend(["## File Tree Preview", "", "```text"])
    lines.extend(snapshot["tree_preview"] or ["<empty>"])
    if snapshot["tree_truncated"]:
        lines.append("...")
    lines.extend(["```", "", "## Included File Catalog", ""])
    if snapshot["catalog_entries"]:
        for entry in snapshot["catalog_entries"]:
            lines.append(
                f"- `{entry['relative_path']}` ({entry['language']}, {entry['size_bytes']} bytes)"
            )
        if snapshot["catalog_truncated"]:
            lines.append("- `...` additional files omitted from the catalog preview")
    else:
        lines.append("- No eligible files added to the catalog.")

    if snapshot.get("graph_artifacts"):
        lines.extend(["", "## Search Graph Artifacts", ""])
        graph = snapshot["graph_artifacts"]
        lines.append(f"- Search graph JSON: `{graph['graph_json']}`")
        lines.append(f"- Interactive graph viewer: `{graph['graph_html']}`")
        lines.append(f"- Audit report: `{graph['graph_audit']}`")
    return "\n".join(lines)


def _build_repo_structure_section(
    repo_root: Path, snapshot: dict[str, Any], max_files: int = 8
) -> list[str]:
    lines: list[str] = ["## Repository Structure", ""]
    if not repo_root.exists() or not repo_root.is_dir():
        return lines + ["- Repository root is not available for structure extraction."]

    candidate_paths: list[str] = []
    candidate_paths.extend(snapshot.get("key_files", []))
    candidate_paths.extend(entry["relative_path"] for entry in snapshot.get("excerpt_entries", []))
    candidate_paths.extend(
        entry["relative_path"] for entry in snapshot.get("catalog_entries", [])[:max_files]
    )
    candidate_paths = [path for path in dict.fromkeys(candidate_paths) if path]
    candidate_paths = candidate_paths[:max_files]

    if not candidate_paths:
        return lines + ["- No candidate files could be selected for structure extraction."]

    lines.append("### Candidate files for AST inspection")
    lines.append("")
    for relative_path in candidate_paths:
        path = repo_root / relative_path
        if not path.exists() or not path.is_file():
            lines.append(f"- `{relative_path}`: missing file")
            continue

        structure = _extract_code_structure(path)
        imports = sorted(set(structure.get("imports", [])))
        classes = structure.get("classes", [])
        functions = structure.get("functions", [])
        rationale = structure.get("rationale_comments", [])
        parse_error = structure.get("parse_error")

        lines.append(f"### `{relative_path}`")
        if parse_error:
            lines.append(f"- Parse error: `{parse_error}`")
        lines.append(f"- Language: `{structure.get('language', 'unknown')}`")
        if imports:
            lines.append(f"- Imports: {', '.join(f'`{item}`' for item in imports[:8])}")
        if classes:
            class_names = [cls["name"] for cls in classes[:4]]
            lines.append(
                "- Classes: {} ({})".format(
                    len(classes),
                    ", ".join(f"`{name}`" for name in class_names),
                )
            )
        if functions:
            function_names = [fn["name"] for fn in functions[:6]]
            lines.append(
                "- Functions: {} ({})".format(
                    len(functions),
                    ", ".join(f"`{name}`" for name in function_names),
                )
            )
        if rationale:
            lines.append(f"- Rationale comments: {len(rationale)}")
            for comment in rationale[:3]:
                lines.append(f"  - `{comment}`")
        lines.append("")

    return lines


def _build_source_note_from_repo(
    *,
    repo_name: str,
    repo_root: Path,
    raw_snapshot_relative_path: str,
    raw_manifest_relative_path: str,
    preview_text: str,
    snapshot: dict[str, Any],
    related_documents: list[WikiDocument] | None = None,
    suggested_pages: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"# {repo_name} Repository",
        "",
        f"> Derived from local repository `{repo_root}` via raw snapshot `{raw_snapshot_relative_path}`.",
        "",
        "## Summary",
        "",
        snapshot["summary"],
        "",
        "## Repository Metadata",
        "",
        f"- Repository path: `{repo_root}`",
        f"- Raw snapshot: `{raw_snapshot_relative_path}`",
        f"- Raw manifest: `{raw_manifest_relative_path}`",
        f"- Files scanned: `{snapshot['total_file_count']}`",
        f"- Text/code files included: `{snapshot['included_file_count']}`",
        f"- Test-like files detected: `{snapshot['test_file_count']}`",
    ]
    if snapshot.get("graph_artifacts"):
        graph = snapshot["graph_artifacts"]
        lines.extend(
            [
                "",
                "## Search Graph Artifacts",
                "",
                f"- Search graph JSON: `{graph['graph_json']}`",
                f"- Interactive graph viewer: `{graph['graph_html']}`",
                f"- Audit report: `{graph['graph_audit']}`",
            ]
        )
    if snapshot["top_level_areas"]:
        lines.append(
            f"- Top-level areas: {', '.join(f'`{area}`' for area in snapshot['top_level_areas'][:6])}"
        )
    lines.extend(_build_repo_structure_section(repo_root, snapshot, max_files=8))
    lines.extend(
        [
            "",
            "## Ingest Focus",
            "",
            "- Capture the architecture, entrypoints, and core module boundaries.",
            "- Note how the repository is operated, tested, and configured.",
            "- Link this repo note to existing workflows, concepts, or decision pages.",
            "- Prefer follow-up pages for durable topics rather than stuffing everything into one note.",
        ]
    )
    if related_documents:
        lines.extend(["", "## Related Existing Notes", ""])
        for document in related_documents:
            lines.append(f"- {_source_note_wikilink(document.relative_path, document.title)}")
    if suggested_pages:
        lines.extend(["", "## Suggested Follow-Up Pages", ""])
        for suggestion in suggested_pages:
            lines.append(
                f"- `{suggestion['action']}` `{suggestion['relative_path']}`: {suggestion['reason']}"
            )
    if preview_text:
        lines.extend(["", "## Snapshot Excerpt", "", "```text", preview_text, "```"])
    return "\n".join(lines)


def _suggest_repo_follow_up_pages(
    *,
    source_dir: Path,
    repo_name: str,
    test_file_count: int,
) -> list[dict[str, Any]]:
    repo_slug = _slugify(repo_name)
    existing_paths = {document.relative_path for document in _scan_documents(source_dir)}
    candidates = [
        (
            f"concepts/{repo_slug}-architecture.md",
            "Capture the repository's major modules, boundaries, and how the pieces fit together.",
        ),
        (
            f"concepts/{repo_slug}-entrypoints.md",
            "Document the main commands, services, scripts, or runtime entrypoints exposed by this codebase.",
        ),
    ]
    if test_file_count:
        candidates.append(
            (
                f"concepts/{repo_slug}-testing.md",
                "Summarize the test strategy, important fixtures, and what parts of the codebase are exercised.",
            )
        )

    suggestions = []
    for relative_path, reason in candidates:
        if relative_path in existing_paths:
            continue
        suggestions.append(
            {
                "action": "create",
                "relative_path": relative_path,
                "title": _titleize(Path(relative_path).stem),
                "reason": reason,
            }
        )
    return suggestions


def _build_source_note_from_raw(
    *,
    raw_path: Path,
    raw_relative_path: str,
    preview_text: str,
    title: str | None = None,
    related_documents: list[WikiDocument] | None = None,
    suggested_pages: list[dict[str, Any]] | None = None,
) -> str:
    resolved_title = title or _extract_title(raw_path, preview_text or raw_path.stem)
    summary = (
        _summarize(preview_text)
        if preview_text
        else "Summarize this source and integrate its durable claims."
    )
    lines = [
        f"# {resolved_title}",
        "",
        f"> Derived from raw artifact `{raw_relative_path}`. Expand this note into a maintained synthesis page.",
        "",
        "## Summary",
        "",
        summary,
        "",
        "## Source Metadata",
        "",
        f"- Raw artifact: `{raw_relative_path}`",
        f"- Imported at: `{_now_iso()}`",
        f"- Source kind: `{raw_path.suffix.lower() or 'unknown'}`",
        "",
        "## Integration Tasks",
        "",
        "- Extract the durable facts and claims from the source.",
        "- Link this note to related wiki pages.",
        "- Record contradictions, uncertainty, or follow-up questions explicitly.",
        "- Rewrite this scaffold into a proper maintained note after review.",
    ]
    if related_documents:
        lines.extend(["", "## Related Existing Notes", ""])
        for document in related_documents:
            lines.append(f"- {_source_note_wikilink(document.relative_path, document.title)}")
    if suggested_pages:
        lines.extend(["", "## Suggested Follow-Up Pages", ""])
        for suggestion in suggested_pages:
            lines.append(
                f"- `{suggestion['action']}` `{suggestion['relative_path']}`: {suggestion['reason']}"
            )
    if preview_text:
        lines.extend(
            [
                "",
                "## Source Excerpt",
                "",
                "```text",
                preview_text,
                "```",
            ]
        )
    return "\n".join(lines)


def _is_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme.lower() in {"http", "https"}


def _paper_reference_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    arxiv_id = _extract_arxiv_id(text)
    if arxiv_id and not _is_url(text):
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    parsed = urlparse(text)
    if parsed.netloc.lower().endswith("arxiv.org") and parsed.path.startswith("/abs/"):
        arxiv_id = parsed.path.removeprefix("/abs/").strip("/")
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return text


def _filename_from_paper_reference(reference: str) -> str:
    parsed = urlparse(reference)
    if parsed.scheme:
        name = Path(unquote(parsed.path)).name
    else:
        name = Path(reference).name
    return name or "paper.pdf"


def _paper_base_name(
    *,
    reference: str,
    target_name: str | None,
    title: str | None,
) -> str:
    if target_name:
        name = Path(target_name).name
        return _slugify(Path(name).stem)
    if title:
        return _slugify(title)
    reference_name = _filename_from_paper_reference(reference)
    return _slugify(Path(reference_name).stem or "paper")


def _copy_or_download_paper_pdf(
    reference: str,
    *,
    target_path: Path,
    timeout_sec: float,
    max_bytes: int,
) -> dict[str, Any]:
    resolved_reference = _paper_reference_url(reference)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_url(resolved_reference):
        request = Request(
            resolved_reference,
            headers={
                "User-Agent": "LogicianWiki/0.1 (+https://local)",
                "Accept": "application/pdf,*/*;q=0.8",
            },
        )
        total = 0
        with urlopen(request, timeout=max(1.0, float(timeout_sec))) as response:
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(
                    f"PDF is too large ({content_length} bytes > limit {max_bytes} bytes)"
                )
            with target_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 128)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            f"PDF is too large (> limit {max_bytes} bytes); partial file kept at {target_path}"
                        )
                    handle.write(chunk)
        return {
            "mode": "download",
            "source": resolved_reference,
            "bytes": total,
        }

    source_path = Path(resolved_reference).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Paper PDF not found: {source_path}")
    size = source_path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"PDF is too large ({size} bytes > limit {max_bytes} bytes)")
    shutil.copyfile(source_path, target_path)
    return {
        "mode": "copy",
        "source": str(source_path),
        "bytes": size,
    }


def _clean_pdf_metadata_value(value: Any) -> str:
    text = str(value or "").replace("\x00", "").strip()
    return re.sub(r"\s+", " ", text)


def _extract_arxiv_id(text: str) -> str | None:
    candidate = str(text or "")
    match = re.search(
        r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)?(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)",
        candidate,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("id")
    old_style = re.search(
        r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)?(?P<id>[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
        candidate,
        flags=re.IGNORECASE,
    )
    return old_style.group("id") if old_style else None


def _extract_doi(text: str) -> str | None:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", str(text or ""), re.IGNORECASE)
    if not match:
        return None
    return match.group(0).rstrip(".,);")


def _extract_abstract(text: str, *, max_chars: int = 1200) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return ""
    match = re.search(
        r"\babstract\b[:\s-]*(?P<abstract>.*?)(?:\b(?:keywords|index terms|1\s+introduction|introduction)\b)",
        clean,
        flags=re.IGNORECASE,
    )
    abstract = match.group("abstract").strip() if match else ""
    if not abstract:
        # Keep a compact first-page preview when no explicit abstract heading
        # survives PDF extraction.
        abstract = clean[:max_chars].strip()
    if len(abstract) > max_chars:
        abstract = abstract[:max_chars].rstrip() + "..."
    return abstract


def _extract_pdf_text_and_metadata(
    pdf_path: Path,
    *,
    max_chars: int,
    max_pages: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    text_parts: list[str] = []
    extraction_error = ""
    page_count = 0
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        raw_metadata = reader.metadata or {}
        for key, value in dict(raw_metadata).items():
            cleaned = _clean_pdf_metadata_value(value)
            if cleaned:
                metadata[str(key).lstrip("/").lower()] = cleaned
        remaining = max(0, int(max_chars))
        for page in reader.pages[: max(1, int(max_pages))]:
            if remaining <= 0:
                break
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            page_text = page_text.strip()
            text_parts.append(page_text[:remaining])
            remaining -= len(page_text)
    except Exception as exc:
        extraction_error = str(exc) or exc.__class__.__name__

    text = "\n\n".join(part.strip() for part in text_parts if part.strip()).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return {
        "metadata": metadata,
        "text": text,
        "page_count": page_count,
        "extraction_error": extraction_error,
    }


def _paper_title_from_metadata(
    *,
    explicit_title: str | None,
    metadata: dict[str, Any],
    pdf_path: Path,
) -> str:
    for candidate in (
        explicit_title,
        metadata.get("title"),
        metadata.get("dc:title"),
        pdf_path.stem.replace("-", " ").replace("_", " "),
    ):
        text = _clean_pdf_metadata_value(candidate)
        if text:
            return text
    return "Research Paper"


def _build_source_note_from_paper(
    *,
    title: str,
    paper: dict[str, Any],
    extracted_text: str,
    related_documents: list[WikiDocument] | None = None,
    suggested_pages: list[dict[str, Any]] | None = None,
) -> str:
    abstract = _extract_abstract(extracted_text)
    metadata = dict(paper.get("metadata") or {})
    authors = metadata.get("author") or metadata.get("authors") or metadata.get("dc:creator") or ""
    keywords = metadata.get("keywords") or metadata.get("subject") or ""
    lines = [
        f"# {title}",
        "",
        f"> Research paper imported from `{paper['input_reference']}` into raw artifact `{paper['raw_pdf_relative_path']}`.",
        "",
        "## Summary",
        "",
        abstract
        or "Summarize the paper's main question, method, results, and limitations after review.",
        "",
        "## Paper Metadata",
        "",
        f"- PDF: `{paper['raw_pdf_relative_path']}`",
        f"- Extracted text: `{paper['raw_text_relative_path']}`",
        f"- Metadata: `{paper['raw_metadata_relative_path']}`",
        f"- Imported at: `{paper['imported_at']}`",
        f"- SHA256: `{paper['sha256']}`",
        f"- Pages detected: `{paper.get('page_count', 0)}`",
    ]
    if paper.get("source_url"):
        lines.append(f"- Source URL: {paper['source_url']}")
    if paper.get("source_path"):
        lines.append(f"- Source path: `{paper['source_path']}`")
    if authors:
        lines.append(f"- Authors: {authors}")
    if paper.get("arxiv_id"):
        lines.append(f"- arXiv: `{paper['arxiv_id']}`")
    if paper.get("doi"):
        lines.append(f"- DOI: `{paper['doi']}`")
    if keywords:
        lines.append(f"- Keywords/subject: {keywords}")
    if paper.get("extraction_error"):
        lines.append(f"- Extraction note: `{paper['extraction_error']}`")

    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- Problem: _fill in the research question or gap._",
            "- Method: _summarize the core approach and assumptions._",
            "- Evidence: _capture datasets, experiments, proofs, or qualitative support._",
            "- Results: _record the strongest reported findings with page/section anchors when possible._",
            "- Limitations: _note caveats, threats to validity, missing comparisons, and uncertainty._",
            "- Follow-up: _link related papers, repos, datasets, or wiki notes._",
        ]
    )
    if related_documents:
        lines.extend(["", "## Related Existing Notes", ""])
        for document in related_documents:
            lines.append(f"- {_source_note_wikilink(document.relative_path, document.title)}")
    if suggested_pages:
        lines.extend(["", "## Suggested Follow-Up Pages", ""])
        for suggestion in suggested_pages:
            lines.append(
                f"- `{suggestion['action']}` `{suggestion['relative_path']}`: {suggestion['reason']}"
            )
    if extracted_text:
        excerpt = extracted_text[:4000].rstrip()
        if len(extracted_text) > len(excerpt):
            excerpt += "\n\n[truncated]"
        lines.extend(["", "## Extracted Text Excerpt", "", "```text", excerpt, "```"])
    return "\n".join(lines)


def _related_documents_for_raw(
    *,
    source_dir: Path,
    raw_relative_path: str,
    title: str,
    preview_text: str,
    exclude_relative_paths: set[str] | None = None,
    limit: int = 3,
) -> list[WikiDocument]:
    exclude_paths = exclude_relative_paths or set()
    raw_parts = {part.lower() for part in Path(raw_relative_path).parts}
    raw_keywords = set(
        _extract_keywords(title, _extract_markdown_headings(preview_text), preview_text)
    )
    candidates: list[tuple[int, WikiDocument]] = []
    for document in _scan_documents(source_dir):
        if document.relative_path in exclude_paths:
            continue
        score = 0
        shared_keywords = raw_keywords & set(document.keywords)
        score += len(shared_keywords) * 2
        if document.top_level.lower() in raw_parts:
            score += 2
        if document.title.lower() in preview_text.lower():
            score += 3
        if score <= 0:
            continue
        candidates.append((score, document))
    candidates.sort(key=lambda item: (-item[0], item[1].relative_path))
    return [document for _, document in candidates[:limit]]


def _append_source_update_reference(
    *,
    source_dir: Path,
    target_relative_path: str,
    new_note_relative_path: str,
    new_note_title: str,
    raw_relative_path: str,
) -> bool:
    target = _ensure_safe_target(source_dir, target_relative_path)
    if not target.exists():
        return False
    existing = _read_text(target)
    link = _source_note_wikilink(new_note_relative_path, new_note_title)
    if link in existing:
        return False
    block_lines = [
        "",
        "## Source Updates",
        "",
        f"- Related source note added: {link} from raw artifact `{raw_relative_path}`.",
    ]
    updated = existing.rstrip() + "\n" + "\n".join(block_lines) + "\n"
    _write_text(target, updated)
    return True


_SUGGESTED_PAGE_RE = re.compile(
    r"^- `(?P<action>create|update)` `(?P<path>[^`]+)`: (?P<reason>.+)$"
)


def _extract_section_lines(content: str, heading: str) -> list[str]:
    lines = content.splitlines()
    collected: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == heading:
            in_section = True
            continue
        if in_section and re.match(r"^##\s+", stripped):
            break
        if in_section:
            collected.append(line)
    return collected


def _parse_suggested_pages_from_note(content: str) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []
    for line in _extract_section_lines(content, "## Suggested Follow-Up Pages"):
        match = _SUGGESTED_PAGE_RE.match(line.strip())
        if not match:
            continue
        suggestions.append(
            {
                "action": match.group("action"),
                "relative_path": match.group("path"),
                "reason": match.group("reason").strip(),
            }
        )
    return suggestions


def _build_follow_up_note_from_suggestion(
    *,
    suggestion_relative_path: str,
    reason: str,
    source_note_relative_path: str,
    source_note_title: str,
) -> str:
    title = _titleize(Path(suggestion_relative_path).stem)
    source_link = _source_note_wikilink(source_note_relative_path, source_note_title)
    lines = [
        f"# {title}",
        "",
        f"> Suggested follow-up page promoted from {source_link}.",
        "",
        "## Why This Page Exists",
        "",
        reason,
        "",
        "## Source Note",
        "",
        f"- Originating source note: {source_link}",
        "",
        "## Next Steps",
        "",
        "- Summarize the durable claim or topic this page should cover.",
        "- Add links to related wiki notes.",
        "- Record uncertainty, comparisons, or contradictions explicitly.",
    ]
    return "\n".join(lines)


def _append_suggested_follow_up_update(
    *,
    source_dir: Path,
    target_relative_path: str,
    source_note_relative_path: str,
    source_note_title: str,
    reason: str,
) -> bool:
    target = _ensure_safe_target(source_dir, target_relative_path)
    if not target.exists():
        return False
    existing = _read_text(target)
    source_link = _source_note_wikilink(source_note_relative_path, source_note_title)
    if source_link in existing and reason in existing:
        return False
    block_lines = [
        "",
        "## Suggested Follow-Up Work",
        "",
        f"- Review {source_link}: {reason}",
    ]
    updated = existing.rstrip() + "\n" + "\n".join(block_lines) + "\n"
    _write_text(target, updated)
    return True


def _focus_slug_for_raw(title: str, preview_text: str) -> str | None:
    keywords = [
        token
        for token in _extract_keywords(
            title, _extract_markdown_headings(preview_text), preview_text
        )
        if token not in _GENERIC_SUGGESTION_TOKENS
    ]
    if len(keywords) >= 2:
        return _slugify("-".join(keywords[:2]))
    if keywords:
        return _slugify(keywords[0])
    title_slug = _slugify(title)
    if title_slug and title_slug not in _GENERIC_SUGGESTION_TOKENS:
        return title_slug
    return None


def _suggest_follow_up_pages(
    *,
    source_dir: Path,
    related_documents: list[WikiDocument],
    title: str,
    preview_text: str,
    source_note_path: str,
    max_suggested_pages: int = 5,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []

    for document in related_documents:
        suggestions.append(
            {
                "action": "update",
                "relative_path": document.relative_path,
                "title": document.title,
                "reason": (
                    "This existing note appears related to the new source and may need "
                    "fresh evidence, links, or contradiction checks."
                ),
            }
        )

    focus_slug = _focus_slug_for_raw(title, preview_text)
    if focus_slug:
        all_documents = _scan_documents(source_dir)
        existing_slugs = {
            _slugify(Path(document.relative_path).stem) for document in all_documents
        } | {_slugify(document.title) for document in all_documents}
        if focus_slug not in existing_slugs:
            suggestions.append(
                {
                    "action": "create",
                    "relative_path": f"concepts/{focus_slug}.md",
                    "title": _titleize(focus_slug),
                    "reason": (
                        "This source appears to introduce or strengthen a durable topic "
                        "that is not yet represented as its own wiki note."
                    ),
                }
            )

    return suggestions[:max_suggested_pages]


def wiki_add_document(
    relative_path: str,
    content: str,
    source_dir: str | None = None,
    wiki_path: str | None = None,
    raw_dir: str | None = None,
    vault_dir: str | None = None,
    rebuild: bool = True,
) -> dict[str, Any]:
    """Create a new markdown source note and optionally rebuild the wiki."""
    resolved_source_dir = _infer_source_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    target = _ensure_safe_target(resolved_source_dir, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    result = _ok(
        source_path=str(target),
        relative_path=_normalize_relative_path(target.relative_to(resolved_source_dir)),
    )
    result["log_path"] = str(
        _append_log_entry(
            _infer_vault_dir(
                source_dir=resolved_source_dir,
                wiki_path=wiki_path,
                raw_dir=raw_dir,
                vault_dir=vault_dir,
            ),
            "ingest",
            target.name,
            [
                f"source_note={_normalize_relative_path(target.relative_to(resolved_source_dir))}",
                f"rebuild={rebuild}",
            ],
        )
    )
    if rebuild:
        result["build"] = wiki_build(
            source_dir=str(resolved_source_dir),
            wiki_path=wiki_path,
            raw_dir=raw_dir,
            vault_dir=vault_dir,
        )
    return result


def wiki_add_raw_document(
    relative_path: str,
    content: str,
    raw_dir: str | None = None,
) -> dict[str, Any]:
    """Create a raw artifact as text inside the raw directory."""
    resolved_raw_dir = _resolve_raw_dir(raw_dir)
    target = _ensure_safe_target(resolved_raw_dir, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return _ok(
        raw_path=str(target),
        relative_path=_normalize_relative_path(target.relative_to(resolved_raw_dir)),
        log_path=str(
            _append_log_entry(
                _infer_vault_dir(raw_dir=resolved_raw_dir),
                "ingest",
                target.name,
                [f"raw_artifact={_normalize_relative_path(target.relative_to(resolved_raw_dir))}"],
            )
        ),
    )


def wiki_add_file(
    path: str,
    output_dir: str | None = None,
    source_label: str | None = None,
    chunk_size: int | None = None,
    overlap: float | None = None,
    source_dir: str | None = None,
    rebuild: bool = True,
    raw_dir: str | None = None,
    vault_dir: str | None = None,
) -> dict[str, Any]:
    """Copy a file into source and optionally rebuild the wiki."""
    del chunk_size, overlap
    resolved_source_dir = _infer_source_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=output_dir,
        vault_dir=vault_dir,
    )
    resolved_source_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        return _error("Source file not found", source_path=str(source_path))

    destination_name = source_label or source_path.name
    target = _ensure_safe_target(resolved_source_dir, destination_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target)

    result = _ok(
        source_path=str(source_path),
        target_path=str(target),
        relative_path=_normalize_relative_path(target.relative_to(resolved_source_dir)),
    )
    result["log_path"] = str(
        _append_log_entry(
            _infer_vault_dir(
                source_dir=resolved_source_dir,
                wiki_path=output_dir,
                raw_dir=raw_dir,
                vault_dir=vault_dir,
            ),
            "ingest",
            target.name,
            [
                f"source_file={source_path}",
                f"target_note={_normalize_relative_path(target.relative_to(resolved_source_dir))}",
                f"rebuild={rebuild}",
            ],
        )
    )
    if rebuild:
        result["build"] = wiki_build(
            source_dir=str(resolved_source_dir),
            wiki_path=output_dir,
            raw_dir=raw_dir,
            vault_dir=vault_dir,
        )
    return result


def wiki_add_raw_file(
    path: str,
    *,
    raw_dir: str | None = None,
    target_name: str | None = None,
) -> dict[str, Any]:
    """Copy any artifact into the raw directory."""
    resolved_raw_dir = _resolve_raw_dir(raw_dir)
    resolved_raw_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        return _error("Raw source file not found", source_path=str(source_path))

    destination_name = target_name or source_path.name
    target = _ensure_safe_target(resolved_raw_dir, destination_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target)

    return _ok(
        source_path=str(source_path),
        raw_path=str(target),
        relative_path=_normalize_relative_path(target.relative_to(resolved_raw_dir)),
        log_path=str(
            _append_log_entry(
                _infer_vault_dir(raw_dir=resolved_raw_dir),
                "ingest",
                target.name,
                [
                    f"raw_source={source_path}",
                    f"raw_target={_normalize_relative_path(target.relative_to(resolved_raw_dir))}",
                ],
            )
        ),
    )


def wiki_ingest_raw(
    raw_relative_path: str,
    *,
    source_note_path: str | None = None,
    title: str | None = None,
    source_dir: str | None = None,
    raw_dir: str | None = None,
    wiki_path: str | None = None,
    vault_dir: str | None = None,
    rebuild: bool = True,
    max_chars: int = 6000,
    update_related: bool = True,
    max_related_notes: int = 3,
    max_suggested_pages: int = 5,
) -> dict[str, Any]:
    """Promote a raw artifact into a maintained wiki source note scaffold and optionally rebuild."""
    resolved_raw_dir = _infer_raw_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_source_dir = _infer_source_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    raw_path = _ensure_safe_target(resolved_raw_dir, raw_relative_path)
    if not raw_path.exists() or not raw_path.is_file():
        return _error("Raw artifact not found", raw_path=str(raw_path))

    target_relative_path = source_note_path or _default_source_note_path_for_raw(raw_relative_path)
    preview_text = _raw_artifact_text_preview(raw_path, max_chars=max_chars)
    resolved_title = title or _extract_title(raw_path, preview_text or raw_path.stem)
    related_documents = _related_documents_for_raw(
        source_dir=resolved_source_dir,
        raw_relative_path=_normalize_relative_path(raw_path.relative_to(resolved_raw_dir)),
        title=resolved_title,
        preview_text=preview_text,
        exclude_relative_paths={target_relative_path},
        limit=max_related_notes,
    )
    suggested_pages = _suggest_follow_up_pages(
        source_dir=resolved_source_dir,
        related_documents=related_documents,
        title=resolved_title,
        preview_text=preview_text,
        source_note_path=target_relative_path,
        max_suggested_pages=max_suggested_pages,
    )
    source_note = _build_source_note_from_raw(
        raw_path=raw_path,
        raw_relative_path=_normalize_relative_path(raw_path.relative_to(resolved_raw_dir)),
        preview_text=preview_text,
        title=resolved_title,
        related_documents=related_documents,
        suggested_pages=suggested_pages,
    )
    result = wiki_add_document(
        relative_path=target_relative_path,
        content=source_note,
        source_dir=str(resolved_source_dir),
        wiki_path=wiki_path,
        raw_dir=str(resolved_raw_dir),
        vault_dir=vault_dir,
        rebuild=False,
    )
    if result.get("status") == "success":
        updated_related_paths = []
        if update_related:
            for document in related_documents:
                if _append_source_update_reference(
                    source_dir=resolved_source_dir,
                    target_relative_path=document.relative_path,
                    new_note_relative_path=target_relative_path,
                    new_note_title=resolved_title,
                    raw_relative_path=_normalize_relative_path(
                        raw_path.relative_to(resolved_raw_dir)
                    ),
                ):
                    updated_related_paths.append(document.relative_path)
        if rebuild:
            result["build"] = wiki_build(
                source_dir=str(resolved_source_dir),
                wiki_path=wiki_path,
                raw_dir=str(resolved_raw_dir),
                vault_dir=vault_dir,
            )
        result["raw_path"] = str(raw_path)
        result["source_note_path"] = target_relative_path
        result["related_documents"] = [
            {"relative_path": document.relative_path, "title": document.title}
            for document in related_documents
        ]
        result["suggested_pages"] = suggested_pages
        result["updated_related_paths"] = updated_related_paths
        result["log_path"] = str(
            _append_log_entry(
                _infer_vault_dir(
                    source_dir=resolved_source_dir,
                    raw_dir=resolved_raw_dir,
                    wiki_path=wiki_path,
                    vault_dir=vault_dir,
                ),
                "ingest",
                raw_path.name,
                [
                    f"raw_artifact={_normalize_relative_path(raw_path.relative_to(resolved_raw_dir))}",
                    f"source_note={target_relative_path}",
                    f"related_notes={len(related_documents)}",
                    f"suggested_pages={len(suggested_pages)}",
                    f"updated_related={len(updated_related_paths)}",
                    f"rebuild={rebuild}",
                ],
            )
        )
    return result


def wiki_add_raw_paper(
    paper: str,
    *,
    paper_title: str | None = None,
    source_note_path: str | None = None,
    target_name: str | None = None,
    source_dir: str | None = None,
    raw_dir: str | None = None,
    wiki_path: str | None = None,
    vault_dir: str | None = None,
    rebuild: bool = True,
    max_chars: int = 12000,
    max_pages: int = 12,
    timeout_sec: float = 30.0,
    max_bytes: int = 100 * 1024 * 1024,
    update_related: bool = True,
    max_related_notes: int = 3,
    max_suggested_pages: int = 6,
) -> dict[str, Any]:
    """Download/copy a paper PDF into raw/papers/ and create a maintained paper note."""
    resolved_raw_dir = _infer_raw_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_source_dir = _infer_source_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_raw_dir.mkdir(parents=True, exist_ok=True)
    resolved_source_dir.mkdir(parents=True, exist_ok=True)

    normalized_reference = _paper_reference_url(paper)
    initial_base = _paper_base_name(
        reference=normalized_reference,
        target_name=target_name,
        title=paper_title,
    )
    raw_papers_dir = _ensure_safe_target(resolved_raw_dir, "papers")
    raw_papers_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_papers_dir / f"{initial_base}.pdf"

    try:
        materialization = _copy_or_download_paper_pdf(
            normalized_reference,
            target_path=pdf_path,
            timeout_sec=timeout_sec,
            max_bytes=max_bytes,
        )
    except Exception as exc:
        return _error(
            "Failed to materialize paper PDF",
            input_reference=paper,
            normalized_reference=normalized_reference,
            error=str(exc),
        )

    extraction = _extract_pdf_text_and_metadata(
        pdf_path,
        max_chars=max_chars,
        max_pages=max_pages,
    )
    extracted_text = str(extraction.get("text") or "")
    metadata = dict(extraction.get("metadata") or {})
    title = _paper_title_from_metadata(
        explicit_title=paper_title,
        metadata=metadata,
        pdf_path=pdf_path,
    )
    arxiv_id = _extract_arxiv_id(" ".join([paper, normalized_reference, extracted_text]))
    doi = _extract_doi(" ".join([extracted_text, json.dumps(metadata, ensure_ascii=False)]))
    raw_pdf_relative_path = _normalize_relative_path(pdf_path.relative_to(resolved_raw_dir))
    text_path = raw_papers_dir / f"{initial_base}.txt"
    metadata_path = raw_papers_dir / f"{initial_base}.json"
    raw_text_relative_path = _normalize_relative_path(text_path.relative_to(resolved_raw_dir))
    raw_metadata_relative_path = _normalize_relative_path(
        metadata_path.relative_to(resolved_raw_dir)
    )

    if extracted_text:
        text_content = extracted_text
    else:
        text_content = (
            f"No extractable text was found in `{raw_pdf_relative_path}`.\n"
            "Use a PDF OCR or document extraction pass if the paper is scanned."
        )
    _write_text(text_path, text_content)

    paper_record = {
        "input_reference": paper,
        "normalized_reference": normalized_reference,
        "source_url": normalized_reference if _is_url(normalized_reference) else "",
        "source_path": materialization["source"] if materialization["mode"] == "copy" else "",
        "materialization": materialization,
        "title": title,
        "metadata": metadata,
        "page_count": int(extraction.get("page_count", 0) or 0),
        "extraction_error": str(extraction.get("extraction_error") or ""),
        "arxiv_id": arxiv_id or "",
        "doi": doi or "",
        "raw_pdf_relative_path": raw_pdf_relative_path,
        "raw_text_relative_path": raw_text_relative_path,
        "raw_metadata_relative_path": raw_metadata_relative_path,
        "sha256": _sha256_file(pdf_path),
        "imported_at": _now_iso(),
    }
    metadata_path.write_text(
        json.dumps(paper_record, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    target_relative_path = source_note_path or _normalize_relative_path(
        Path("papers") / f"{initial_base}.md"
    )
    related_documents = _related_documents_for_raw(
        source_dir=resolved_source_dir,
        raw_relative_path=raw_pdf_relative_path,
        title=title,
        preview_text=extracted_text,
        exclude_relative_paths={target_relative_path},
        limit=max_related_notes,
    )
    suggested_pages = _suggest_follow_up_pages(
        source_dir=resolved_source_dir,
        related_documents=related_documents,
        title=title,
        preview_text=extracted_text,
        source_note_path=target_relative_path,
        max_suggested_pages=max_suggested_pages,
    )
    source_note = _build_source_note_from_paper(
        title=title,
        paper=paper_record,
        extracted_text=extracted_text,
        related_documents=related_documents,
        suggested_pages=suggested_pages,
    )
    result = wiki_add_document(
        relative_path=target_relative_path,
        content=source_note,
        source_dir=str(resolved_source_dir),
        wiki_path=wiki_path,
        raw_dir=str(resolved_raw_dir),
        vault_dir=vault_dir,
        rebuild=False,
    )
    if result.get("status") == "success":
        updated_related_paths = []
        if update_related:
            for document in related_documents:
                if _append_source_update_reference(
                    source_dir=resolved_source_dir,
                    target_relative_path=document.relative_path,
                    new_note_relative_path=target_relative_path,
                    new_note_title=title,
                    raw_relative_path=raw_pdf_relative_path,
                ):
                    updated_related_paths.append(document.relative_path)
        if rebuild:
            result["build"] = wiki_build(
                source_dir=str(resolved_source_dir),
                wiki_path=wiki_path,
                raw_dir=str(resolved_raw_dir),
                vault_dir=vault_dir,
            )
        result["paper"] = paper_record
        result["raw_pdf_path"] = str(pdf_path)
        result["raw_text_path"] = str(text_path)
        result["raw_metadata_path"] = str(metadata_path)
        result["raw_pdf_relative_path"] = raw_pdf_relative_path
        result["raw_text_relative_path"] = raw_text_relative_path
        result["raw_metadata_relative_path"] = raw_metadata_relative_path
        result["source_note_path"] = target_relative_path
        result["related_documents"] = [
            {"relative_path": document.relative_path, "title": document.title}
            for document in related_documents
        ]
        result["suggested_pages"] = suggested_pages
        result["updated_related_paths"] = updated_related_paths
        result["log_path"] = str(
            _append_log_entry(
                _infer_vault_dir(
                    source_dir=resolved_source_dir,
                    raw_dir=resolved_raw_dir,
                    wiki_path=wiki_path,
                    vault_dir=vault_dir,
                ),
                "ingest",
                title,
                [
                    f"raw_pdf={raw_pdf_relative_path}",
                    f"raw_text={raw_text_relative_path}",
                    f"raw_metadata={raw_metadata_relative_path}",
                    f"source_note={target_relative_path}",
                    f"related_notes={len(related_documents)}",
                    f"suggested_pages={len(suggested_pages)}",
                    f"updated_related={len(updated_related_paths)}",
                    f"rebuild={rebuild}",
                ],
            )
        )
    return result


def wiki_ingest_repo(
    repo_path: str,
    *,
    repo_name: str | None = None,
    source_note_path: str | None = None,
    source_dir: str | None = None,
    raw_dir: str | None = None,
    wiki_path: str | None = None,
    vault_dir: str | None = None,
    rebuild: bool = True,
    max_files: int = 200,
    max_tree_entries: int = 120,
    max_excerpt_files: int = 8,
    max_excerpt_chars: int = 1200,
    max_chars: int = 8000,
    update_related: bool = True,
    max_related_notes: int = 3,
    max_suggested_pages: int = 6,
) -> dict[str, Any]:
    """Create a raw repository snapshot, promote it into a repo note, and optionally rebuild."""
    resolved_raw_dir = _infer_raw_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_source_dir = _infer_source_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_repo_path, resolved_repo_name, materialized = _materialize_repo_for_ingest(
        repo_path,
        raw_dir=resolved_raw_dir,
        repo_name=repo_name,
    )
    if materialized["status"] == "error":
        return _error(
            materialized["message"],
            **{
                key: value
                for key, value in materialized.items()
                if key not in {"status", "message"}
            },
        )

    snapshot = _collect_repo_snapshot(
        resolved_repo_path,
        max_files=max_files,
        max_tree_entries=max_tree_entries,
        max_excerpt_files=max_excerpt_files,
        max_excerpt_chars=max_excerpt_chars,
    )

    raw_snapshot_dir = _ensure_safe_target(
        resolved_raw_dir, Path("repos") / _slugify(resolved_repo_name)
    )
    raw_snapshot_dir.mkdir(parents=True, exist_ok=True)
    raw_snapshot_path = raw_snapshot_dir / "snapshot.md"
    raw_manifest_path = raw_snapshot_dir / "manifest.json"
    graph_artifacts = _refresh_repo_search_graph(raw_snapshot_dir, resolved_repo_path)
    if graph_artifacts:
        snapshot["graph_artifacts"] = graph_artifacts
    _write_text(raw_snapshot_path, _build_repo_snapshot_markdown(snapshot))
    raw_manifest_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")

    raw_snapshot_relative_path = _normalize_relative_path(
        raw_snapshot_path.relative_to(resolved_raw_dir)
    )
    raw_manifest_relative_path = _normalize_relative_path(
        raw_manifest_path.relative_to(resolved_raw_dir)
    )
    target_relative_path = source_note_path or _default_source_note_path_for_repo(
        resolved_repo_name
    )
    preview_text = _raw_artifact_text_preview(raw_snapshot_path, max_chars=max_chars)
    title = f"{resolved_repo_name} Repository"
    related_documents = _related_documents_for_raw(
        source_dir=resolved_source_dir,
        raw_relative_path=raw_snapshot_relative_path,
        title=title,
        preview_text=preview_text,
        exclude_relative_paths={target_relative_path},
        limit=max_related_notes,
    )
    suggested_pages: list[dict[str, Any]] = []
    seen_suggestions: set[str] = set()
    for suggestion in _suggest_follow_up_pages(
        source_dir=resolved_source_dir,
        related_documents=related_documents,
        title=title,
        preview_text=preview_text,
        source_note_path=target_relative_path,
        max_suggested_pages=max_suggested_pages,
    ) + _suggest_repo_follow_up_pages(
        source_dir=resolved_source_dir,
        repo_name=resolved_repo_name,
        test_file_count=snapshot["test_file_count"],
    ):
        if suggestion["relative_path"] in seen_suggestions:
            continue
        suggested_pages.append(suggestion)
        seen_suggestions.add(suggestion["relative_path"])
        if len(suggested_pages) >= max_suggested_pages:
            break

    source_note = _build_source_note_from_repo(
        repo_name=resolved_repo_name,
        repo_root=resolved_repo_path,
        raw_snapshot_relative_path=raw_snapshot_relative_path,
        raw_manifest_relative_path=raw_manifest_relative_path,
        preview_text=preview_text,
        snapshot=snapshot,
        related_documents=related_documents,
        suggested_pages=suggested_pages,
    )
    result = wiki_add_document(
        relative_path=target_relative_path,
        content=source_note,
        source_dir=str(resolved_source_dir),
        wiki_path=wiki_path,
        raw_dir=str(resolved_raw_dir),
        vault_dir=vault_dir,
        rebuild=False,
    )
    if result.get("status") == "success":
        updated_related_paths = []
        if update_related:
            for document in related_documents:
                if _append_source_update_reference(
                    source_dir=resolved_source_dir,
                    target_relative_path=document.relative_path,
                    new_note_relative_path=target_relative_path,
                    new_note_title=title,
                    raw_relative_path=raw_snapshot_relative_path,
                ):
                    updated_related_paths.append(document.relative_path)
        if rebuild:
            result["build"] = wiki_build(
                source_dir=str(resolved_source_dir),
                wiki_path=wiki_path,
                raw_dir=str(resolved_raw_dir),
                vault_dir=vault_dir,
            )
        result["repo_path"] = str(resolved_repo_path)
        result["repo_name"] = resolved_repo_name
        result["repo_reference"] = repo_path
        result["repo_materialization"] = materialized
        result["raw_snapshot_path"] = str(raw_snapshot_path)
        result["raw_manifest_path"] = str(raw_manifest_path)
        result["raw_snapshot_relative_path"] = raw_snapshot_relative_path
        result["raw_manifest_relative_path"] = raw_manifest_relative_path
        result["source_note_path"] = target_relative_path
        result["repo_snapshot"] = {
            "summary": snapshot["summary"],
            "total_file_count": snapshot["total_file_count"],
            "included_file_count": snapshot["included_file_count"],
            "test_file_count": snapshot["test_file_count"],
            "top_level_areas": snapshot["top_level_areas"],
            "key_files": snapshot["key_files"],
        }
        result["related_documents"] = [
            {"relative_path": document.relative_path, "title": document.title}
            for document in related_documents
        ]
        result["suggested_pages"] = suggested_pages
        result["updated_related_paths"] = updated_related_paths
        result["log_path"] = str(
            _append_log_entry(
                _infer_vault_dir(
                    source_dir=resolved_source_dir,
                    raw_dir=resolved_raw_dir,
                    wiki_path=wiki_path,
                    vault_dir=vault_dir,
                ),
                "ingest",
                f"{resolved_repo_name} Repository",
                [
                    f"repo_path={resolved_repo_path}",
                    f"raw_snapshot={raw_snapshot_relative_path}",
                    f"raw_manifest={raw_manifest_relative_path}",
                    f"source_note={target_relative_path}",
                    f"related_notes={len(related_documents)}",
                    f"suggested_pages={len(suggested_pages)}",
                    f"updated_related={len(updated_related_paths)}",
                    f"rebuild={rebuild}",
                ],
            )
        )
    return result


def wiki_list_suggestions(
    source_note_path: str,
    *,
    source_dir: str | None = None,
) -> dict[str, Any]:
    """List suggested follow-up pages embedded in a source note."""
    resolved_source_dir = _infer_source_dir(source_dir=source_dir)
    target = _ensure_safe_target(resolved_source_dir, source_note_path)
    if not target.exists():
        return _error("Source note not found", source_path=str(target))
    suggestions = _parse_suggested_pages_from_note(_read_text(target))
    return _ok(
        source_path=str(target),
        source_note_path=_normalize_relative_path(target.relative_to(resolved_source_dir)),
        count=len(suggestions),
        suggestions=suggestions,
    )


def wiki_promote_suggestion(
    source_note_path: str,
    suggestion_path: str,
    *,
    source_dir: str | None = None,
    wiki_path: str | None = None,
    raw_dir: str | None = None,
    vault_dir: str | None = None,
    rebuild: bool = True,
) -> dict[str, Any]:
    """Promote one suggested follow-up page from a source note into a concrete wiki edit."""
    resolved_source_dir = _infer_source_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    source_note_target = _ensure_safe_target(resolved_source_dir, source_note_path)
    if not source_note_target.exists():
        return _error("Source note not found", source_path=str(source_note_target))

    source_content = _read_text(source_note_target)
    suggestions = _parse_suggested_pages_from_note(source_content)
    selected = next(
        (item for item in suggestions if item["relative_path"] == suggestion_path), None
    )
    if selected is None:
        return _error(
            "Suggestion not found in source note",
            source_note_path=source_note_path,
            suggestion_path=suggestion_path,
            suggestions=suggestions,
        )

    source_note_document = _build_document(source_note_target, resolved_source_dir)
    action = selected["action"]
    target_relative_path = selected["relative_path"]
    reason = selected["reason"]
    result: dict[str, Any]

    if action == "create":
        scaffold = _build_follow_up_note_from_suggestion(
            suggestion_relative_path=target_relative_path,
            reason=reason,
            source_note_relative_path=source_note_document.relative_path,
            source_note_title=source_note_document.title,
        )
        result = wiki_add_document(
            relative_path=target_relative_path,
            content=scaffold,
            source_dir=str(resolved_source_dir),
            wiki_path=wiki_path,
            raw_dir=raw_dir,
            vault_dir=vault_dir,
            rebuild=False,
        )
    else:
        updated = _append_suggested_follow_up_update(
            source_dir=resolved_source_dir,
            target_relative_path=target_relative_path,
            source_note_relative_path=source_note_document.relative_path,
            source_note_title=source_note_document.title,
            reason=reason,
        )
        target_path = _ensure_safe_target(resolved_source_dir, target_relative_path)
        result = _ok(
            target_path=str(target_path),
            relative_path=target_relative_path,
            updated=updated,
        )

    if rebuild:
        result["build"] = wiki_build(
            source_dir=str(resolved_source_dir),
            wiki_path=wiki_path,
            raw_dir=raw_dir,
            vault_dir=vault_dir,
        )

    result["promoted_suggestion"] = selected
    result["log_path"] = str(
        _append_log_entry(
            _infer_vault_dir(
                source_dir=resolved_source_dir,
                raw_dir=raw_dir,
                wiki_path=wiki_path,
                vault_dir=vault_dir,
            ),
            "ingest",
            Path(suggestion_path).name,
            [
                f"source_note={source_note_document.relative_path}",
                f"suggestion_action={action}",
                f"suggestion_path={suggestion_path}",
                f"rebuild={rebuild}",
            ],
        )
    )
    return result


def wiki_update_document(
    relative_path: str,
    *,
    content: str | None = None,
    find_text: str | None = None,
    replace_text: str | None = None,
    replace_all: bool = False,
    append_text: str | None = None,
    prepend_text: str | None = None,
    source_dir: str | None = None,
    wiki_path: str | None = None,
    raw_dir: str | None = None,
    vault_dir: str | None = None,
    rebuild: bool = True,
) -> dict[str, Any]:
    """Modify a source document, then optionally rebuild the wiki."""
    resolved_source_dir = _infer_source_dir(
        source_dir=source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    target = _ensure_safe_target(resolved_source_dir, relative_path)
    if not target.exists():
        return _error("Document not found", source_path=str(target))

    updated = content if content is not None else _read_text(target)
    if find_text is not None:
        if replace_text is None:
            return _error("replace_text is required when find_text is provided")
        if find_text not in updated:
            return _error("find_text was not found in the document", source_path=str(target))
        updated = updated.replace(find_text, replace_text, -1 if replace_all else 1)

    if prepend_text:
        updated = prepend_text + updated
    if append_text:
        separator = (
            "" if not updated or updated.endswith("\n") or append_text.startswith("\n") else "\n"
        )
        updated = f"{updated}{separator}{append_text}"

    target.write_text(updated, encoding="utf-8")
    result = _ok(
        source_path=str(target),
        relative_path=_normalize_relative_path(target.relative_to(resolved_source_dir)),
    )
    result["log_path"] = str(
        _append_log_entry(
            _infer_vault_dir(
                source_dir=resolved_source_dir,
                wiki_path=wiki_path,
                raw_dir=raw_dir,
                vault_dir=vault_dir,
            ),
            "update",
            target.name,
            [
                f"source_note={_normalize_relative_path(target.relative_to(resolved_source_dir))}",
                f"rebuild={rebuild}",
            ],
        )
    )
    if rebuild:
        result["build"] = wiki_build(
            source_dir=str(resolved_source_dir),
            wiki_path=wiki_path,
            raw_dir=raw_dir,
            vault_dir=vault_dir,
        )
    return result


def wiki_write_output(
    relative_path: str,
    content: str,
    *,
    vault_dir: str | None = None,
) -> dict[str, Any]:
    """Write a derived output into the compiled workspace."""
    resolved_vault_dir = _resolve_vault_dir(vault_dir)
    outputs_dir = resolved_vault_dir / "outputs"
    target = _ensure_safe_target(outputs_dir, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _write_text(outputs_dir / "README.md", _build_outputs_index(outputs_dir, resolved_vault_dir))
    return _ok(
        output_path=str(target),
        relative_path=_normalize_relative_path(target.relative_to(resolved_vault_dir)),
        log_path=str(
            _append_log_entry(
                resolved_vault_dir,
                "query",
                target.name,
                [
                    f"filed_output={_normalize_relative_path(target.relative_to(resolved_vault_dir))}"
                ],
            )
        ),
    )


def wiki_health(
    source_dir: str | None = None,
    wiki_path: str | None = None,
    raw_dir: str | None = None,
    vault_dir: str | None = None,
) -> dict[str, Any]:
    """Check whether the monolithic wiki and compiled workspace match the current source tree."""
    resolved_source_dir = _infer_source_dir(source_dir=source_dir, vault_dir=vault_dir)
    resolved_raw_dir = _infer_raw_dir(
        source_dir=resolved_source_dir,
        raw_dir=raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_wiki_path = _infer_wiki_path(
        source_dir=resolved_source_dir,
        raw_dir=resolved_raw_dir,
        wiki_path=wiki_path,
        vault_dir=vault_dir,
    )
    resolved_vault_dir = _infer_vault_dir(
        source_dir=resolved_source_dir,
        raw_dir=resolved_raw_dir,
        wiki_path=resolved_wiki_path,
        vault_dir=vault_dir,
    )
    current_documents = _scan_documents(resolved_source_dir)
    return _compute_health_report(
        resolved_source_dir,
        resolved_raw_dir,
        resolved_wiki_path,
        resolved_vault_dir,
        current_documents,
    )


def wiki_lint(
    source_dir: str | None = None,
    vault_dir: str | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Run structural lint checks over the source documents."""
    resolved_source_dir = _infer_source_dir(source_dir=source_dir, vault_dir=vault_dir)
    resolved_vault_dir = _infer_vault_dir(source_dir=resolved_source_dir, vault_dir=vault_dir)
    documents = _scan_documents(resolved_source_dir)
    relationships = _build_relationships(documents)
    concepts = _build_concepts(documents)
    report = _build_lint_report(documents, relationships, concepts)

    if write_report:
        report_path = resolved_vault_dir / "reports" / "Lint.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(report_path, _build_lint_report_page(report))
        report["report_path"] = str(report_path)
    report["log_path"] = str(
        _append_log_entry(
            resolved_vault_dir,
            "lint",
            "Wiki Lint Pass",
            [f"documents={len(documents)}", f"findings={report['finding_count']}"],
        )
    )
    return report


def wiki_verify(
    query: str | None = None,
    output_dir: str | None = None,
    source_dir: str | None = None,
    raw_dir: str | None = None,
    vault_dir: str | None = None,
) -> dict[str, Any]:
    """Verify that the wiki is healthy and can answer a sample query."""
    health = wiki_health(
        source_dir=source_dir, wiki_path=output_dir, raw_dir=raw_dir, vault_dir=vault_dir
    )
    if health["status"] == "error":
        return health

    result: dict[str, Any] = {
        "status": health["status"],
        "health": health,
        "lint": wiki_lint(source_dir, vault_dir),
    }
    if query:
        result["search"] = wiki_search(pattern=query, wiki_path=output_dir)
    result["log_path"] = str(
        _append_log_entry(
            _infer_vault_dir(
                source_dir=source_dir, raw_dir=raw_dir, wiki_path=output_dir, vault_dir=vault_dir
            ),
            "verify",
            "Wiki Verification",
            [
                f"health={health['status']}",
                f"query={query or '<none>'}",
            ],
        )
    )
    return result


# Convenience aliases
wiki = wiki_list
build = wiki_build
recreate = wiki_recreate
search = wiki_search
search_repo = wiki_search_repo
add_dir = wiki_add_dir
add_file = wiki_add_file
add_document = wiki_add_document
add_raw_document = wiki_add_raw_document
add_raw_file = wiki_add_raw_file
add_raw_paper = wiki_add_raw_paper
ingest_raw = wiki_ingest_raw
ingest_repo = wiki_ingest_repo
list_suggestions = wiki_list_suggestions
promote_suggestion = wiki_promote_suggestion
update_document = wiki_update_document
write_output = wiki_write_output
load = wiki_load_index
read_source_note = wiki_read_source_note
get = wiki_get_document
health = wiki_health
lint = wiki_lint
verify = wiki_verify
list_sources = wiki_list_sources
list_raw = wiki_list_raw

# Export tools for registry discovery

__skill__ = {
    "name": "Wiki Operations",
    "description": "Concrete operations on the markdown wiki corpus: build, search, ingest, update, and health checks.",
    "aliases": ["wiki ops", "wiki maintenance", "wiki compiler"],
    "triggers": [
        "build wiki",
        "rebuild wiki",
        "search wiki",
        "read wiki source note",
        "search ingested repo",
        "add wiki document",
        "update wiki document",
        "add raw note",
        "add raw paper",
        "ingest paper",
        "ingest raw source",
        "ingest repo code",
        "promote wiki suggestion",
        "file wiki output",
        "wiki health",
        "wiki lint",
    ],
    "preferred_tools": [
        "wiki_recreate",
        "wiki_ingest_repo",
        "wiki_add_raw_paper",
        "wiki_ingest_raw",
        "wiki_promote_suggestion",
    ],
    "example_queries": [
        "rebuild the wiki from source",
        "search wiki for deployment timeout",
        "read repos/scip.md directly",
        "search the highs repo for branch and bound",
        "add a new page to the wiki",
        "ingest this note into raw and rebuild",
        "download this arXiv paper into raw/papers and ingest it",
        "ingest this repository into the wiki",
        "promote raw/research/article.txt into a wiki note",
        "promote a suggested concept page from a source note",
        "check whether wiki.md is stale",
    ],
    "when_not_to_use": [
        "vector or embedding retrieval",
        "generic repo search unrelated to wiki",
    ],
}
