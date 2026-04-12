from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

WIKI_ROOT = "wiki"
WIKI_FILE = "wiki.md"
WIKI_HOME = "dist/Home.md"
WIKI_INDEX_PATHS = (
    "dist/index.md",
    "dist/indexes/Documents.md",
    "dist/indexes/Concepts.md",
    "dist/indexes/Backlinks.md",
    "dist/reports/Health.md",
    "dist/reports/Lint.md",
)
MANIFEST_START = "<!-- WIKI_MANIFEST_START -->"
MANIFEST_END = "<!-- WIKI_MANIFEST_END -->"


def _wiki_root(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    candidate = root / WIKI_ROOT
    if candidate.exists():
        return candidate.resolve()
    fallback = root / "wiki_ingest"
    if fallback.exists():
        return fallback.resolve()
    return candidate.resolve()


def _wiki_path(base_dir: str | Path | None = None) -> Path:
    return _wiki_root(base_dir) / WIKI_FILE


def _compact(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _query_terms(query: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]{1,}", str(query or ""))
        if len(token) >= 2
    ]


def _score_match(query: str, *parts: str) -> int:
    terms = _query_terms(query)
    if not terms:
        return 0
    haystack = "\n".join(str(part or "") for part in parts).lower()
    score = 0
    for term in terms:
        if term in haystack:
            score += 6
        split_term = term.replace("/", " ").replace(".", " ").replace("-", " ").split()
        if len(split_term) > 1 and all(piece in haystack for piece in split_term):
            score += 3
    return score


def _extract_manifest(content: str) -> dict[str, Any]:
    try:
        start = content.index(MANIFEST_START) + len(MANIFEST_START)
        end = content.index(MANIFEST_END, start)
    except ValueError:
        return {}
    block = content[start:end].strip()
    if block.startswith("```json"):
        block = block[len("```json") :].strip()
    if block.endswith("```"):
        block = block[:-3].rstrip()
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def wiki_workspace_summary(base_dir: str | Path | None = None) -> dict[str, Any]:
    wiki_path = _wiki_path(base_dir)
    if not wiki_path.exists():
        return {}
    content = wiki_path.read_text(encoding="utf-8", errors="replace")
    manifest = _extract_manifest(content)
    if not manifest:
        return {}
    return {
        "wiki_path": str(wiki_path),
        "vault_dir": str(manifest.get("vault_dir") or ""),
        "source_dir": str(manifest.get("source_dir") or ""),
        "raw_dir": str(manifest.get("raw_dir") or ""),
        "document_count": int(manifest.get("document_count", 0) or 0),
        "raw_artifact_count": int(manifest.get("raw_artifact_count", 0) or 0),
    }


def _candidate_files(base_dir: str | Path | None = None) -> list[Path]:
    root = _wiki_root(base_dir)
    files = []
    for rel in (WIKI_HOME, *WIKI_INDEX_PATHS):
        path = root / rel
        if path.exists() and path.is_file():
            files.append(path)
    vault_paths = [root / "dist", root / "wiki"]
    for vault_path in vault_paths:
        articles_dir = vault_path / "articles"
        if articles_dir.exists():
            files.extend(sorted(path for path in articles_dir.glob("*.md") if path.is_file()))
        outputs_dir = vault_path / "outputs"
        if outputs_dir.exists():
            files.extend(sorted(path for path in outputs_dir.rglob("*.md") if path.is_file()))
        if files:
            break
    return files


def build_wiki_context(
    query: str,
    *,
    base_dir: str | Path | None = None,
    max_chars: int = 1400,
    max_results: int = 4,
) -> str:
    summary = wiki_workspace_summary(base_dir)
    if not summary:
        return ""

    lines = ["Use this local wiki workspace if it helps answer the request:"]
    lines.append(
        "Workspace snapshot: "
        f"{summary['document_count']} compiled notes, {summary['raw_artifact_count']} raw artifacts."
    )
    if summary.get("vault_dir"):
        lines.append(
            f"Primary entry points: `{summary['vault_dir']}/Home.md` and the wiki index pages."
        )

    files = _candidate_files(base_dir)
    query = str(query or "").strip()
    matches: list[tuple[int, Path, str]] = []
    top_names = {
        "Home.md",
        "index.md",
        "schema.md",
        "Documents.md",
        "Concepts.md",
        "Backlinks.md",
        "Health.md",
        "Lint.md",
    }
    for path in files:
        content = path.read_text(encoding="utf-8", errors="replace")
        score = 0
        if query:
            score = _score_match(query, path.name, path.as_posix(), content)
            if path.name in {"Home.md", "index.md", "schema.md"}:
                score = max(score, 10)
            if score <= 0:
                continue
        else:
            if path.name not in top_names:
                continue
            score = 1
        matches.append((score, path, _compact(content, 260)))

    matches.sort(key=lambda item: (-item[0], str(item[1])))
    if matches:
        lines.append("Relevant wiki pages:")
        seen: set[str] = set()
        for _, path, preview in matches:
            rel = path.relative_to(_wiki_root(base_dir)).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            lines.append(f"- {rel} — {preview}")
            if len(seen) >= max(1, int(max_results)):
                break

    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, int(max_chars) - 4)].rstrip() + " ..."
