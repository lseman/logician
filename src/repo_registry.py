from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_repo_root(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    return root / ".logician" / "repos"


def get_repo_checkout_root(base_dir: str | Path | None = None) -> Path:
    return get_repo_root(base_dir) / "_checkouts"


def get_repo_index_path(base_dir: str | Path | None = None) -> Path:
    return get_repo_root(base_dir) / "index.json"


def load_repo_index(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    path = get_repo_index_path(base_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def save_repo_index(index: list[dict[str, Any]], base_dir: str | Path | None = None) -> None:
    root = get_repo_root(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    get_repo_index_path(base_dir).write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def slugify_repo_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return slug or "repo"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _repo_fingerprint(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]


def _git_value(path: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(path),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "").strip()


def detect_git_metadata(path: Path) -> dict[str, str]:
    branch = _git_value(path, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _git_value(path, "rev-parse", "--short", "HEAD")
    remote = _git_value(path, "config", "--get", "remote.origin.url")
    return {
        "branch": branch,
        "commit": commit,
        "remote": remote,
    }


def ensure_repo_artifacts(
    repo_id: str,
    *,
    name: str,
    path: str,
    base_dir: str | Path | None = None,
) -> dict[str, str]:
    repo_root = get_repo_root(base_dir)
    repo_dir = repo_root / repo_id
    repo_dir.mkdir(parents=True, exist_ok=True)

    graph_path = repo_dir / "graph.jsonl"
    summary_path = repo_dir / "summary.md"
    manifest_path = repo_dir / "manifest.json"

    if not graph_path.exists():
        graph_path.write_text("", encoding="utf-8")
    if not summary_path.exists():
        summary_path.write_text(
            "\n".join(
                [
                    f"# {name}",
                    "",
                    f"- path: {path}",
                    "- graph_status: pending",
                    "- summary_status: pending",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if not manifest_path.exists():
        manifest_path.write_text("{}", encoding="utf-8")

    return {
        "repo_dir": str(repo_dir),
        "graph_path": str(graph_path),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
    }


def _normalize_repo_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    normalized["id"] = str(entry.get("id") or "").strip()
    normalized["name"] = str(entry.get("name") or normalized["id"] or "repo").strip()
    normalized["path"] = str(entry.get("path") or "").strip()
    normalized["source_url"] = str(entry.get("source_url") or "").strip()
    normalized["added_at"] = str(entry.get("added_at") or _utc_now()).strip()
    normalized["last_used_at"] = str(entry.get("last_used_at") or "").strip()
    normalized["last_ingested_at"] = str(entry.get("last_ingested_at") or "").strip()
    normalized["last_graph_built_at"] = str(entry.get("last_graph_built_at") or "").strip()
    normalized["files_processed"] = int(entry.get("files_processed", 0) or 0)
    normalized["chunks_added"] = int(entry.get("chunks_added", 0) or 0)
    normalized["graph_nodes"] = int(entry.get("graph_nodes", 0) or 0)
    normalized["graph_edges"] = int(entry.get("graph_edges", 0) or 0)
    normalized["graph_symbols"] = int(entry.get("graph_symbols", 0) or 0)
    normalized["glob"] = str(entry.get("glob") or "").strip()
    git = dict(entry.get("git") or {})
    normalized["git"] = {
        "branch": str(git.get("branch") or "").strip(),
        "commit": str(git.get("commit") or "").strip(),
        "remote": str(git.get("remote") or "").strip(),
    }
    normalized["artifacts"] = dict(entry.get("artifacts") or {})
    return normalized


def register_repo(
    path: str,
    *,
    name: str = "",
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    repo_path = Path(path).expanduser().resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise FileNotFoundError(f"Repository path not found: {path}")

    repo_name = str(name or repo_path.name).strip() or repo_path.name or "repo"
    index = load_repo_index(base_dir)
    repo_path_text = str(repo_path)

    for idx, item in enumerate(index):
        if str(item.get("path") or "").strip() != repo_path_text:
            continue
        existing = _normalize_repo_entry(item)
        existing["name"] = repo_name
        existing["git"] = detect_git_metadata(repo_path)
        existing["artifacts"] = ensure_repo_artifacts(
            existing["id"],
            name=repo_name,
            path=repo_path_text,
            base_dir=base_dir,
        )
        index[idx] = existing
        save_repo_index(index, base_dir)
        (Path(existing["artifacts"]["manifest_path"])).write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return existing

    slug = slugify_repo_name(repo_name)
    taken_ids = {
        str(item.get("id") or "").strip()
        for item in index
        if str(item.get("id") or "").strip()
    }
    repo_id = slug
    if repo_id in taken_ids:
        repo_id = f"{slug}-{_repo_fingerprint(repo_path)}"

    entry = _normalize_repo_entry(
        {
            "id": repo_id,
            "name": repo_name,
            "path": repo_path_text,
            "added_at": _utc_now(),
            "git": detect_git_metadata(repo_path),
        }
    )
    entry["artifacts"] = ensure_repo_artifacts(
        repo_id,
        name=repo_name,
        path=repo_path_text,
        base_dir=base_dir,
    )
    index.append(entry)
    index.sort(key=lambda item: str(item.get("name") or item.get("id") or "").lower())
    save_repo_index(index, base_dir)
    (Path(entry["artifacts"]["manifest_path"])).write_text(
        json.dumps(entry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return entry


def update_repo(
    repo_id: str,
    *,
    base_dir: str | Path | None = None,
    **fields: Any,
) -> dict[str, Any] | None:
    index = load_repo_index(base_dir)
    for idx, item in enumerate(index):
        if str(item.get("id") or "").strip() != str(repo_id or "").strip():
            continue
        updated = _normalize_repo_entry({**item, **fields})
        updated["artifacts"] = ensure_repo_artifacts(
            updated["id"],
            name=updated["name"],
            path=updated["path"],
            base_dir=base_dir,
        )
        index[idx] = updated
        save_repo_index(index, base_dir)
        (Path(updated["artifacts"]["manifest_path"])).write_text(
            json.dumps(updated, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return updated
    return None


def remove_repo(repo_id: str, *, base_dir: str | Path | None = None) -> dict[str, Any] | None:
    index = load_repo_index(base_dir)
    kept: list[dict[str, Any]] = []
    removed: dict[str, Any] | None = None
    for item in index:
        if str(item.get("id") or "").strip() == str(repo_id or "").strip():
            removed = _normalize_repo_entry(item)
            continue
        kept.append(_normalize_repo_entry(item))
    if removed is None:
        return None
    save_repo_index(kept, base_dir)
    return removed
