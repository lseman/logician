from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import Config
from .db.document import DocumentDB
from .repo_graph import _collect_matching_files, build_repo_graph
from .repo_registry import (
    get_repo_checkout_root,
    load_repo_index,
    register_repo,
    slugify_repo_name,
    update_repo,
)
from .runtime_paths import state_path

DEFAULT_REPO_GLOB = (
    "**/*.{py,rs,ts,tsx,js,jsx,java,go,rb,php,c,cc,cpp,h,hpp,cs,kt,swift,"
    "md,toml,yaml,yml,json,sql,sh}"
)
DEFAULT_REPO_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_REPO_VECTOR_BACKEND = "hnsw"
DEFAULT_AGENT_CONFIG_NAME = "agent_config.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_vector_path(base_dir: Path) -> Path:
    del base_dir
    return state_path("rag_docs.vector")


def _load_workspace_agent_config(base_dir: Path) -> dict[str, Any]:
    candidates = [
        base_dir / DEFAULT_AGENT_CONFIG_NAME,
        Path.cwd().resolve() / DEFAULT_AGENT_CONFIG_NAME,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if not candidate.exists() or not candidate.is_file():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _resolve_repo_ingest_settings(
    *,
    workspace_root: Path,
    embedding_model_name: str | None,
    vector_backend: str | None,
) -> tuple[str, str]:
    config_defaults = Config()
    config_payload = _load_workspace_agent_config(workspace_root)
    default_embedding_model = getattr(config_defaults, "embedding_model", None)
    default_rag_vector_backend = getattr(config_defaults, "rag_vector_backend", None)
    default_vector_backend = getattr(config_defaults, "vector_backend", None)

    resolved_embedding_model = str(
        embedding_model_name
        or config_payload.get("embedding_model")
        or default_embedding_model
        or DEFAULT_REPO_EMBEDDING_MODEL
    ).strip()
    if not resolved_embedding_model:
        resolved_embedding_model = DEFAULT_REPO_EMBEDDING_MODEL

    resolved_vector_backend = (
        str(
            vector_backend
            or config_payload.get("rag_vector_backend")
            or config_payload.get("vector_backend")
            or default_rag_vector_backend
            or default_vector_backend
            or DEFAULT_REPO_VECTOR_BACKEND
        )
        .strip()
        .lower()
    )
    if not resolved_vector_backend:
        resolved_vector_backend = DEFAULT_REPO_VECTOR_BACKEND

    return resolved_embedding_model, resolved_vector_backend


def _looks_like_git_url(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https", "ssh", "git", "file"}:
        return True
    return bool(re.match(r"^[A-Za-z0-9_.-]+@[^:]+:.+$", text))


def _repo_name_from_source(source: str) -> str:
    text = str(source or "").strip().rstrip("/")
    if not text:
        return "repo"
    if _looks_like_git_url(text):
        if "://" in text:
            path = urlparse(text).path
        elif ":" in text:
            path = text.split(":", 1)[1]
        else:
            path = text
        name = Path(path).name or Path(path).stem
        if name.endswith(".git"):
            name = name[:-4]
        return name or "repo"
    return Path(text).expanduser().resolve().name or "repo"


def _git_value(path: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(path),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "").strip()


def _existing_repo_for_source_url(
    source_url: str,
    *,
    base_dir: Path,
) -> dict[str, Any] | None:
    source = str(source_url or "").strip()
    if not source:
        return None
    for item in load_repo_index(base_dir):
        if str(item.get("source_url") or "").strip() != source:
            continue
        repo_path = Path(str(item.get("path") or "")).expanduser()
        if repo_path.exists() and repo_path.is_dir():
            return dict(item)
    return None


def _pick_checkout_dir(
    source_url: str,
    *,
    workspace_root: Path,
    repo_name: str,
) -> Path:
    checkout_root = get_repo_checkout_root(workspace_root)
    checkout_root.mkdir(parents=True, exist_ok=True)

    slug = slugify_repo_name(repo_name or _repo_name_from_source(source_url))
    candidate = checkout_root / slug
    if not candidate.exists():
        return candidate

    existing_remote = _git_value(candidate, "config", "--get", "remote.origin.url")
    if existing_remote and existing_remote == str(source_url or "").strip():
        return candidate

    suffix = 2
    while True:
        alternative = checkout_root / f"{slug}-{suffix}"
        if not alternative.exists():
            return alternative
        existing_remote = _git_value(alternative, "config", "--get", "remote.origin.url")
        if existing_remote and existing_remote == str(source_url or "").strip():
            return alternative
        suffix += 1


def _format_clone_failure(detail: str, attempts: list[str]) -> str:
    lines = [str(detail or "").strip()] if str(detail or "").strip() else []
    if attempts:
        lines.append("clone attempts:")
        lines.extend(f"- {attempt}" for attempt in attempts)
    return "\n".join(lines).strip() or "git clone failed"


def _clone_repository(source_url: str, checkout_dir: Path) -> None:
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = checkout_dir.parent / f".tmp-{checkout_dir.name}-{uuid.uuid4().hex[:8]}"
    attempts = [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--filter=blob:none",
            source_url,
            str(temp_dir),
        ],
        [
            "git",
            "-c",
            "http.version=HTTP/1.1",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            source_url,
            str(temp_dir),
        ],
        ["git", "clone", source_url, str(temp_dir)],
    ]

    failure_messages: list[str] = []
    try:
        for cmd in attempts:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode == 0:
                temp_dir.replace(checkout_dir)
                return

            stderr = str(proc.stderr or "").strip()
            stdout = str(proc.stdout or "").strip()
            detail = stderr or stdout or "git clone failed"
            failure_messages.append(f"`{' '.join(cmd[:-1])} <target>`\n{detail}")
            shutil.rmtree(temp_dir, ignore_errors=True)

        raise RuntimeError(_format_clone_failure("\n\n".join(failure_messages), []))
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _resolve_repo_input(
    repo_source: str | Path,
    *,
    name: str,
    base_dir: str | Path | None,
) -> tuple[Path, Path, str]:
    source_text = str(repo_source or "").strip()
    if not source_text:
        raise FileNotFoundError("Repository path or URL is required")

    if not _looks_like_git_url(source_text):
        resolved_repo_path = Path(source_text).expanduser().resolve()
        if not resolved_repo_path.exists() or not resolved_repo_path.is_dir():
            raise FileNotFoundError(f"Repository path not found: {repo_source}")
        workspace_root = (
            Path(base_dir).expanduser().resolve() if base_dir is not None else resolved_repo_path
        )
        return resolved_repo_path, workspace_root, ""

    workspace_root = (
        Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd().resolve()
    )
    existing = _existing_repo_for_source_url(source_text, base_dir=workspace_root)
    if existing is not None:
        existing_path = Path(str(existing.get("path") or "")).expanduser().resolve()
        if existing_path.exists() and existing_path.is_dir():
            return existing_path, workspace_root, source_text

    repo_name = str(name or _repo_name_from_source(source_text)).strip() or "repo"
    checkout_dir = _pick_checkout_dir(
        source_text,
        workspace_root=workspace_root,
        repo_name=repo_name,
    )
    _clone_repository(source_text, checkout_dir)

    return checkout_dir.resolve(), workspace_root, source_text


def _file_kind_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {
        ".py",
        ".rs",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".go",
        ".rb",
        ".php",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".kt",
        ".swift",
        ".sh",
        ".sql",
    }:
        return "code"
    if ext in {".md", ".rst", ".txt", ".adoc"}:
        return "doc"
    if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "config"
    return "file"


def _repo_metadata(
    path: Path,
    *,
    repo_id: str,
    repo_name: str,
    repo_root: Path,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": path.name,
        "path": str(path),
        "file_kind": _file_kind_for_path(path),
        "ext": path.suffix.lower(),
        "repo_id": repo_id,
        "repo_name": repo_name,
        "repo_root": str(repo_root),
    }
    try:
        metadata["repo_rel_path"] = str(path.relative_to(repo_root))
    except Exception:
        pass
    return metadata


def _ingest_repo_documents(
    *,
    repo: dict[str, Any],
    repo_path: Path,
    vector_path: Path,
    embedding_model_name: str,
    vector_backend: str,
    glob_pattern: str,
    max_files: int,
    chunk_size: int,
    overlap: float,
    exclude: str,
) -> dict[str, Any]:
    files = _collect_matching_files(
        repo_path,
        glob_pattern=glob_pattern,
        max_files=max_files,
        exclude=exclude,
    )
    if not files:
        return {
            "status": "ok",
            "files_processed": 0,
            "total_chunks_added": 0,
            "results": [],
        }

    vector_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    error_count = 0
    total_chunks = 0
    files_processed = 0

    repo_id = str(repo.get("id") or "").strip()
    repo_name = str(repo.get("name") or repo_id or "repo").strip()

    for path in files:
        rel_path = str(path.relative_to(repo_path))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            results.append({"file": rel_path, "status": "error", "error": str(exc)})
            error_count += 1
            continue
        if not text.strip():
            results.append({"file": rel_path, "status": "skipped", "reason": "empty"})
            continue
        try:
            doc_db = DocumentDB(
                vector_path=str(vector_path),
                embedding_model_name=embedding_model_name,
                vector_backend=vector_backend,
                rerank_enabled=False,
            )
            chunk_ids = doc_db.add_documents(
                texts=[text],
                metadatas=[
                    _repo_metadata(
                        path,
                        repo_id=repo_id,
                        repo_name=repo_name,
                        repo_root=repo_path,
                    )
                ],
                chunk_size_tokens=chunk_size,
                chunk_overlap_ratio=overlap,
            )
            chunk_count = len(chunk_ids)
            total_chunks += chunk_count
            files_processed += 1
            results.append(
                {
                    "file": rel_path,
                    "status": "ok",
                    "chunks": chunk_count,
                }
            )
        except Exception as exc:
            error_count += 1
            results.append({"file": rel_path, "status": "error", "error": str(exc)})
            continue

    if files_processed <= 0:
        return {
            "status": "error" if error_count else "ok",
            "files_processed": 0,
            "total_chunks_added": 0,
            "results": results,
            "error": "No readable non-empty files matched the ingest selection."
            if error_count
            else "",
        }

    return {
        "status": "partial" if error_count else "ok",
        "files_processed": files_processed,
        "total_chunks_added": total_chunks,
        "results": results,
        "errors": error_count,
    }


def _delete_existing_repo_chunks(
    *,
    vector_path: Path,
    repo_id: str,
    embedding_model_name: str,
    vector_backend: str,
) -> int:
    clean_repo_id = str(repo_id or "").strip()
    if not clean_repo_id:
        return 0

    db = DocumentDB(
        vector_path=str(vector_path),
        embedding_model_name=embedding_model_name,
        vector_backend=vector_backend,
        rerank_enabled=False,
    )
    existing = int(db.count(where={"repo_id": clean_repo_id}) or 0)
    if existing > 0:
        db.delete(where={"repo_id": clean_repo_id})
    return existing


def ingest_repo(
    repo_path: str | Path,
    *,
    name: str = "",
    base_dir: str | Path | None = None,
    glob_pattern: str = DEFAULT_REPO_GLOB,
    max_files: int = 120,
    chunk_size: int = 400,
    overlap: float = 0.2,
    exclude: str = "",
    vector_path: str | Path | None = None,
    embedding_model_name: str | None = None,
    vector_backend: str | None = None,
    purge_existing: bool = True,
) -> dict[str, Any]:
    resolved_repo_path, workspace_root, source_url = _resolve_repo_input(
        repo_path,
        name=name,
        base_dir=base_dir,
    )
    repo = register_repo(
        str(resolved_repo_path),
        name=name,
        base_dir=workspace_root,
    )

    resolved_vector_path = (
        Path(vector_path).expanduser().resolve()
        if vector_path is not None
        else _default_vector_path(workspace_root).resolve()
    )
    resolved_embedding_model, resolved_vector_backend = _resolve_repo_ingest_settings(
        workspace_root=workspace_root,
        embedding_model_name=embedding_model_name,
        vector_backend=vector_backend,
    )
    deleted_chunks = 0
    if purge_existing:
        deleted_chunks = _delete_existing_repo_chunks(
            vector_path=resolved_vector_path,
            repo_id=str(repo.get("id") or ""),
            embedding_model_name=resolved_embedding_model,
            vector_backend=resolved_vector_backend,
        )
    promote_payload = _ingest_repo_documents(
        repo=repo,
        repo_path=resolved_repo_path,
        vector_path=resolved_vector_path,
        embedding_model_name=resolved_embedding_model,
        vector_backend=resolved_vector_backend,
        glob_pattern=glob_pattern,
        max_files=max_files,
        chunk_size=chunk_size,
        overlap=overlap,
        exclude=exclude,
    )

    graph_payload = build_repo_graph(
        repo,
        glob_pattern=glob_pattern,
        max_files=max_files,
        exclude=exclude,
        base_dir=workspace_root,
    )

    now = _utc_now()
    repo_updates: dict[str, Any] = {
        "last_used_at": now,
        "glob": glob_pattern,
    }
    if source_url:
        repo_updates["source_url"] = source_url
    ingest_status = str(promote_payload.get("status") or "").lower()
    if ingest_status in {"ok", "partial"}:
        repo_updates.update(
            {
                "last_ingested_at": now,
                "files_processed": int(promote_payload.get("files_processed", 0) or 0),
                "chunks_added": int(promote_payload.get("total_chunks_added", 0) or 0),
            }
        )
    if str(graph_payload.get("status") or "").lower() == "ok":
        repo_updates.update(
            {
                "last_graph_built_at": now,
                "graph_nodes": int(graph_payload.get("nodes", 0) or 0),
                "graph_edges": int(graph_payload.get("edges", 0) or 0),
                "graph_symbols": int(graph_payload.get("symbols", 0) or 0),
            }
        )

    updated_repo = (
        update_repo(
            str(repo.get("id") or ""),
            base_dir=workspace_root,
            **repo_updates,
        )
        or repo
    )

    ingest_ok = ingest_status in {"ok", "partial"}
    graph_ok = str(graph_payload.get("status") or "").lower() == "ok"
    if ingest_ok and graph_ok:
        status = "ok"
    elif ingest_ok or graph_ok:
        status = "partial"
    else:
        status = "error"

    errors: list[str] = []
    if not ingest_ok:
        errors.append(f"RAG ingest failed: {promote_payload.get('error', 'unknown error')}")
    if not graph_ok:
        errors.append(f"Graph build failed: {graph_payload.get('error', 'unknown error')}")

    return {
        "status": status,
        "repo": updated_repo,
        "source_url": source_url,
        "workspace_root": str(workspace_root),
        "vector_path": str(resolved_vector_path),
        "vector_backend": resolved_vector_backend,
        "embedding_model_name": resolved_embedding_model,
        "deleted_chunks": deleted_chunks,
        "ingest": promote_payload,
        "graph": graph_payload,
        "errors": errors,
    }


def migrate_registered_repos(
    *,
    base_dir: str | Path | None = None,
    vector_path: str | Path | None = None,
    embedding_model_name: str | None = None,
    vector_backend: str | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    workspace_root = (
        Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd().resolve()
    )
    repos = load_repo_index(workspace_root)
    results: list[dict[str, Any]] = []
    status = "ok"
    for item in repos:
        if not isinstance(item, dict):
            continue
        repo_path = str(item.get("path") or "").strip()
        if not repo_path:
            continue
        cmd = [
            sys.executable,
            "-m",
            "src.repo_ingest",
            repo_path,
            "--name",
            str(item.get("name") or "").strip(),
            "--base-dir",
            str(workspace_root),
            "--glob",
            str(item.get("glob") or DEFAULT_REPO_GLOB).strip() or DEFAULT_REPO_GLOB,
            "--max-files",
            str(int(max_files or item.get("files_processed", 0) or 120)),
            "--json",
        ]
        if vector_path is not None:
            cmd.extend(["--vector-path", str(Path(vector_path).expanduser().resolve())])
        if embedding_model_name is not None:
            cmd.extend(["--embedding-model", str(embedding_model_name)])
        if vector_backend is not None:
            cmd.extend(["--vector-backend", str(vector_backend)])
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if proc.returncode == 0:
            try:
                payload = json.loads(str(proc.stdout or "").strip())
            except Exception as exc:
                payload = {
                    "status": "error",
                    "errors": [f"Could not parse migration payload: {exc}"],
                    "ingest": {},
                    "repo": {"id": str(item.get("id") or "")},
                }
        else:
            detail = str(proc.stderr or "").strip() or str(proc.stdout or "").strip()
            payload = {
                "status": "error",
                "errors": [detail or "repo migration subprocess failed"],
                "ingest": {},
                "repo": {"id": str(item.get("id") or "")},
            }
        results.append(
            {
                "repo_id": str((payload.get("repo") or {}).get("id") or ""),
                "status": str(payload.get("status") or ""),
                "deleted_chunks": int(payload.get("deleted_chunks", 0) or 0),
                "chunks_added": int(
                    ((payload.get("ingest") or {}).get("total_chunks_added", 0) or 0)
                ),
                "files_processed": int(
                    ((payload.get("ingest") or {}).get("files_processed", 0) or 0)
                ),
                "errors": list(payload.get("errors") or []),
            }
        )
        if str(payload.get("status") or "").lower() != "ok":
            status = "partial" if status == "ok" else status

    return {
        "status": status,
        "count": len(results),
        "workspace_root": str(workspace_root),
        "vector_path": str(
            Path(vector_path).expanduser().resolve()
            if vector_path is not None
            else _default_vector_path(workspace_root).resolve()
        ),
        "vector_backend": str(vector_backend or "").strip(),
        "embedding_model_name": str(embedding_model_name or "").strip(),
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Register a repo, ingest it into the RAG store, and build repo graph artifacts."
        )
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        help="Path or git URL for the repository to ingest.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Optional display name for the repo. Defaults to the directory name.",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help=("Workspace root for .logician artifacts. Defaults to the target repo path."),
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=DEFAULT_REPO_GLOB,
        help="Glob pattern used for both RAG ingest and graph building.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=120,
        help="Maximum number of files to ingest and graph.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Target token chunk size for RAG ingestion.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Chunk overlap ratio for RAG ingestion.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma- or newline-separated relative paths to exclude.",
    )
    parser.add_argument(
        "--vector-path",
        default=None,
        help="Optional explicit path for the RAG vector store.",
    )
    parser.add_argument(
        "--vector-backend",
        default=None,
        help="Optional vector backend override for repo ingestion.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional embedding model override for repo ingestion.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result payload as JSON.",
    )
    parser.add_argument(
        "--migrate-existing",
        action="store_true",
        help="Re-ingest all repos already registered in the repo index into the shared vector store.",
    )
    parser.add_argument(
        "--no-purge-existing",
        action="store_true",
        help="Keep existing chunks for the repo instead of deleting them before re-ingest.",
    )
    parser.add_argument("--internal-add-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-vector-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-repo-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--internal-repo-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--internal-repo-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-chunk-size", type=int, default=400, help=argparse.SUPPRESS)
    parser.add_argument("--internal-overlap", type=float, default=0.2, help=argparse.SUPPRESS)
    return parser


def _print_human_summary(payload: dict[str, Any]) -> None:
    repo = dict(payload.get("repo") or {})
    ingest = dict(payload.get("ingest") or {})
    graph = dict(payload.get("graph") or {})
    artifacts = dict(repo.get("artifacts") or {})

    print(f"status: {payload.get('status', 'unknown')}")
    print(f"repo: {repo.get('id', '?')} · {repo.get('name', '?')} · {repo.get('path', '-')}")
    if payload.get("source_url"):
        print(f"source_url: {payload.get('source_url')}")
    print(f"workspace_root: {payload.get('workspace_root', '-')}")
    print(f"vector_path: {payload.get('vector_path', '-')}")
    if int(payload.get("deleted_chunks", 0) or 0) > 0:
        print(f"deleted_chunks: {int(payload.get('deleted_chunks', 0) or 0)}")
    print(
        "ingest: "
        f"{ingest.get('status', 'unknown')} · "
        f"files={int(ingest.get('files_processed', 0) or 0)} · "
        f"chunks={int(ingest.get('total_chunks_added', 0) or 0)}"
    )
    print(
        "graph: "
        f"{graph.get('status', 'unknown')} · "
        f"files={int(graph.get('files_indexed', 0) or 0)} · "
        f"nodes={int(graph.get('nodes', 0) or 0)} · "
        f"edges={int(graph.get('edges', 0) or 0)}"
    )
    if artifacts:
        print(f"repo_dir: {artifacts.get('repo_dir', '-')}")
        print(f"manifest: {artifacts.get('manifest_path', '-')}")
        print(f"summary: {artifacts.get('summary_path', '-')}")
        print(f"graph_file: {artifacts.get('graph_path', '-')}")
    for error in payload.get("errors") or []:
        print(f"error: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.internal_add_file:
        file_path = Path(str(args.internal_add_file)).expanduser().resolve()
        repo_root = (
            Path(str(args.internal_repo_root)).expanduser().resolve()
            if args.internal_repo_root is not None
            else file_path.parent
        )
        text = file_path.read_text(encoding="utf-8", errors="replace")
        doc_db = DocumentDB(
            vector_path=str(Path(str(args.internal_vector_path)).expanduser().resolve()),
            embedding_model_name=str(args.embedding_model or DEFAULT_REPO_EMBEDDING_MODEL),
            vector_backend=str(args.vector_backend or DEFAULT_REPO_VECTOR_BACKEND),
            rerank_enabled=False,
        )
        chunk_ids = doc_db.add_documents(
            texts=[text],
            metadatas=[
                _repo_metadata(
                    file_path,
                    repo_id=str(args.internal_repo_id or "").strip(),
                    repo_name=str(args.internal_repo_name or "").strip(),
                    repo_root=repo_root,
                )
            ],
            chunk_size_tokens=int(args.internal_chunk_size),
            chunk_overlap_ratio=float(args.internal_overlap),
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "path": str(file_path),
                    "chunks_added": len(chunk_ids),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.migrate_existing:
        payload = migrate_registered_repos(
            base_dir=args.base_dir,
            vector_path=args.vector_path,
            embedding_model_name=args.embedding_model,
            vector_backend=args.vector_backend,
            max_files=args.max_files,
        )
    else:
        if not args.repo_path:
            parser.error("repo_path is required unless --migrate-existing is used")
        payload = ingest_repo(
            args.repo_path,
            name=args.name,
            base_dir=args.base_dir,
            glob_pattern=args.glob_pattern,
            max_files=args.max_files,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            exclude=args.exclude,
            vector_path=args.vector_path,
            embedding_model_name=args.embedding_model,
            vector_backend=args.vector_backend,
            purge_existing=not bool(args.no_purge_existing),
        )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if args.migrate_existing:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_human_summary(payload)

    return 0 if str(payload.get("status") or "").lower() == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
