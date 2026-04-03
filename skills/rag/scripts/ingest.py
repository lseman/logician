"""RAG ingestion tools.

Focused on writing data into the RAG index.

Tool inventory
--------------
rag_add_file      -- chunk + ingest one file
rag_add_text      -- ingest an inline text payload
rag_add_dir       -- bulk ingest matching files from a directory
rag_promote_paths -- promote selected files/folders into RAG
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from src.rag_runtime import rag_runtime_settings

if "_safe_json" not in globals():

    def _safe_json(obj: Any) -> str:  # type: ignore[misc]
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json.dumps({"status": "error", "error": repr(obj)})


# ---------------------------------------------------------------------------
# Lazy DocumentDB accessor
# ---------------------------------------------------------------------------


def _get_doc_db():
    """Return a DocumentDB instance, reusing the one on the live agent if present."""
    try:
        from src.db.document import DocumentDB

        settings = rag_runtime_settings()

        return DocumentDB(
            vector_path=settings["vector_path"],
            embedding_model_name=settings["embedding_model_name"],
            vector_backend=settings["vector_backend"],
        )
    except Exception as exc:
        raise RuntimeError(f"Cannot initialise DocumentDB: {exc}") from exc


def _doc_db_from_agent():
    """Try to get DocumentDB from the live agent's memory (avoids a second model load)."""
    try:
        mem = globals().get("ctx") and getattr(globals()["ctx"], "memory", None)
        if mem is not None:
            doc_db = getattr(mem, "_doc_db", None)
            if doc_db is not None:
                return doc_db
            if hasattr(mem, "_ensure_doc_db"):
                mem._ensure_doc_db()
                if mem._doc_db is not None:
                    return mem._doc_db
    except Exception:
        pass
    return _get_doc_db()


def _parse_promote_paths(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            out.extend(part.strip() for part in line.split(",") if part.strip())
        else:
            out.append(line)

    if out:
        return out
    return [text]


def _expand_brace_glob(glob: str) -> list[str]:
    """Expand a single-level brace glob such as **/*.{py,md,txt}."""
    brace_match = re.search(r"\{([^}]+)\}", glob)
    if not brace_match:
        return [glob]

    exts = brace_match.group(1).split(",")
    prefix = glob[: brace_match.start()]
    suffix = glob[brace_match.end() :]
    return [f"{prefix}{ext.strip()}{suffix}" for ext in exts if ext.strip()]


def _parse_exclude_paths(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    seen: set[str] = set()
    excludes: list[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            item = part.strip().replace("\\", "/").strip("/")
            if item.startswith("./"):
                item = item[2:].strip("/")
            if not item or item in seen:
                continue
            seen.add(item)
            excludes.append(item)
    return excludes


def _is_excluded_relative_path(rel_path: Path, exclude_paths: list[str]) -> bool:
    if not exclude_paths:
        return False

    rel_posix = rel_path.as_posix()
    rel_parts = rel_path.parts
    for excluded in exclude_paths:
        if "/" in excluded:
            if rel_posix == excluded or rel_posix.startswith(f"{excluded}/"):
                return True
            continue
        if excluded in rel_parts:
            return True
    return False


def _collect_matching_files(
    root: Path,
    glob_pattern: str,
    max_files: int,
    exclude: str = "",
) -> list[Path]:
    exclude_paths = _parse_exclude_paths(exclude)
    files: list[Path] = []
    for pat in _expand_brace_glob(glob_pattern):
        for fpath in sorted(root.glob(pat)):
            if not fpath.is_file() or fpath in files:
                continue
            try:
                rel_path = fpath.relative_to(root)
            except ValueError:
                continue
            if _is_excluded_relative_path(rel_path, exclude_paths):
                continue
            files.append(fpath)
            if len(files) >= max_files:
                return files
    return files


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
    repo_id: str = "",
    repo_name: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "file_kind": _file_kind_for_path(path),
        "ext": path.suffix.lower(),
    }
    repo_id = str(repo_id or "").strip()
    repo_name = str(repo_name or "").strip()
    repo_root_text = str(repo_root or "").strip()
    if repo_id:
        metadata["repo_id"] = repo_id
    if repo_name:
        metadata["repo_name"] = repo_name
    if repo_root_text:
        try:
            repo_root_path = Path(repo_root_text).expanduser().resolve()
            metadata["repo_root"] = str(repo_root_path)
            try:
                metadata["repo_rel_path"] = str(path.relative_to(repo_root_path))
            except Exception:
                pass
        except Exception:
            metadata["repo_root"] = repo_root_text
    return metadata


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def rag_add_file(
    path: str,
    source_label: str = "",
    chunk_size: int = 400,
    overlap: float = 0.2,
    repo_id: str = "",
    repo_name: str = "",
    repo_root: str = "",
) -> str:
    """Use when: Add a document, source file, or notes file to the agent's long-term RAG memory.

    Triggers: add to rag, index file, ingest document, remember this file, add context.
    Avoid when: The content is a short factual note — use scratch_write instead.
    Inputs:
      path (str, required): Path to the file to ingest.
      source_label (str, optional): Label stored as metadata (default: filename).
      chunk_size (int, optional): Target tokens per chunk (default 400).
      overlap (float, optional): Overlap ratio between chunks, 0.0–0.4 (default 0.2).
    Returns: JSON with number of chunks added and their IDs.
    Side effects: Writes to the RAG vector store on disk.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return _safe_json({"status": "error", "error": f"File not found: {path}"})

        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return _safe_json({"status": "error", "error": "File is empty"})

        label = source_label.strip() or p.name
        doc_db = _doc_db_from_agent()
        ids = doc_db.add_documents(
            texts=[text],
            metadatas=[
                {
                    "source": label,
                    "path": str(p),
                    **_repo_metadata(
                        p,
                        repo_id=repo_id,
                        repo_name=repo_name,
                        repo_root=repo_root,
                    ),
                }
            ],
            chunk_size_tokens=chunk_size,
            chunk_overlap_ratio=overlap,
        )
        return _safe_json(
            {
                "status": "ok",
                "source": label,
                "path": str(p),
                "chunks_added": len(ids),
                "chunk_ids": ids[:10],
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def rag_add_text(
    text: str,
    source: str = "inline",
    chunk_size: int = 400,
    overlap: float = 0.2,
) -> str:
    """Use when: Store a code snippet, documentation block, or knowledge note in the RAG store.

    Triggers: remember this, store knowledge, add to rag, index text, save context.
    Avoid when: You want scratch-pad storage — use scratch_write for temporary notes.
    Inputs:
      text (str, required): The text content to index.
      source (str, optional): Human-readable label for the source (default "inline").
      chunk_size (int, optional): Target tokens per chunk (default 400).
      overlap (float, optional): Overlap ratio (default 0.2).
    Returns: JSON with number of chunks added.
    Side effects: Writes to the RAG vector store.
    """
    try:
        if not text.strip():
            return _safe_json({"status": "error", "error": "Text is empty"})

        doc_db = _doc_db_from_agent()
        doc_id = str(uuid.uuid4())
        ids = doc_db.add_documents(
            texts=[text],
            metadatas=[{"source": source}],
            ids=[doc_id],
            chunk_size_tokens=chunk_size,
            chunk_overlap_ratio=overlap,
        )
        return _safe_json(
            {
                "status": "ok",
                "source": source,
                "chunks_added": len(ids),
                "doc_id": doc_id,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def rag_add_dir(
    directory: str = ".",
    glob: str = "**/*.{py,md,txt,rst}",
    chunk_size: int = 400,
    overlap: float = 0.2,
    max_files: int = 50,
    exclude: str = "",
    repo_id: str = "",
    repo_name: str = "",
    repo_root: str = "",
) -> str:
    """Use when: Index an entire codebase, docs folder, or set of notes in one shot.

    Triggers: index project, add all files, bulk ingest, index codebase, add docs.
    Avoid when: You only need one file — use rag_add_file instead.
    Inputs:
      directory (str, optional): Root directory to scan (default ".").
      glob (str, optional): Glob pattern for files (default "**/*.{py,md,txt,rst}").
      chunk_size (int, optional): Target tokens per chunk (default 400).
      overlap (float, optional): Overlap ratio (default 0.2).
      max_files (int, optional): Maximum number of files to ingest (default 50, hard limit 200).
      exclude (str, optional): Comma/newline-separated subpaths to skip, e.g. "node_modules,dist".
    Returns: JSON with per-file status and total chunks added.
    Side effects: Writes to the RAG vector store.
    """
    try:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})

        max_files = min(max(1, int(max_files)), 200)
        files = _collect_matching_files(
            root=root,
            glob_pattern=glob,
            max_files=max_files,
            exclude=exclude,
        )

        if not files:
            return _safe_json(
                {
                    "status": "ok",
                    "message": "No matching files found",
                    "files_processed": 0,
                    "total_chunks_added": 0,
                    "results": [],
                }
            )

        doc_db = _doc_db_from_agent()
        results = []
        total_chunks = 0

        for fp in files:
            rel = str(fp.relative_to(root))
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    results.append({"file": rel, "status": "skipped", "reason": "empty"})
                    continue

                ids = doc_db.add_documents(
                    texts=[text],
                    metadatas=[
                        {
                            "source": fp.name,
                            "path": str(fp),
                            **_repo_metadata(
                                fp,
                                repo_id=repo_id,
                                repo_name=repo_name,
                                repo_root=repo_root or str(root),
                            ),
                        }
                    ],
                    chunk_size_tokens=chunk_size,
                    chunk_overlap_ratio=overlap,
                )
                total_chunks += len(ids)
                results.append({"file": rel, "status": "ok", "chunks": len(ids)})
            except Exception as exc:
                results.append({"file": rel, "status": "error", "error": str(exc)})

        return _safe_json(
            {
                "status": "ok",
                "files_processed": len(files),
                "total_chunks_added": total_chunks,
                "results": results,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


def rag_promote_paths(
    paths: str,
    recursive: bool = True,
    glob: str = "**/*.{py,md,txt,rst}",
    chunk_size: int = 400,
    overlap: float = 0.2,
    max_files: int = 80,
    exclude: str = "",
    repo_id: str = "",
    repo_name: str = "",
    repo_root: str = "",
) -> str:
    """Use when: You want explicit control over which repo paths become retrievable context.

    Triggers: promote to rag, index selected files, add these paths to context, ingest these folders.
    Avoid when: You only need one file — use rag_add_file.
    Inputs:
      paths (str, required): JSON array or newline/comma-separated list of file/dir paths.
      recursive (bool, optional): Recurse into directories (default True).
      glob (str, optional): Directory file glob (default "**/*.{py,md,txt,rst}").
      chunk_size (int, optional): Target tokens per chunk (default 400).
      overlap (float, optional): Chunk overlap ratio (default 0.2).
      max_files (int, optional): Total file cap across directory promotions (default 80, max 300).
      exclude (str, optional): Comma/newline-separated subpaths to skip during directory promotion.
    Returns: JSON summary with per-path outcomes.
    Side effects: Writes to the RAG vector store.
    """
    try:
        parsed_paths = _parse_promote_paths(paths)
        if not parsed_paths:
            return _safe_json({"status": "error", "error": "No paths provided in 'paths'."})

        budget = max(1, min(int(max_files or 80), 300))
        remaining = budget
        results: list[dict[str, Any]] = []
        total_chunks = 0
        total_files = 0

        for raw in parsed_paths:
            p = Path(raw).expanduser().resolve()
            if not p.exists():
                results.append({"path": str(raw), "status": "error", "error": "Path not found"})
                continue

            if p.is_file():
                if remaining <= 0:
                    results.append(
                        {
                            "path": str(p),
                            "status": "skipped",
                            "reason": "max_files budget exhausted",
                        }
                    )
                    continue

                out = json.loads(
                    rag_add_file(
                        path=str(p),
                        source_label=p.name,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        repo_id=repo_id,
                        repo_name=repo_name,
                        repo_root=repo_root or str(p.parent),
                    )
                )
                chunks_added = int(out.get("chunks_added", 0) or 0)
                status = str(out.get("status", "error"))
                results.append(
                    {
                        "path": str(p),
                        "kind": "file",
                        "status": status,
                        "chunks_added": chunks_added,
                        "error": out.get("error"),
                    }
                )
                if status == "ok":
                    total_chunks += chunks_added
                    total_files += 1
                    remaining -= 1
                continue

            if p.is_dir():
                if remaining <= 0:
                    results.append(
                        {
                            "path": str(p),
                            "status": "skipped",
                            "reason": "max_files budget exhausted",
                        }
                    )
                    continue

                dir_glob = glob if recursive else glob.replace("**/", "")
                out = json.loads(
                    rag_add_dir(
                        directory=str(p),
                        glob=dir_glob,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        max_files=remaining,
                        exclude=exclude,
                        repo_id=repo_id,
                        repo_name=repo_name,
                        repo_root=repo_root or str(p),
                    )
                )
                files_processed = int(out.get("files_processed", 0) or 0)
                chunks_added = int(out.get("total_chunks_added", 0) or 0)
                status = str(out.get("status", "error"))
                results.append(
                    {
                        "path": str(p),
                        "kind": "directory",
                        "status": status,
                        "files_processed": files_processed,
                        "chunks_added": chunks_added,
                        "error": out.get("error"),
                    }
                )
                if status == "ok":
                    total_chunks += chunks_added
                    total_files += files_processed
                    remaining = max(0, remaining - files_processed)
                continue

            results.append(
                {
                    "path": str(p),
                    "status": "error",
                    "error": "Unsupported path type",
                }
            )

        success_count = sum(1 for item in results if item.get("status") == "ok")
        overall = "ok" if success_count > 0 else "error"
        return _safe_json(
            {
                "status": overall,
                "selected_paths": parsed_paths,
                "items_total": len(results),
                "items_ok": success_count,
                "files_processed": total_files,
                "total_chunks_added": total_chunks,
                "max_files_budget": budget,
                "max_files_remaining": remaining,
                "results": results,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


__tools__ = [rag_add_file, rag_add_text, rag_add_dir, rag_promote_paths]
