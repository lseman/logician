"""RAG document management tools.

These tools let the agent and users populate (and inspect) the RAG vector store
so that `get_rag_context()` actually returns useful results.

Tool inventory
--------------
rag_add_file   -- chunk and ingest a file into the RAG store
rag_add_text   -- ingest a raw text string directly
rag_add_dir    -- recursively ingest all matching files in a directory
rag_promote_paths -- explicitly promote selected files/folders to retrievable context
rag_list       -- show what sources are currently indexed
rag_search     -- manually query the RAG store (useful for debugging)
"""

from __future__ import annotations

if "llm" not in globals():

    class _NoOpLLM:
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()

if "_safe_json" not in globals():

    def _safe_json(obj: Any) -> str:  # type: ignore[misc]
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json.dumps({"status": "error", "error": repr(obj)})

import uuid
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lazy DocumentDB accessor
# ---------------------------------------------------------------------------


def _get_doc_db():
    """Return a DocumentDB instance, reusing the one on the live agent if present."""
    # When running inside the agent, `ctx` may expose the Memory object.
    try:
        from src.db.document import DocumentDB

        return DocumentDB()
    except Exception as exc:
        raise RuntimeError(f"Cannot initialise DocumentDB: {exc}") from exc


def _doc_db_from_agent():
    """Try to get DocumentDB from the live agent's memory (avoids a second model load)."""
    try:
        # The ToolRegistry injects `ctx` into execution_globals; the agent sets ctx.memory.
        mem = globals().get("ctx") and getattr(globals()["ctx"], "memory", None)
        if mem is not None:
            doc_db = getattr(mem, "_doc_db", None)
            if doc_db is not None:
                return doc_db
            # Force materialise it (lazy_rag=True by default)
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


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@llm.tool(
    description=(
        "Ingest a file into the RAG vector store so the agent can retrieve it as context. "
        "Supports plain text, Markdown, Python, and any UTF-8 readable format. "
        "The file is chunked automatically."
    )
)
def rag_add_file(
    path: str,
    source_label: str = "",
    chunk_size: int = 400,
    overlap: float = 0.2,
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
            metadatas=[{"source": label, "path": str(p)}],
            chunk_size_tokens=chunk_size,
            chunk_overlap_ratio=overlap,
        )
        return _safe_json(
            {
                "status": "ok",
                "source": label,
                "path": str(p),
                "chunks_added": len(ids),
                "chunk_ids": ids[:10],  # show first 10
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(
    description=(
        "Ingest a raw text string into the RAG vector store. "
        "Useful for adding notes, summaries, API docs, or any text that isn't a file."
    )
)
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


@llm.tool(
    description=(
        "Recursively ingest all matching files in a directory into the RAG vector store. "
        "Use to bulk-index a project, docs folder, or notes directory."
    )
)
def rag_add_dir(
    directory: str = ".",
    glob: str = "**/*.{py,md,txt,rst}",
    chunk_size: int = 400,
    overlap: float = 0.2,
    max_files: int = 50,
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
    Returns: JSON with per-file status and total chunks added.
    Side effects: Writes to the RAG vector store.
    """
    try:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            return _safe_json(
                {"status": "error", "error": f"Not a directory: {directory}"}
            )

        max_files = min(int(max_files), 200)

        # Expand brace patterns like {py,md,txt} into multiple globs
        import re as _re

        brace_match = _re.search(r"\{([^}]+)\}", glob)
        if brace_match:
            exts = brace_match.group(1).split(",")
            prefix = glob[: brace_match.start()]
            suffix = glob[brace_match.end() :]
            patterns = [f"{prefix}{ext.strip()}{suffix}" for ext in exts]
        else:
            patterns = [glob]

        files: list[Path] = []
        for pat in patterns:
            for f in sorted(root.glob(pat)):
                if f.is_file() and f not in files:
                    files.append(f)
        files = files[:max_files]

        if not files:
            return _safe_json(
                {
                    "status": "ok",
                    "message": "No matching files found",
                    "files_processed": 0,
                }
            )

        doc_db = _doc_db_from_agent()
        results = []
        total_chunks = 0
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    results.append(
                        {
                            "file": str(fp.relative_to(root)),
                            "status": "skipped",
                            "reason": "empty",
                        }
                    )
                    continue
                ids = doc_db.add_documents(
                    texts=[text],
                    metadatas=[{"source": fp.name, "path": str(fp)}],
                    chunk_size_tokens=chunk_size,
                    chunk_overlap_ratio=overlap,
                )
                total_chunks += len(ids)
                results.append(
                    {
                        "file": str(fp.relative_to(root)),
                        "status": "ok",
                        "chunks": len(ids),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "file": str(fp.relative_to(root)),
                        "status": "error",
                        "error": str(e),
                    }
                )

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


@llm.tool(
    description=(
        "Promote selected files/folders into RAG in one command. "
        "Accepts a JSON list or newline/comma-separated paths."
    )
)
def rag_promote_paths(
    paths: str,
    recursive: bool = True,
    glob: str = "**/*.{py,md,txt,rst}",
    chunk_size: int = 400,
    overlap: float = 0.2,
    max_files: int = 80,
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
    Returns: JSON summary with per-path outcomes.
    Side effects: Writes to the RAG vector store.
    """
    try:
        parsed_paths = _parse_promote_paths(paths)
        if not parsed_paths:
            return _safe_json(
                {"status": "error", "error": "No paths provided in 'paths'."}
            )

        budget = max(1, min(int(max_files or 80), 300))
        remaining = budget
        results: list[dict[str, Any]] = []
        total_chunks = 0
        total_files = 0

        for raw in parsed_paths:
            p = Path(raw).expanduser().resolve()
            if not p.exists():
                results.append(
                    {"path": str(raw), "status": "error", "error": "Path not found"}
                )
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
                dir_glob = glob
                if not recursive:
                    dir_glob = glob.replace("**/", "")
                out = json.loads(
                    rag_add_dir(
                        directory=str(p),
                        glob=dir_glob,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        max_files=remaining,
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


@llm.tool(
    description="Query the RAG vector store directly. Useful for debugging what the agent will retrieve."
)
def rag_search(query: str, top_k: int = 5) -> str:
    """Use when: Inspect what RAG context the agent would retrieve for a given query.

    Triggers: what's in rag, search memory, check rag, debug retrieval, what do you know about.
    Inputs:
      query (str, required): The search query.
      top_k (int, optional): Number of results to return (default 5).
    Returns: JSON with matching chunks, sources, and similarity distances.
    Side effects: Read-only.
    """
    try:
        doc_db = _doc_db_from_agent()
        results = doc_db.query(query, n_results=min(int(top_k), 20))
        if not results:
            return _safe_json(
                {
                    "status": "ok",
                    "query": query,
                    "results": [],
                    "message": "No matches found — RAG store may be empty.",
                }
            )

        hits = [
            {
                "source": r["metadata"].get("source", "unknown"),
                "distance": round(r["distance"], 4),
                "chunk": r["metadata"].get("chunk", 0),
                "preview": r["content"][:300]
                + ("…" if len(r["content"]) > 300 else ""),
            }
            for r in results
        ]
        return _safe_json(
            {"status": "ok", "query": query, "count": len(hits), "results": hits}
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})
