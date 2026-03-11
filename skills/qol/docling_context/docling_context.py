from __future__ import annotations

import json
import mimetypes
import uuid
from pathlib import Path
from skills.coding.bootstrap.runtime_access import tool

__skill__ = {
    "name": "Docling Context",
    "description": "Use for rich document extraction from PDFs, DOCX, PPTX, and other structured files that need layout-aware conversion before context ingestion or RAG indexing.",
    "aliases": ["pdf extract", "document parse", "docling ingest", "structured document"],
    "triggers": [
        "extract this PDF",
        "parse this document",
        "convert this paper to text",
        "ingest this DOCX",
        "extract tables from this PDF",
    ],
    "preferred_tools": ["docling_add_file", "docling_add_dir"],
    "example_queries": [
        "extract this research paper PDF into markdown",
        "parse the tables from this DOCX report",
        "convert this slide deck to text for RAG ingestion",
    ],
    "when_not_to_use": [
        "content is plain text or already in markdown",
        "a simple HTML page",
        "the file is an image without embedded text",
    ],
    "next_skills": ["rag", "coding/explore"],
    "workflow": [
        "Convert structured documents before indexing or long-context use.",
        "Prefer section-aware chunking for long documents.",
        "Pipe extracted content into RAG rather than dumping giant blobs into context.",
    ],
}

if "_safe_json" not in globals():

    def _safe_json(obj):
        try:
            return json.dumps(obj)
        except Exception as exc:
            return json.dumps({"status": "error", "error": str(exc)})


if "_try_parse_json" not in globals():

    def _try_parse_json(text):
        if isinstance(text, dict):
            return text
        return json.loads(text)


def _get_doc_db():
    from src.db.document import DocumentDB

    return DocumentDB()


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


def _read_with_docling(path: Path) -> str:
    """Extract text/markdown from a document using Docling.

    This supports PDFs, Office docs, images, and text-like formats as available
    in the installed Docling build.
    """
    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:
        raise RuntimeError(
            "docling is not available. Install with: pip install docling"
        ) from exc

    converter = DocumentConverter()
    result = converter.convert(str(path))

    doc = getattr(result, "document", None)
    if doc is None:
        raise RuntimeError("Docling conversion returned no document output")

    if hasattr(doc, "export_to_markdown"):
        text = doc.export_to_markdown() or ""
    elif hasattr(doc, "export_to_text"):
        text = doc.export_to_text() or ""
    else:
        text = str(doc)

    return text.strip()


@tool
def docling_add_file(
    path: str,
    source_label: str = "",
    chunk_size: int = 400,
    overlap: float = 0.2,
) -> str:
    """Use when: Add a local document to RAG context using Docling parsing.

    Triggers: upload file, ingest pdf, add document to context, parse doc with docling.
    Avoid when: You already have plain text content; use rag_add_text instead.
    Inputs:
      path (str, required): Path to a local file.
      source_label (str, optional): Label stored as document source metadata.
      chunk_size (int, optional): Chunk size in pseudo-tokens (default 400).
      overlap (float, optional): Chunk overlap ratio 0.0-0.4 (default 0.2).
    Returns: JSON with ingestion status and chunk count.
    Side effects: Writes chunks to the persistent RAG vector store.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return _safe_json({"status": "error", "error": f"File not found: {path}"})

        text = _read_with_docling(p)
        if not text:
            return _safe_json(
                {
                    "status": "error",
                    "error": "Docling produced empty text for this file",
                }
            )

        label = source_label.strip() or p.name
        mime, _ = mimetypes.guess_type(str(p))
        doc_db = _get_doc_db()

        doc_id = str(uuid.uuid4())
        ids = doc_db.add_documents(
            texts=[text],
            metadatas=[
                {
                    "source": label,
                    "path": str(p),
                    "mime": mime or "unknown",
                    "parser": "docling",
                }
            ],
            ids=[doc_id],
            chunk_size_tokens=chunk_size,
            chunk_overlap_ratio=overlap,
        )

        return _safe_json(
            {
                "status": "ok",
                "path": str(p),
                "source": label,
                "doc_id": doc_id,
                "chunks_added": len(ids),
                "chunk_ids": ids[:10],
                "char_count": len(text),
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@tool
def docling_add_dir(
    directory: str = ".",
    glob: str = "**/*",
    max_files: int = 25,
    chunk_size: int = 400,
    overlap: float = 0.2,
    exclude: str = "",
) -> str:
    """Use when: Add a directory of documents to RAG context using Docling.

    Triggers: upload folder, add docs directory, bulk document ingestion.
    Avoid when: You only need one file; use docling_add_file.
    Inputs:
      directory (str, optional): Root directory to scan.
      glob (str, optional): Glob pattern for candidate files.
      max_files (int, optional): Max files to ingest (hard cap 100).
      chunk_size (int, optional): Chunk size in pseudo-tokens.
      overlap (float, optional): Chunk overlap ratio.
      exclude (str, optional): Comma/newline-separated subpaths to skip, e.g. "node_modules,dist".
    Returns: JSON with per-file status and total chunks added.
    Side effects: Writes chunks to the persistent RAG vector store.
    """
    try:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            return _safe_json(
                {"status": "error", "error": f"Not a directory: {directory}"}
            )

        max_files = max(1, min(int(max_files), 100))
        exclude_paths = _parse_exclude_paths(exclude)
        files = []
        for path in sorted(root.glob(glob)):
            if not path.is_file():
                continue
            try:
                rel_path = path.relative_to(root)
            except ValueError:
                continue
            if _is_excluded_relative_path(rel_path, exclude_paths):
                continue
            files.append(path)
            if len(files) >= max_files:
                break
        if not files:
            return _safe_json(
                {
                    "status": "ok",
                    "message": "No matching files found",
                    "files_processed": 0,
                    "total_chunks_added": 0,
                }
            )

        total_chunks = 0
        results = []
        for p in files:
            out = docling_add_file(
                path=str(p),
                source_label=p.name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            try:
                parsed = _try_parse_json(out)
            except Exception:
                parsed = {"status": "error", "error": "Invalid tool output"}

            rec = {
                "file": str(p.relative_to(root)),
                "status": parsed.get("status", "error"),
            }
            if parsed.get("status") == "ok":
                chunks = int(parsed.get("chunks_added", 0))
                total_chunks += chunks
                rec["chunks"] = chunks
            else:
                rec["error"] = parsed.get("error", "unknown error")
            results.append(rec)

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


__all__ = ["docling_add_file", "docling_add_dir"]


__tools__ = [docling_add_file, docling_add_dir]
