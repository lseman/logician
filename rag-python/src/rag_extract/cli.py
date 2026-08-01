"""Subcommand: extract a file using Docling and emit JSON to stdout.

Usage: python -m rag_extract.cli extract <filepath> [options]
       python -m rag_extract.cli extract-from-text <text> --source <name>

Exit codes: 0 = success, 1 = error (JSON emitted on stderr)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any


def _extract_via_docling(filepath: str, doc_id: str | None = None) -> dict[str, Any]:
    """Parse a document with Docling and return structured extraction."""
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import-untyped]
    except ImportError as e:
        _emit_error(f"docling not installed: {e}")
    
    converter = DocumentConverter()
    result = converter.convert(filepath)
    
    # Extract markdown + metadata
    md_text = result.document.export_to_markdown() if hasattr(result.document, 'export_to_markdown') else str(result.document)
    
    chunks = _chunk_text(md_text, doc_id or str(hash(filepath)))
    
    return {
        "id": doc_id or str(hash(filepath)),
        "filename": Path(filepath).name,
        "content": md_text[:64_000],  # cap content size
        "meta": _extract_meta(result),
        "chunks": chunks,
        "extracted_at": __import__("time").time() * 1000,
    }


def _chunk_text(text: str, doc_id: str) -> list[dict[str, Any]]:
    """Split text into ~800 char chunks respecting heading boundaries."""
    MAX_CHUNK = 800
    MIN_CHUNK = 100
    chunks: list[dict[str, Any]] = []
    
    if not text.strip():
        return chunks
    
    # Split by markdown headings to respect structure
    sections = _split_by_headings(text)
    
    for section in sections:
        stripped = section.strip()
        if len(stripped) < MIN_CHUNK:
            continue
        
        if len(stripped) > MAX_CHUNK * 2:
            sub_chunks = _split_by_size(stripped, MAX_CHUNK)
            for i, sub in enumerate(sub_chunks):
                chunks.append(_make_chunk(f"{doc_id}_chunk_{len(chunks)}_{i}", sub.strip(), doc_id))
        else:
            chunks.append(_make_chunk(f"{doc_id}_chunk_{len(chunks)}", stripped, doc_id))
    
    # Fallback: if no headings matched, size-split everything
    if not chunks:
        for i, chunk_text in enumerate(_split_by_size(text, MAX_CHUNK)):
            if chunk_text.strip():
                chunks.append(_make_chunk(f"{doc_id}_chunk_{i}", chunk_text.strip(), doc_id))
    
    return chunks


def _split_by_headings(text: str) -> list[str]:
    """Split markdown text by heading markers (##, ###)."""
    parts = []
    current = []
    for line in text.split("\n"):
        if line.startswith("## ") or line.startswith("### "):
            if current:
                parts.append("\n".join(current))
                current = []
        current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts


def _split_by_size(text: str, max_size: int) -> list[str]:
    """Split text into chunks at word boundaries."""
    chunks: list[str] = []
    remaining = text
    
    while len(remaining) > max_size:
        split_at = min(max_size, len(remaining))
        search_start = max(0, split_at - 100)
        space_idx = remaining.rfind(" ", search_start, split_at + 50)
        
        if space_idx > search_start:
            split_at = space_idx
        elif split_at > 0:
            split_at = max_size
        
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    
    if remaining.strip():
        chunks.append(remaining)
    
    return chunks


def _make_chunk(chunk_id: str, text: str, doc_id: str) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {"source": doc_id, "type": "text"},
        "document_id": doc_id,
    }


def _extract_meta(result: Any) -> dict[str, Any]:
    """Extract metadata from Docling conversion result."""
    meta = {"format": "pdf"}  # default
    
    try:
        doc = result.document
        if hasattr(doc, 'meta') and doc.meta:
            m = doc.meta
            meta["title"] = getattr(m, 'title', None) or None
            authors = getattr(m, 'authors', None)
            meta["author"] = ", ".join(authors) if authors else None
            meta["page_count"] = getattr(m, 'page_count', None) or None
            inp = getattr(m, 'input_format', None)
            meta["format"] = str(inp) if inp else "pdf"
    except Exception:
        pass
    
    return meta


def _emit_error(msg: str) -> None:
    import traceback
    err = {"error": msg}
    try:
        tb = traceback.format_exc()
        if tb.strip():
            err["traceback"] = tb
    except Exception:
        pass
    sys.stderr.write(json.dumps(err))
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG document extraction via Docling")
    subparsers = parser.add_subparsers(dest="command")
    
    # extract <filepath> [--doc-id ID]
    p_extract = subparsers.add_parser("extract", help="Extract from file path")
    p_extract.add_argument("filepath", help="Path to document file")
    p_extract.add_argument("--doc-id", default=None, help="Document ID override")
    
    # extract-from-text <text> --source NAME [--doc-id ID]
    p_text = subparsers.add_parser("extract-from-text", help="Index raw text")
    p_text.add_argument("text", help="Text content to index")
    p_text.add_argument("--source", required=True, help="Source identifier")
    p_text.add_argument("--doc-id", default=None, help="Document ID override")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "extract":
            data = _extract_via_docling(args.filepath, args.doc_id)
        elif args.command == "extract-from-text":
            chunks = [_make_chunk((args.doc_id or "text") + "_raw_0", args.text[:64_000], args.doc_id or "text")]
            data = {
                "id": args.doc_id or "text",
                "filename": args.source,
                "content": args.text[:64_000],
                "meta": {"format": "raw-text"},
                "chunks": chunks,
                "extracted_at": __import__("time").time() * 1000,
            }
        else:
            parser.print_help()
            sys.exit(1)
        
        print(json.dumps(data))
    except Exception as e:
        _emit_error(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
