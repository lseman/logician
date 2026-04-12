---
name: Docling Context
description: Use for rich document extraction from PDFs, DOCX, PPTX, and other structured files before ingestion and retrieval.
aliases:
  - pdf extract
  - document parse
  - docling ingest
triggers:
  - extract this PDF
  - parse this document
  - convert this paper to text
preferred_tools:
  - docling_add_file
  - docling_add_dir
example_queries:
  - ingest a PDF into RAG after layout-aware extraction
  - parse a DOCX research report into markdown
  - extract tables from a scanned slide deck
when_not_to_use:
  - content is plain text or already markdown
  - the document is a simple HTML page that does not need layout parsing
  - you only need to process a one-line text note
next_skills:
  - rag
  - coding/explore
workflow:
  - Convert structured documents before indexing or retrieval.
  - Prefer section-aware extraction for PDFs and DOCX.
  - Send extracted text into RAG instead of large verbatim blobs.
---

## Role

This skill handles document extraction and conversion using Docling. Use it when the task requires layout-aware parsing of PDFs, Office documents, images, or other structured file formats prior to RAG ingestion, summarization, or downstream search.

## Tools

- `docling_add_file(path, source_label, chunk_size, overlap)`
- `docling_add_dir(directory, glob, max_files, chunk_size, overlap, exclude)`

## Implementation

The executable code lives in `skills/qol/docling_context/scripts/docling_context.py`.
