---
name: RAG
description: Use for document ingestion, retrieval, and retrieval-quality tuning in the local RAG store.
aliases:
  - retrieval augmented generation
  - retrieval augmented
  - vector retrieval
  - semantic search
triggers:
  - ingest files into rag
  - search indexed documents
  - inspect rag coverage
  - tune retrieval latency and quality
preferred_tools:
  - rag_search
  - rag_add_file
  - rag_add_dir
  - rag_list
  - rag_tuning_status
example_queries:
  - index this folder into rag
  - search the rag store for sqlite pragmas
  - show what documents are already indexed
  - make rag retrieval faster
when_not_to_use:
  - the user only needs a one-off file read and does not want to persist it in RAG
next_skills:
  - docling_context
---

## Scope

This skill owns long-context document memory:

- ingest files or inline text into the local vector store
- retrieve chunks directly for debugging and inspection
- inspect corpus coverage and indexed sources
- tune retrieval knobs such as reranking and ANN search breadth

## Playbooks

1. **Ingest content**
Use `rag_add_file`, `rag_add_text`, `rag_add_dir`, or `rag_promote_paths` when the user wants content persisted into long-term retrieval memory.

2. **Inspect retrieval**
Use `rag_search` and `rag_list` to verify what the agent can currently retrieve and whether ingestion worked.

3. **Tune runtime quality**
Use `rag_tuning_status`, `rag_apply_profile`, and `rag_benchmark` when the user asks for faster retrieval, better relevance, or benchmarking.

Avoid when: the request is just to read a file once without adding it to RAG.
