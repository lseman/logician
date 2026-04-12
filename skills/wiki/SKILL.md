---
name: Wiki Operations
description: Use for concrete operations on the markdown wiki corpus: build, search, ingest raw notes or repositories, update source notes, file outputs, and run health/lint checks.
aliases:
  - wiki ops
  - wiki maintenance
  - wiki compiler
triggers:
  - build wiki
  - rebuild wiki
  - search wiki
  - add wiki document
  - update wiki document
  - add raw note
  - file wiki output
  - wiki health
  - wiki lint
preferred_tools:
  - python
  - filesystem
example_queries:
  - rebuild the wiki from source
  - search wiki for deployment timeout
  - add a new page to the wiki
  - ingest this note into raw and rebuild
  - file this answer into wiki outputs
  - check whether wiki.md is stale
when_not_to_use:
  - vector or embedding retrieval
  - generic repo search unrelated to wiki
---

## Role

This is the operational playbook for maintaining the local markdown knowledge
base stored under `./wiki/`.

Use it for concrete maintenance tasks on the corpus and generated workspace.

## Main Operations

Use `skills/wiki/scripts/wiki_ops.py` for these workflows:

- `wiki_build()` rebuilds `wiki.md` and the compiled workspace from source notes
- `wiki_list()` lists normalized source notes available for compilation
- `wiki_list_raw()` inspects raw collected artifacts
- `wiki_search()` searches structured markdown content in the generated wiki
- `wiki_add_document()` creates a source note and optionally rebuilds
- `wiki_add_raw_document()` creates a raw artifact as text
- `wiki_add_file()` copies a file into `source`
- `wiki_add_raw_file()` copies an artifact into `raw`
- `wiki_ingest_raw()` promotes an existing raw artifact into a maintained source-note scaffold, can update related notes, and suggests follow-up pages to create or update
- `wiki_ingest_repo()` snapshots a local repository, or clones a git remote into `raw/` first, then creates a maintained repo overview note and suggests follow-up pages such as architecture or testing
- `wiki_search_repo()` searches the ingested repository checkout directly for code/text matches when the summary note is too coarse for a code-level question
- `wiki_list_suggestions()` reads the suggested follow-up pages embedded in a source note
- `wiki_promote_suggestion()` promotes one suggested follow-up page into a concrete wiki edit
- `wiki_update_document()` edits an existing source note and optionally rebuilds
- `wiki_get_document()` fetches one indexed document by path or id
- `wiki_list_sources()` lists what is currently indexed in `wiki.md`
- `wiki_read_source_note()` reads a source note directly from `source/`, even if the wiki has not been rebuilt yet
- `wiki_write_output()` files a derived result back into `dist/outputs/`
- `wiki_health()` checks for stale, missing, or orphaned generated content
- `wiki_lint()` writes structural quality findings for the wiki graph
- `wiki_verify()` combines health, lint, and optional sample search checks

## Working Model

Follow this loop:

1. Put original material in `raw/` or normalized markdown in `source/`.
2. When starting from a raw artifact, prefer promoting it into a maintained note scaffold before deeper editing. When starting from code, prefer `wiki_ingest_repo()` so the raw layer captures a structured repository snapshot instead of a loose file dump.
3. Rebuild the wiki so `wiki.md` and `dist/` stay in sync.
3a. Read `dist/index.md` first to orient yourself on the current structure and
   find the most relevant pages before searching or answering.
4. Search and inspect the generated workspace instead of treating it like a
   black-box vector index.
5. File useful generated outputs back into the workspace so future queries build
   on earlier work.
6. Use the schema and append-only log to keep future sessions consistent.

## Notes

- Prefer rebuilding the whole wiki after source edits; the corpus is expected to
  stay small enough that full regeneration remains fast.
- Keep generated files machine-maintained; when facts change, edit the source
  note or raw artifact rather than hand-editing compiled pages.
- `wiki/index.md` is the first retrieval/orientation artifact; read it before
  deeper page inspection.
- Keep `dist/log.md` chronological and append-only; use it to record ingests,
  queries, rebuilds, and maintenance actions.
- Never wait for the user to paste the contents of a wiki note that already exists locally.
  Use `wiki_read_source_note()` for `source/` paths, `wiki_get_document()` for compiled indexed notes,
  and `wiki_search_repo()` for code-level repository questions.
