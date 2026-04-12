# Schema

> The canonical maintainer schema lives outside the compiled vault and tells the LLM how to ingest, query, file outputs, and lint the wiki.

- Canonical schema: [AGENTS.md](../AGENTS.md)

## Operating Model

- Raw sources are immutable inputs under `raw/`.
- `source/` holds LLM-maintained source notes and summaries.
- `dist/` is the compiled browsing workspace for Obsidian and query-time navigation.
- Useful answers should be filed back into `dist/outputs/` so they compound over time.
- `index.md` is the content-oriented table of contents; `log.md` is the chronological record.

## Status

- Expected schema file is missing at `/data/dev/logician/wiki/AGENTS.md`.
