# Coding Skills Layout

This folder is now a container for numbered leaf skills.

## Structure

- `bootstrap/`: shared runtime helpers loaded before the leaf skills.
- `file_ops/` through `search_replace/`: leaf skills with Python modules and tool metadata co-located in the module docstrings.

## Leaf skills

- `file_ops`: file reads, writes, listing, and single-file edits.
- `multi_edit`: coordinated changes across multiple files.
- `web`: web and documentation fetching for coding tasks.
- `shell`: shell commands, Python execution, and process control.
- `git`: git state, history, diffs, and commits.
- `quality`: tests, linting, formatting, and quality gates.
- `repl`: short runtime experiments.
- `patch`: patch application workflows.
- `edit_block`: precise block-level edits.
- `explore`: project mapping, search, and structural inspection.
- `search_replace`: search/replace and refactor-style text transforms.

## Adjacent domains

- SVG generation lives in `skills/svg/svg_viz/`.
- RAG ingestion and retrieval lives in `skills/rag/`.
- Keep `coding` focused on editing, execution, exploration, quality, and git workflows.
