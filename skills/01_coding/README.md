# Coding Skills Layout

This folder contains coding-focused tools loaded by the local skill runtime.

## Current modules

- `00_bootstrap.py`: shared runtime helpers (`_run_cmd`, `_coding_config`).
- `10_file_ops.py`: read/write/list/edit single files.
- `12_multi_edit.py`, `60_patch.py`, `65_edit_block.py`, `95_search_replace.py`: multi-file and patch workflows.
- `20_shell.py`: shell/Python execution and process control.
- `30_git.py`: git status/diff/log/commit/checkpoint helpers.
- `40_quality.py`: pytest/ruff/mypy and quality gates.
- `50_repl.py`: in-process REPL helpers.
- `70_rag.py`, `15_web.py`: external context/doc ingestion.
- `90_explore.py`: structural code exploration/search.

## Moved domains

- SVG visualization/generation tools moved to `skills/04_svg/10_svg_viz.py`.
- Keep `01_coding` focused on filesystem, shell, editing, quality, and git workflows.

## Reorganization guardrails

- Keep public tool names stable to avoid prompt/routing regressions.
- Keep skill IDs stable (`shell`, `quality`, `explore`, etc.) unless migration is explicit.
- Move shared helpers into internal modules only when imports are explicit and tested.
- Prefer extracting private helpers before splitting tool entrypoints.

## Phase 2 split targets

- `90_explore.py`: split AST outline vs grep/find utilities.
- `40_quality.py`: split command runners vs parsers/gates.
- `95_search_replace.py`: split chunked-read vs regex/replace engine.
