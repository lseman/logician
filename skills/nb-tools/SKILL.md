---
name: nb-tools
description: "Read, edit, run, and convert Jupyter notebooks (.ipynb). Use for inspecting, editing, running, and converting notebooks. File-based sources for adding cells, output extraction, format normalization."
---

# Jupyter Notebook Tools

## Setup

No setup required. Python stdlib only (json, re, sys, uuid, shutil, pathlib).

## Notebook Format

- `source` is always an array of strings, one per line
- Cell IDs are UUIDs in `id` field
- Code cells may have `outputs` array
- JSON is indented with 1 space (by our scripts)

## Quick Reference

All scripts are in `nb-tools/scripts/` relative to this skill directory.

### Inspect

```bash
nb_read.py <notebook>              # summary: cell count, types, source lengths
nb_read.py <notebook> --summary    # same as above (explicit)
nb_read.py <notebook> --cells      # dump full source of every cell
nb_read.py <notebook> --cell 3     # show cell at index 3 only
```

### Edit Cells

Use **file-based sources** for non-trivial content. Scripts read source from files or stdin.

```bash
nb_edit.py <notebook> add-code    --source-file code.py   [--after N] [--before N]
nb_edit.py <notebook> add-code    --source "x = 1"        # short inline only
nb_edit.py <notebook> add-markdown --source-file doc.md    [--append]
nb_edit.py <notebook> delete-cell   N
nb_edit.py <notebook> replace-cell  N --source-file code.py
nb_edit.py <notebook> replace-cell  N --source "short inline"
nb_edit.py <notebook> move-cell     FROM to TO
nb_edit.py <notebook> rename        N new-id-string
nb_edit.py <notebook> clear-output  N        # clear outputs of a code cell
nb_edit.py <notebook> clear-all-outputs          # clear all code cell outputs
```

### Run Cells

```bash
nb_run.py <notebook>                         # run all cells in order
nb_run.py <notebook> --cells 0,2,4           # specific cells (comma-separated)
nb_run.py <notebook> --cells 0-5             # cell range
nb_run.py <notebook> --cells all -i 10       # run all, 10s sleep between cells (interactivity)
nb_run.py <notebook> --cells 3 -t 120        # cell 3 with 120s timeout
nb_run.py <notebook> --quiet                 # no terminal output, just update file
```

Requires either `jupyter_client` (pip) or falls back to running Python directly. If neither kernel is available, cells are skipped with a note.

### Convert

```bash
nb_convert.py <notebook> --to-py              # extract code → stdout
nb_convert.py <notebook> --to-py -o script.py # extract code → file
nb_convert.py <notebook> --to-md              # markdown with code blocks
nb_convert.py --from-py <script.py> --out nb.ipynb [--title Title] [--kernel python3]
```

- `--to-py`: joins code cell sources with `### Cell N\n` separators
- `--to-md`: markdown with `# Cell N` headers, code in fenced blocks, output in comments
- `--from-py`: all code cells, adds markdown title cell on top

### Format

```bash
nb_format.py <notebook> [--check] [--indent 2]
```

Normalizes:
- `source` arrays (ensures proper line splitting)
- Cell metadata (removes unknown keys)
- Sorts output by `output_type` then `execution_count`
- Trailing newline on JSON file
- `--check`: exit 1 if diff exists, no write

## Workflow Patterns

### Add a new analysis cell after cell 3:
```bash
nb_edit.py notebook.ipynb add-code --source-file analysis.py --after 3
```

### Replace cell 5 with updated code:
```bash
nb_edit.py notebook.ipynb replace-cell 5 --source-file updated.py
```

### Run cells 4-8 and inspect:
```bash
nb_run.py notebook.ipynb --cells 4-8
nb_read.py notebook.ipynb --cell 8
```

### Convert for git diff:
```bash
nb_convert.py notebook.ipynb --to-py > notebook_code.py
```

### Format before commit:
```bash
nb_format.py notebook.ipynb
```

### Clear all outputs (smaller file, cleaner for sharing):
```bash
nb_edit.py notebook.ipynb clear-all-outputs
```

## Error Handling

- All write scripts create `.bak` backup before modifying
- Scripts exit non-zero on invalid JSON or file-not-found
- Cell indices are 0-based
- Cell count = len(cells)
