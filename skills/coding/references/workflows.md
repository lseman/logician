# Coding Workflows

Use this reference when a coding task spans more than one leaf skill and you need to pick the next step cleanly.

## Default Loop

The default coding loop is:

1. `explore`
2. `file_ops` or `edit_block` or `multi_edit` or `search_replace`
3. `quality`
4. `git`

Use this as the baseline unless the task is obviously a pure execution or pure documentation lookup task.

## Start Skill Selection

Start with `explore` when:

- the codebase is unfamiliar
- the user names a symbol but not the file
- you need references, outlines, or project structure first

Within `explore`, prefer this order:

1. `get_project_map` for a repo or package overview
2. `find_symbol` or `rg_search` to narrow to candidate files
3. `get_file_outline` before reading long source files

Use `get_project_map` as the first pass for unfamiliar repos because it now covers important source, config, and documentation files instead of only Python modules.

Start with `file_ops` when:

- the exact file is already known
- the task is a direct read or small write
- there is no real discovery step

Start with `shell` when:

- the user explicitly asks to run commands
- the next useful fact comes from execution rather than static inspection
- you need a dev server, test run, or short Python execution

Start with `web` when:

- the task depends on external documentation or an exact URL
- the repo does not contain the needed reference material

## Edit Skill Selection

Choose `edit_block` when:

- the change is local and anchorable
- you want the smallest possible modification

Choose `multi_edit` when:

- several files need coordinated changes
- imports, call sites, and definitions must move together

Choose `search_replace` when:

- the change is pattern-driven
- scope can be described as files plus search criteria
- regex or repeated textual transforms are the main operation

Choose `patch` when:

- the intended edit already exists as a diff
- previewing or applying unified diff is the cleanest path

## Validation Strategy

Use `quality` immediately after meaningful edits.

Prefer:

1. targeted tests or linters for touched files
2. broader checks only when the narrow checks pass or are insufficient

If execution problems are unclear, hand off to `shell`.

## Repo State Strategy

Use `git` when:

- you need to inspect current changes before editing
- you need a checkpoint before a risky refactor
- you need history or blame to recover intent
- you are done and want a reviewable diff or commit

Do not start with `git` when the task is purely local file inspection.

## Common Sequences

`explore -> edit_block -> quality`

Use for a small localized fix in unfamiliar code.

`explore -> multi_edit -> quality -> git`

Use for coordinated refactors across several files.

`web -> file_ops -> shell -> quality`

Use when external docs inform a concrete implementation and execution confirms it.

`shell -> explore -> edit_block -> shell`

Use when a failing command or test identifies the area before the code change.
