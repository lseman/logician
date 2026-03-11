---
name: cc_tools
description: >
  Precise file exploration and editing for coding tasks. Use for finding files,
  reading code with line ranges, and making surgical targeted edits. Preferred
  over explore/file_ops for any new coding work.
aliases:
  - find files
  - read file
  - edit file
  - search code
  - navigate codebase
  - glob
  - grep
triggers:
  - look for
  - find where
  - read the
  - edit
  - change
  - fix
  - modify
  - search for
preferred_tools:
  - cc_glob
  - cc_grep
  - cc_read
  - cc_edit
  - cc_multi_edit
when_not_to_use:
  - shell execution (use shell skill)
  - git operations (use git skill)
  - running tests (use quality skill)
next_skills:
  - shell
  - git
  - quality
---

## Workflow

Always follow this sequence — it mirrors how Claude Code works:

1. **Find** — use `cc_glob` for file patterns (`**/*.py`), `cc_grep output_mode=files_with_matches` to find which files contain a pattern
2. **Read** — use `cc_read offset=N limit=50` to read only the relevant lines; never read an entire large file
3. **Edit** — use `cc_edit` for one surgical change; `cc_multi_edit` for ≥2 changes in the same file; `cc_write` only for new files

## Rules

- Never edit a file you have not read in the current session
- `cc_edit` requires `old_string` to be unique in the file — include enough surrounding lines to make it unique
- For multiple edits in one file, always use `cc_multi_edit` (batched = fewer LLM round-trips)
- Use `cc_grep output_mode=files_with_matches` first; switch to `content` only when you need to see matching lines
- Use `cc_read` with `offset` and `limit` when you know approximately where in the file the relevant code lives

## Tool Quick Reference

| Tool | Use for |
|------|---------|
| `cc_glob` | Find files by name pattern (e.g. `**/*.py`, `src/**/config.*`) |
| `cc_grep` | Find files or lines matching a regex (content search) |
| `cc_read` | Read a file, optionally starting at line N for M lines |
| `cc_edit` | Replace one exact string in a file (surgical edit) |
| `cc_write` | Write a new file from scratch (or full overwrite) |
| `cc_multi_edit` | Apply multiple replacements to one file in a single call |
