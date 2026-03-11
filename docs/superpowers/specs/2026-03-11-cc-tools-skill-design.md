# Design Spec: `cc_tools` Skill — Claude Code-Parity File Tools

**Date:** 2026-03-11
**Status:** Approved
**Scope:** New skill `skills/coding/cc_tools/` with 6 tools mirroring Claude Code's file tool contracts, plus llama.cpp grammar enforcement for edits.

---

## Problem

The agent's existing exploration and editing tools (`rg_search`, `fd_find`, `read_file`, `edit_file_replace`, `multi_edit`) lack the precise contracts that make Claude Code effective:

- No line-range reads (reads whole files unnecessarily)
- No output mode selection for search (always returns full content)
- No uniqueness enforcement on edits (allows lazy, ambiguous replacements)
- No constrained decoding for edit format (malformed tool calls common with local models)
- Skill prompts don't enforce "read before edit" discipline

---

## Approach

Add a new skill `skills/coding/cc_tools/` with clean CC-parity implementations. Old tools remain as fallback. Once validated, old tools can be retired.

---

## File Structure

```
skills/coding/cc_tools/
  SKILL.md
  scripts/
    tools.py      # 6 tool implementations + __tools__ + __grammars__
    grammar.py    # GBNF grammar constants
```

---

## Tool Contracts

### `cc_glob`
```
Parameters: pattern (glob), path=".", head_limit=0
Returns:    File paths matching pattern, sorted by mtime (newest first)
Behavior:   head_limit=0 means unlimited
```

### `cc_grep`
```
Parameters: pattern (regex), path=".", glob="", type="",
            output_mode="files_with_matches"|"content"|"count",
            context=0, case_insensitive=False,
            head_limit=0, offset=0, multiline=False
Returns:    Depends on output_mode:
            - files_with_matches: one file path per line
            - content: matching lines with optional context
            - count: match count per file
```

### `cc_read`
```
Parameters: file_path, offset=0, limit=2000
Returns:    Lines in cat-n format ("     N\tline content")
            offset: 0-indexed line to start from
            limit: max lines to return
            Lines >2000 chars are truncated
```

### `cc_edit`
```
Parameters: file_path, old_string, new_string, replace_all=False
Contract:
  - old_string not found     → ToolError
  - old_string not unique    → ToolError (unless replace_all=True)
  - replace_all=True         → replaces all occurrences
Requires:   File must have been read in current session
```

### `cc_write`
```
Parameters: file_path, content
Behavior:   Full overwrite; creates parent dirs if needed
Use for:    New files only; use cc_edit for existing files
```

### `cc_multi_edit`
```
Parameters: file_path, edits=[{old_string, new_string, replace_all?}, ...]
Behavior:   Applies edits sequentially; each edit sees result of previous
Use for:    ≥2 edits in same file
```

---

## llama.cpp Grammar Enforcement

`__grammars__` dict exported from `tools.py` registers GBNF grammars for `cc_edit` and `cc_multi_edit`. These enforce valid JSON string escaping and structure.

### Agent Core Changes (minimal)

**`src/tools/registry/loading.py`** — collect grammars on skill load:
```python
if hasattr(module, "__grammars__"):
    self._grammars.update(module.__grammars__)
```

**`src/tools/registry/introspection.py`** — expose lookup:
```python
def get_grammar(self, tool_name: str) -> str | None:
    return self._grammars.get(tool_name)
```

**`src/agent/core.py`** — select grammar before LLM call:
```python
grammar = self.tool_registry.get_grammar(predicted_next_tool)
response = await self.llm.generate(messages, grammar=grammar)
```

---

## Skill Prompt (`SKILL.md`)

Enforces the Claude Code workflow:

1. **Find** — `cc_glob` for file patterns, `cc_grep output_mode=files_with_matches` for content
2. **Read** — `cc_read offset=N limit=50` for targeted reads
3. **Edit** — `cc_edit` for surgical changes; `cc_multi_edit` for ≥2 edits; `cc_write` for new files only

**Hard rules in prompt:**
- Never edit without reading first
- `old_string` must include enough context to be unique
- Use `cc_multi_edit` for multiple changes in one file

---

## What Does NOT Change

- Existing tools (`rg_search`, `fd_find`, `read_file`, `edit_file_replace`, `multi_edit`) remain untouched
- No changes to agent config defaults
- No changes to reasoning/thinking pipeline
- Grammar hook is additive — no grammar = current behavior

---

## Success Criteria

- `cc_edit` rejects ambiguous `old_string` with a clear error
- `cc_read` returns line-numbered output for any offset/limit range
- `cc_grep` with `output_mode=files_with_matches` returns only paths (no noise)
- llama.cpp grammar prevents malformed edit tool calls
- Skill routes correctly on "find", "read", "edit", "fix", "modify" triggers
