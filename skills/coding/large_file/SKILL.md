---
name: large_file
description: Use when a file exceeds ~300 lines or when read_file would return more than 8 000 chars — guides strategic outline-first reading, targeted line-range extraction, and safe multi-chunk editing without loading the full file into context.
triggers:
  - "read this large file"
  - "file is too big"
  - "context window"
  - "read lines"
  - "outline first"
anti_triggers:
  - "file is small"
  - "under 200 lines"
preferred_tools:
  - get_file_outline
  - sed_read
  - read_file_smart
  - find_in_file
  - rg_search
---

# Large File Strategy

## Core Rule

> **Never call `read_file` on a file you haven't measured.**
> If you don't know the line count, use `get_file_outline` or `read_file_smart` first.
> Loading a 3 000-line file in full wastes ≈15 000 tokens and crowds out context for actual reasoning.

---

## Decision Tree

```
Is the file > ~300 lines OR unknown size?
 ├─ YES → Outline-first strategy (see below)
 └─ NO  → read_file is fine — proceed normally
```

---

## Phase 1 — Measure and Map

Always start here for unfamiliar or large files.

```
1. get_file_outline(path)
   → reveals: class/function names, line ranges, total line count

2. If outline is insufficient (e.g. plain text, config, data files):
   read_file_smart(path, max_lines=80)
   → returns the first N lines + a structural summary
```

**After Phase 1 you should know:**
- Total line count
- Which sections / symbols are relevant
- Exact line ranges to target in Phase 2

---

## Phase 2 — Targeted Extraction

Read only the ranges you actually need.

| Tool | When to use |
|------|-------------|
| `sed_read(path, start, end)` | You know the exact line range (from outline) |
| `read_file_smart(path, symbol=…)` | You want a specific function/class by name |
| `find_in_file(path, pattern)` | You need to locate where something is defined |
| `rg_search(pattern, path)` | Cross-file or regex search |

**Typical sequence:**
```
get_file_outline("src/agent/core.py")
→ see Agent._score_answer_confidence at lines 2142–2176

sed_read("src/agent/core.py", 2142, 2176)
→ read only those 35 lines
```

**Chunk size guide:**

| Purpose | Max lines to read |
|---------|------------------|
| Understanding a function | 50–100 |
| Understanding a class | 100–200 |
| Understanding a module | Use outline only; read sections |
| Understanding a whole file | Almost never needed; summarize from outline |

---

## Phase 3 — Editing Large Files

**Never load the full file before editing.** Use surgical tools:

```
edit_file_replace(path, old_string, new_string)
  → reads only the context it needs; safe for any file size

apply_unified_diff(path, diff)
  → applies a patch without reading the whole file

apply_edit_block(path, …)
  → targeted block replacement
```

**Anti-pattern:**
```
# DON'T DO THIS
content = read_file("huge_module.py")   # 4 000 lines × 4 chars = 16 000 tokens wasted
# ... edit a 5-line function ...
write_file("huge_module.py", new_content)
```

**Correct pattern:**
```
get_file_outline("huge_module.py")      # find the function's line range
edit_file_replace(                      # replace only the target
  path="huge_module.py",
  old_string="def old_impl():\n    ...",
  new_string="def old_impl():\n    <fix>",
)
```

---

## Multi-Section Reads

When you need several non-contiguous sections from the same large file, **batch the reads**:

```
# Read 3 sections in parallel — do NOT call sed_read three times sequentially
sed_read(path, 100, 150)   # section A
sed_read(path, 800, 850)   # section B
sed_read(path, 2100, 2150) # section C
```

Then consolidate findings before any edits.

---

## Per-Language Notes

| Language | Best outline tool | Symbol lookup |
|----------|------------------|---------------|
| Python | `get_file_outline` → classes/functions | `read_file_smart(symbol="ClassName.method")` |
| Rust | `get_file_outline` → `impl` blocks | `rg_search("fn target_fn", path)` |
| TypeScript/JS | `get_file_outline` → exports | `find_in_file(path, "export function target")` |
| Markdown/docs | `sed_read` in 100-line chunks | `rg_search` for headings |
| JSON/YAML (large) | `jq`/`yq` via `run_shell` | Path-based queries |

---

## Anti-Patterns

| ❌ Don't | ✅ Do instead |
|----------|--------------|
| `read_file` on unknown-size file | `get_file_outline` first |
| Read the whole file to find one function | `read_file_smart(symbol=…)` |
| Load 3 000 lines to make a 5-line edit | `edit_file_replace` directly |
| Read sequentially: section A, then B, then C | Batch all reads in one parallel fan-out |
| Summarize a file by reading it fully | Summarize the outline instead |
| Re-read the same range twice | Cache the content in your working memory |

---

## Token Budget Awareness

- A 1 000-line Python file ≈ 25 000–40 000 tokens
- A typical context window is 100 000–200 000 tokens
- Reading 3 large files fully can consume 30–60% of your budget — leaving little room for reasoning, plans, and tool results

**Rule of thumb:** If you've already read > 2 large files in the current turn, stop and summarize what you've learned before reading more.

---

## Integration

**Next skills after large_file:**
- `coding/edit_block` — surgical replacement after locating the target
- `coding/parallel_dispatch` — batch-read multiple sections simultaneously
- `coding/quality` — verify after edits
