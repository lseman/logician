---
title: Skills
description: SKILL.md-driven capabilities, loading, triggers, and writing custom skills.
---

# Skills

Skills are `SKILL.md` files that inject specialized instructions into the agent's system prompt when triggered by task keywords.

## Structure

Each skill lives in a directory with a `SKILL.md` file:

```
skills/
├── coding/
│   └── file_ops/
│       └── SKILL.md
└── testing/
    └── write_tests/
        └── SKILL.md
```

## SKILL.md format

```markdown
<!-- name: coding/file_ops -->
<!-- displayName: File Operations -->
<!-- description: Safe file reading, writing, and editing -->
<!-- triggers: read write edit file -->

# File Operations

When working with files, always:

1. Read before editing
2. Use exact text matching for edits
3. Preserve CRLF line endings
4. Handle BOM correctly
```

## Loading order

Skills are loaded from these locations (priority order):

1. **Project skills** — `skills/` in the project root
2. **User skills** — `~/.logician/skills/`
3. **Global skills** — installed via package manager

## Triggers

Skills activate automatically when the user prompt matches trigger keywords:

```yaml
# In SKILL.md frontmatter
triggers:
  - read
  - write
  - edit
  - file
```

## Writing custom skills

1. Create a directory: `mkdir -p skills/my-skill`
2. Create `SKILL.md` with frontmatter and instructions
3. Restart the agent — the skill loads automatically

## Skill diagnostics

The agent reports skill loading status:

```
[SKILL] Loaded: coding/file_ops (triggers: read, write, edit, file)
[SKILL] Loaded: testing/write_tests (triggers: test, spec)
[WARN] Skill collision: my-skill overrides project/my-skill
```
