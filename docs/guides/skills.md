---
title: Skills
description: Create, discover, and invoke SKILL.md capabilities.
---

# Skills

A skill is a directory containing `SKILL.md`. Its frontmatter describes when it applies; its Markdown body contains the instructions loaded when the skill activates or the user invokes its slash command.

## Where skills load from

Logician recursively discovers skills from:

1. enabled plugins' `skills/` directories;
2. `~/.agents/skills/` for user-wide skills;
3. `skills/` and `.agents/skills/` in trusted project ancestors.

Real paths are deduplicated. When stable IDs collide, the first loaded skill wins and Logician reports a diagnostic.

## Directory and command names

The stable ID retains the path below its skill root, but the slash command uses only the skill directory's final segment:

```text
~/.agents/skills/cpp/cpp-router/SKILL.md
stable ID: cpp/cpp-router
command:   /cpp-router
```

Use a prefixed leaf directory when the command itself needs a namespace, such as `gsd/gsd-plan-phase/SKILL.md` → `/gsd-plan-phase`.

## SKILL.md format

```markdown
---
name: file-operations
description: Safely inspect and modify local files.
triggers:
  - edit this file
  - update the implementation
allowed-tools:
  - read_file
  - edit_file
argument-hint: "[path]"
---

# File operations

Read a file before editing it. Preserve its existing line endings and encoding.
```

`description` is required. `name` is a human-facing display name; the stable ID and slash command come from the directory path. Useful optional fields include `aliases`, `triggers`, `example_queries`, `when_not_to_use`, `next_skills`, `preferred_tools`, `model`, and `disable-model-invocation`.

## Activation

Logician scores the current request against names, aliases, descriptions, triggers, and examples. A bounded set of relevant skills is injected for that turn. Skills with `disable-model-invocation: true` remain available as slash commands but are excluded from automatic activation.

You can invoke a skill directly with `/<directory-name> [arguments]`. Restart or use `/reload` after adding a skill.

## Resources

Put supporting material in `references/` and executable helpers in `scripts/` below the skill directory. Relative paths in `SKILL.md` resolve against that directory. Keep the main file focused and route to supporting files only when needed.

## Diagnose loading

Run `logician doctor --json` to inspect skill roots and diagnostics. Common failures are a missing description, invalid directory characters, malformed YAML, ignored files, or an untrusted project.
