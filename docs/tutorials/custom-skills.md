---
title: Custom Skills
description: Create and test a project-local SKILL.md capability.
---

# Custom Skills

This tutorial creates a project skill with a direct slash command and automatic activation hints.

## 1. Create the directory

```bash
mkdir -p skills/project-code-review
```

The final directory segment becomes the command: `/project-code-review`.

## 2. Write SKILL.md

```markdown
---
name: Project Code Review
description: Review this project's code for correctness, security, and maintainability.
triggers:
  - review this change
  - audit this code
example_queries:
  - Review the authentication changes before I commit them.
allowed-tools:
  - read_file
  - grep
  - bash
argument-hint: "[path]"
---

# Project code review

Read the relevant implementation and tests. Report findings in severity order
with file and line evidence. Do not edit files unless the user asks for fixes.
```

Keep `description` specific: it is required and participates in automatic activation. The `name` is for display; directory naming controls stable IDs and slash commands.

## 3. Trust and reload

Project skills load only in trusted workspaces. Restart Logician or run `/reload`, then inspect diagnostics with:

```bash
logician doctor --json
```

## 4. Test both paths

Invoke it directly:

```text
/project-code-review src/auth.ts
```

Then test automatic matching with a natural request:

```text
Review the authentication changes before I commit them. Do not edit files.
```

If it activates too often, narrow its description and triggers. If it rarely activates, add realistic `example_queries` rather than a long list of generic keywords.

## 5. Add supporting resources

Put detailed policy in `references/` and reusable helpers in `scripts/`. Link them from `SKILL.md` with relative paths so the main instructions stay short.
