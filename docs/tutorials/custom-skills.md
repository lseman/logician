---
title: Custom Skills
description: Write your own SKILL.md-driven capabilities.
---

# Custom Skills

Create custom skills to give the agent specialized instructions for your codebase.

## Step 1: Create the skill directory

```bash
mkdir -p skills/code-review
```

## Step 2: Write SKILL.md

```markdown
<!-- name: code-review -->
<!-- displayName: Code Review -->
<!-- description: Review code for quality, security, and best practices -->
<!-- triggers: review review- code-review audit -->

# Code Review

When reviewing code, check for:

1. **Security** — SQL injection, XSS, auth bypass
2. **Performance** — N+1 queries, unnecessary allocations
3. **Style** — Consistent formatting, naming conventions
4. **Edge cases** — Null handling, error paths, bounds

Provide specific line references and suggested fixes.
```

## Step 3: Activate

The skill activates automatically when your prompt contains "review" or "audit".

```
> Review src/auth.ts for security issues
```

The agent responds with:

```
[SKILL] Activated: code-review
💭 Reviewing auth.ts for security issues...
🔧 read_file src/auth.ts
✅ Found 2 security issues:
   - Line 23: SQL injection risk
   - Line 45: Missing null check
```

## Step 4: Test

```bash
npm start -- exec "review src/auth.ts" --jsonl
```

## Tips

- Use specific trigger words relevant to your codebase
- Keep instructions concise and actionable
- Include examples of good and bad patterns
- Test with `doctor --skills` to verify loading
