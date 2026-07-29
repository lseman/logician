---
title: Troubleshooting
description: Common issues and fixes.
---

# Troubleshooting

Common issues and how to resolve them.

## Connection refused

```
Error: ECONNREFUSED 127.0.0.1:8080
```

**Fix:** Start your LLM backend. The TUI expects an OpenAI-compatible server at the configured URL.

```bash
# Verify the backend is running
curl http://127.0.0.1:8080/v1/models
```

## Context window full

```
Error: context_full
```

**Fix:** Enable compaction or reduce context:

```json
{
  "compaction": {
    "enabled": true,
    "triggerFraction": 0.75
  }
}
```

## Edit fails — text not found

```
Error: Edit failed — text not found in file
```

**Fix:** The file content changed between read and edit. Re-read the file and retry.

## Skill not loading

```
[WARN] Skill not found: my-skill
```

**Fix:** Check that `SKILL.md` is in the correct location and has valid frontmatter:

```
skills/my-skill/SKILL.md
```

## MCP server not starting

```
Error: MCP server filesystem failed to start
```

**Fix:** Verify the command and args are correct:

```bash
npx -y @modelcontextprotocol/server-filesystem /workspace
```

## Diagnostic commands

```bash
# Full system check
npm start -- doctor --json

# Check MCP connectivity
npm start -- doctor --mcp

# Check skills
npm start -- doctor --skills
```
