---
title: Troubleshooting
description: Diagnose model, configuration, skill, MCP, and terminal problems.
---

# Troubleshooting

Start with the read-only doctor report:

```bash
logician doctor
logician doctor --json
```

It validates configuration, workspace detection, dependencies, declared MCP servers, skill diagnostics, permissions, and sandbox readiness without contacting the model or starting MCP servers.

## Model connection fails

Confirm `baseUrl` includes the provider's expected API root and that the model ID is valid. For a local endpoint:

```bash
curl http://127.0.0.1:8080/v1/models
```

Check `LOGICIAN_LLM_URL`, `LOGICIAN_MODEL`, user settings, and trusted project settings in that priority order.

## Project settings do not apply

Project configuration, skills, and extensions load only for trusted workspaces. Restart in the project root and review the trust prompt. Use `LOGICIAN_TRUST=always` only in controlled environments.

## Context is full

Run `/context`, then `/compact`. For proactive compaction, configure token reserves rather than a percentage:

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000
  }
}
```

## An edit cannot find its target

The file changed or the expected text was not exact. Re-read the file and retry with current content. Do not weaken exact-match editing merely to force an ambiguous replacement.

## A skill does not load

Verify that the file is named `SKILL.md`, has YAML frontmatter with a non-empty `description`, and lives below `~/.agents/skills/`, an enabled plugin's `skills/`, or a trusted project's `skills/`/`.agents/skills/`. Run `logician doctor --json` for parse, metadata, ignore, and collision diagnostics.

## An MCP server does not start

Run a stdio server's configured command directly. For HTTP, verify the URL, `POST` support, authentication header, and response content type. Secrets referenced as `${NAME}` should be in `~/.logician/.env` or the parent environment. Use `/mcp list` to inspect runtime state.

## The binary does not reflect source changes

Source builds live at `apps/tui/dist/logician`. Run:

```bash
make install
command -v logician
readlink -f "$(command -v logician)"
```

The default install link should resolve to the current checkout's `apps/tui/dist/logician`. Restart running processes after rebuilding.
