---
title: Getting Started
description: Install Logician, configure a model, and run your first session.
---

# Getting Started

This guide takes you from installation to a verified first turn. Logician needs an OpenAI-compatible model endpoint; everything else is optional.

## Prerequisites

| Requirement | Details |
|---|---|
| Runtime | Bun `>=1.3.14` for source builds; prebuilt binaries include the runtime |
| Model endpoint | An OpenAI-compatible chat-completions API |
| Search helpers | `rg` and `fd` are optional but recommended |

## Install

Use the prebuilt binary:

```bash
curl -fsSL https://raw.githubusercontent.com/lseman/logician/main/apps/tui/install.sh | bash
```

Or run from source:

```bash
git clone https://github.com/lseman/logician.git
cd logician
bun install
bun start
```

`make install` builds the executable and links it at `~/.local/bin/logician`.

## Configure a model

User-wide settings live in `~/.logician/settings.json`. A trusted workspace can override them with `.logician.json`:

```json
{
  "baseUrl": "http://127.0.0.1:8080",
  "model": "your-model-name",
  "permissionMode": "ask",
  "thinkingLevel": "off",
  "inferenceMode": "none"
}
```

`inferenceMode: "none"` is shown as **Provider** in the UI: Logician leaves sampling parameters to the provider. Keep secrets out of JSON. Logician loads `~/.logician/.env`, so an MCP or provider header can reference `${VARIABLE_NAME}` without storing its value in settings.

Selected environment overrides are also supported:

```bash
export LOGICIAN_LLM_URL=http://127.0.0.1:8080
export LOGICIAN_MODEL=your-model-name
```

Run a read-only diagnostic before starting:

```bash
logician doctor
```

## Start the TUI

Run `logician` from the repository you want to work on. On first use in a workspace, review the trust prompt before project configuration, skills, and extensions are loaded.

Type an outcome and press Enter:

```text
Find the cause of the failing authentication test, fix it without changing the public API, and run the focused tests.
```

Use `/help` for the live command list. The TUI streams text and tool progress, and asks before actions that require approval under the selected permission mode.

## Headless mode

For scripts and CI:

```bash
logician exec --jsonl "fix the failing test in src/utils.ts"
```

Standard output is machine-readable JSONL; diagnostics go to standard error.

## Next steps

- Follow the [First Session tutorial](/tutorials/first-session).
- Learn the [configuration layers](/guides/configuration).
- Add capabilities with [skills](/guides/skills), [plugins](/guides/plugins), or [MCP servers](/guides/mcp).
