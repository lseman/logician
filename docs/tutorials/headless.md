---
title: Headless Mode
description: Run one agent task with machine-readable JSONL output.
---

# Headless Mode

`logician exec` runs one prompt without the interactive TUI. Use `--jsonl` when another program will consume the output.

## Run a task

```bash
logician exec --jsonl "fix the failing test in src/utils.ts"
```

Place `--` before a prompt fragment that begins with a hyphen:

```bash
logician exec --jsonl -- "- inspect why this command fails"
```

## Output contract

Standard output contains newline-delimited records. The stream includes content deltas, thinking deltas when supplied by the provider, tool lifecycle records, errors, terminal metadata, and a final `done` record. Diagnostics and configuration warnings go to standard error so stdout remains parseable.

Consumers should branch on each record's `type` and ignore unknown fields and future record types. Do not parse the human-readable TUI event vocabulary as if it were the headless wire contract.

Example shape:

```jsonl
{"type":"content","content":"I found the failing assertion."}
{"type":"tool_use","id":"call_1","name":"read_file","input":{"path":"src/utils.ts"}}
{"type":"tool_result","id":"call_1","name":"read_file","status":"success","output":"..."}
{"type":"metadata","meta":{"receipt_kind":"terminal","status":"completed"}}
{"type":"done"}
```

## Exit status

The command returns `0` after a successful run, `1` when agent execution reports an error, and `2` for invalid CLI usage or setup errors handled before the run begins.

## CI example

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: oven-sh/setup-bun@v2
  - run: bun install --frozen-lockfile
  - run: bun start -- exec --jsonl "inspect the failing checks and propose a fix" > logician.jsonl
  - uses: actions/upload-artifact@v4
    with:
      name: logician-output
      path: logician.jsonl
```

CI must provide the model endpoint, credentials, and an explicit trust policy appropriate for that environment.
