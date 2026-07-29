---
title: Headless Mode
description: JSONL streaming for CI/CD and automation.
---

# Headless Mode

Run Logician without the TUI — outputs structured JSONL for CI/CD pipelines and automation.

## Basic usage

```bash
npm start -- exec --jsonl "fix the failing test in src/utils.ts"
```

## JSONL output

Each line is a JSON object:

```json
{"type":"thinking","content":"Analyzing test failures..."}
{"type":"tool_call","tool":"grep","args":{"pattern":"fail","path":"src/utils.ts"}}
{"type":"tool_result","tool":"grep","success":true,"matches":3}
{"type":"tool_call","tool":"edit_file","args":{"path":"src/utils.ts","changes":1}}
{"type":"tool_result","tool":"edit_file","success":true,"changes":1}
{"type":"verification","tool":"bash","args":{"command":"npm test"}}
{"type":"tool_result","tool":"bash","success":true,"stdout":"12/12 tests passed"}
{"type":"response","content":"Fixed the failing test. All 12 tests now pass."}
```

## CI/CD integration

```yaml
# .github/workflows/logician.yml
name: Logician
on: [pull_request]
jobs:
  fix:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: oven-sh/setup-bun@v1
      - run: cd tui && bun install
      - run: |
          cd tui
          npm start -- exec --jsonl "fix lint errors" \
            > output.jsonl
      - uses: actions/upload-artifact@v4
        with:
          name: logician-output
          path: output.jsonl
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Error |
| 2 | Cancelled |
| 3 | Timeout |

## Configuration

```json
{
  "headless": {
    "timeout": 300,
    "maxIterations": 20,
    "streamOutput": true
  }
}
```
