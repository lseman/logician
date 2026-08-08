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

Each line is a JSON object with typed events:

```json
{"type":"turn_start","turn_id":"turn_1"}
{"type":"message_start","turnId":"turn_1","role":"assistant"}
{"type":"thinking_token","token":"Analyzing test failures..."}
{"type":"token","token":"I'll fix the test."}
{"type":"tool_execution_start","tool":"grep","tool_name":"grep","tool_call_id":"tc_1","tool_args":{"pattern":"fail","path":"src/utils.ts"}}
{"type":"tool_execution_update","tool":"grep","tool_name":"grep","tool_call_id":"tc_1","partial_result":"{\"matches\":3}","update_kind":"arguments"}
{"type":"tool_execution_end","tool":"grep","tool_name":"grep","tool_call_id":"tc_1","result":"3 matches found","is_error":false}
{"type":"tool_execution_start","tool":"edit_file","tool_name":"edit_file","tool_call_id":"tc_2","tool_args":{"path":"src/utils.ts","changes":1}}
{"type":"tool_execution_end","tool":"edit_file","tool_name":"edit_file","tool_call_id":"tc_2","result":"1 change applied","is_error":false}
{"type":"tool_execution_start","tool":"bash","tool_name":"bash","tool_call_id":"tc_3","tool_args":{"command":"npm test"}}
{"type":"tool_execution_end","tool":"bash","tool_name":"bash","tool_call_id":"tc_3","result":"12/12 tests passed","is_error":false}
{"type":"message_update","turnId":"turn_1","message":{"role":"assistant","content":"Fixed the failing test. All 12 tests now pass."}}
{"type":"turn_end","turn_id":"turn_1","message":""}
{"type":"agent_settled","nextTurnCount":0}
```

### Event categories

| Category | Events |
|---|---|
| **Turn lifecycle** | `turn_start`, `message_start`, `message_update`, `message_end`, `turn_end`, `agent_settled` |
| **Streaming** | `token`, `thinking_token`, `text_start`, `text_end` |
| **Tool execution** | `tool_execution_start`, `tool_execution_update`, `tool_execution_end`, `tool_start`, `tool_end` |
| **Context & state** | `context_update`, `compaction`, `phase`, `queue_update`, `todos`, `steered`, `model_select` |
| **Observability** | `notice`, `agent_retry_start`, `agent_retry_end`, `agent_error`, `guardrail_nudge`, `repair_nudge` |
| **Interaction** | `permission_request`, `question_request` |
| **Session** | `session_delete`, `save_point`, `memory_update` |
| **Subagents** | `subagent_chunk`, `subagent_lifecycle` |

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
