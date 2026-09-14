# Diagnosing empty bash calls

Start Logician with tracing enabled, then reproduce the failing prompt:

```sh
LOGICIAN_BASH_DEBUG=1 LOGICIAN_BASH_DEBUG_FILE=/tmp/logician-bash.jsonl bun run dev
```

The file is append-only JSONL. Choose a fresh filename for each reproduction.
Full prompts, tool schemas, model responses, commands, and returned output are
included, so treat the trace as private session data. HTTP headers are not logged.
Without the file variable, only argument and result previews go to stderr.

Follow these events in order:

1. `provider.request`: the final request body after provider transformations,
   including messages and the actual schema sent to the model.
2. `provider.sse`: incoming SSE data lines before JSON parsing.
3. `provider.assembled`: accumulated tool calls before missing arguments are
   replaced with `{}`. Match provider events using `requestId`.
4. `tool.stage`: bash call arguments at raw (0), parsed (1), prepared (2),
   and execution (3) stages. Match the call `id` to the provider's tool call.
5. `tool.result`: the result returned to the agent, including error text.

If arguments are already empty in the provider response, inspect the request's
prompt/schema and provider behavior. If they disappear between tool stages,
the parsing or preparation path is responsible. A nonempty command returning
`(no output)` is distinct from missing arguments.

The in-memory `getBashDebuggerReport()` API summarizes the latest 50 bash calls
(default display: 10). It snapshots arguments so later mutations cannot rewrite
earlier stages. Disable tracing by removing the environment variables on restart.

## Confirmed provider-schema failure (2026-09-14)

A controlled comparison against the configured local server reproduced empty
arguments with the original bash schema. The same prompt produced
`{"command":"printf hello"}` after removing the requirement-only `anyOf`.
Expanding the branches did not resolve the failure. The normalizer now flattens
requirement-only alternatives, preserves their common required fields, and
describes the alternatives in the schema. Bash still validates that callers
supply exactly one of a nonempty command or a valid batch.

The original trace confirms that error feedback reached the provider on the next
request; the client was not dropping arguments or feedback. Of the 18 exposed
tools in that trace, bash alone used this top-level alternative pattern.

Also corrected: invalid commands and unsuccessful command/batch execution now
return error flags. Unrelated string fields (such as a terminal ID or description)
are no longer guessed to be shell commands. Prompt instructions use the canonical
JSON fields consistently; recognized command aliases remain accepted.

Live verification with the original system prompt and all 18 tools:

- A fresh request generated a nonempty command.
- The real bash tool executed an allowlisted printf and returned its output.
- A subsequent provider request received that tool result and reported the output.
- Batch generation produced both requested command entries.

Restart Logician and start a fresh conversation when retesting. Replaying the
long failed history still produced an empty call in one probe even with the fixed
schema; recovery of that existing conversation is not guaranteed.
