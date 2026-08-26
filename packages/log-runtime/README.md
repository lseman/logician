# `@logician/log-runtime`

The composition layer for complete Logician applications.

It combines `@logician/log-core` with tools, skills, commands, MCP, memory,
plugins, Claude Code adapters, EoH, configuration, transcript state, and
TUI-facing application modules.

Use `AgentRuntime` as the application facade. `AgentCoreBridge` remains only as a
deprecated compatibility alias.

`log-runtime` may depend on feature packages; feature packages and `log-core`
must never depend on `log-runtime`. Architecture tests enforce this package
graph and verify that production workspace imports are declared.

## Replayable runtime events

`AgentRuntime.events` retains a bounded notification history for reconnecting
clients and diagnostics. Existing live subscriptions remain unchanged:

```ts
runtime.events.subscribe(handleNotification, {
  replay: { afterId: lastSeenSequence },
  onReplayGap: gap => resynchronize(gap),
});
```

Use `snapshot()` for pull-based inspection or filtering by runtime event type.
Journal cursors match protocol notification sequence numbers and remain
monotonic when retained history is cleared. Notifications retain session, run,
turn, and tool-call correlation across replay. Set
`eventStream.historyCapacity` when constructing `AgentRuntime` to control the
bounded replay window.

Run serialization and active/idle state live behind `RuntimeRunCoordinator`;
the public `AgentRuntime` interface remains unchanged. Operational failures emit
structured, replayable `diagnostic` events alongside compatible transcript
notices.
