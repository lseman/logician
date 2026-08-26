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

`AgentRuntime` is a compatibility facade over focused application modules:

- `ConversationSession` owns live harness creation, history, queues, abort,
  compaction, and branches.
- `ConversationIdentity` keeps the session store, event stream, hook paths, and
  memory identity synchronized.
- `TurnOrchestrator` and `SessionRunner` own turn admission and execution.
- `CommandDispatcher` owns slash, skill, prompt, and reload routing.
- `PluginLifecycle`, `RuntimeConfiguration`, and `RuntimeActivity` own their
  respective state transitions and side effects.
- `LegroomGateway` is the adapter for the optional compression worker.

Run serialization and active/idle state live behind `RuntimeRunCoordinator`;
concurrent exactly-once initialization and retry/reset behavior live behind
`RuntimeStartupCoordinator`. Each module is tested through the same interface
used by the facade. Operational failures emit structured, replayable
`diagnostic` events alongside compatible transcript notices.

Provider turns, tools, hooks, MCP HTTP/stdio requests, and plugin HTTP/shell
hooks share `@logician/log-core/runtime`'s `CancellationScope` for parent
propagation, typed deadlines, and deterministic cleanup.
