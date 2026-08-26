# `@logician/log-core`

The foundational, product-independent agent engine.

It owns the provider loop, execution harness, interactive agent sessions, hooks,
policies, tool contracts, conversation state, durable session storage, and
compaction. It must not import Logician feature or product packages. Runtime
integrations belong in `@logician/log-runtime`.

The lifecycle roles are intentionally distinct:

- `AgentHarness` is the functional execution kernel: a prepared context and
  prompts go in; newly produced messages and events come out.
- `AgentSession` owns interactive coordination: configuration, history, queues,
  persistence, branches, continuation, hooks, and turn lifecycle.
- `AgentSession` calls `AgentHarness`; the harness never imports or retains a session.
- `SessionStore` persists the durable JSONL conversation tree.

Dependency direction:

```text
agent-core <- agent-blocks / eoh / autoresearch <- agent-runtime <- tui
```

Applications use `@logician/log-core/session`. Lower-level engines and tests may
use `@logician/log-core/harness` directly.

## Event journal

Hosts that need diagnostics, reconnection replay, or late subscribers can attach
the opt-in bounded journal to their existing event sink:

```ts
import { EventJournal, runAgentLoop } from "@logician/log-core";
import type { AgentEvent } from "@logician/log-core";

const events = new EventJournal<AgentEvent>({ capacity: 2_000 });

await runAgentLoop(context, prompts, {
  ...config,
  onEvent: event => events.append(event),
});

// Resume a client strictly after its last acknowledged journal cursor.
const missed = events.snapshot({ afterId: lastSeenId });
```

Journal cursors are monotonic even after `clear()`. A capacity of zero disables
retention while preserving live subscriptions. Subscriber failures are isolated
and can be reported through `onSubscriberError`.
