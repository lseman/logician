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
