# `@logician/agent-core`

The foundational, product-independent agent engine.

It owns the provider loop, harness, hooks, policies, tool contracts, conversation
state, sessions, and compaction. It must not import Logician feature or product
packages. Runtime integrations belong in `@logician/agent-runtime`.

Dependency direction:

```text
agent-core <- agent-blocks / eoh / autoresearch <- agent-runtime <- tui
```

Keep the root interface small. Add a subpath export only when another package has
a demonstrated need for that seam.
