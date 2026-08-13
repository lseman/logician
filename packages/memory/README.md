# @logician/memory

Persistent, workspace-scoped memory for TypeScript agents. The module owns
SQLite persistence, observation capture, consolidation, lexical and semantic
retrieval, retention, and memory evolution. It does not depend on Logician's
TUI or agent runtime.

## Runtime

The store currently uses `bun:sqlite`, so hosts must run on Bun. Semantic
retrieval is optional and uses `@huggingface/transformers` only when a local
embedder is configured.

## Store interface

```ts
import { createMemoryStore } from "@logician/memory";

const memory = createMemoryStore("./memory.db");

memory.setCurrentWorkspace(process.cwd());
memory.setCurrentSessionId("session-1");
memory.create("Retries use bounded exponential backoff", {
	type: "architecture",
});

const context = memory.getContext("session-1", 2_000, {
	objective: "debug request retries",
});
```

The `MemoryStore` interface is the primary seam for hosts that want to manage
their own lifecycle.

## Agent lifecycle adapter

`createMemoryHooks(memory, options)` returns a small structural hook object for
prompt capture, tool-result capture, context injection, turn completion, and
compaction. A host can use those hooks directly when its lifecycle matches, or
map them to its own event system. The adapter types (`MemoryAgentHooks`,
`MemoryAgentMessage`, and `MemoryTaskState`) are host-neutral and exported from
the package root.

Run `bun test` and `bun run typecheck` from this directory to verify the module.
