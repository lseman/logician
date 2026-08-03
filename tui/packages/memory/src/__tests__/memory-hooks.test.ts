import assert from "node:assert/strict";
import { unlinkSync } from "node:fs";
import { afterEach, describe, test } from "node:test";
import { createMemoryHooks } from "../memory-hooks.js";
import { createMemoryStore } from "../store.js";
import type { CompressedObservation } from "../types.js";

let counter = 0;
const paths: string[] = [];

function testStore() {
  const path = `/tmp/logician-memory-hooks-${process.pid}-${++counter}.db`;
  paths.push(path);
  const store = createMemoryStore(path);
  store.setCurrentSessionId("session-1");
  store.setCurrentWorkspace("/workspace");
  store.createSession("session-1", { cwd: "/workspace", workspace: "/workspace" });
  return store;
}

afterEach(() => {
  for (const path of paths.splice(0)) {
    try { unlinkSync(path); } catch {}
  }
});

describe("createMemoryHooks observation capture", () => {

  test("retrieves memory against live task state rather than only the initial prompt", async () => {
    const store = testStore();
    store.create("The parser requires CRLF-safe edits", {
      strength: 8,
      files: ["src/parser.ts"],
    });
    store.create("Unrelated deployment credentials", { strength: 10 });
    const hooks = createMemoryHooks(store, { contextBudget: 1000 });

    await hooks.beforeAgentStart?.({ prompt: "Continue", systemPrompt: "", messages: [] });
    const transformed = await hooks.transformContext?.({
      messages: [],
      iteration: 2,
      taskState: {
        objective: "Fix parser line endings",
        phase: "implement",
        hypotheses: [],
        evidence: [],
        changedFiles: ["src/parser.ts"],
        verification: [],
        blockers: [],
        toolCalls: 1,
        toolFailures: 0,
      },
    });
    const last = transformed?.messages?.at(-1);
    const injected = String(last && "content" in last ? last.content || "" : "");
    assert.match(injected, /CRLF-safe edits/);
    assert.doesNotMatch(injected, /deployment credentials/);
    store.close();
  });

  test("replaces stale retrieved context instead of accumulating system blocks", async () => {
    const store = testStore();
    store.create("Authentication uses bounded exponential retries", { strength: 8 });
    const hooks = createMemoryHooks(store, { contextBudget: 1000 });
    await hooks.beforeAgentStart?.({ prompt: "Fix authentication retries", systemPrompt: "", messages: [] });

    const first = await hooks.transformContext?.({ messages: [], iteration: 1 });
    const second = await hooks.transformContext?.({
      messages: first?.messages || [],
      iteration: 2,
    });
    const memoryBlocks = (second?.messages || []).filter(
      (message) =>
        message != null &&
        "content" in message &&
        typeof message.content === "string" &&
        message.content.startsWith("# Agent Context\n"),
    );
    assert.equal(memoryBlocks.length, 1);
    store.close();
  });
  test("captures prompts and tool outcomes without duplicating pre-tool intent", async () => {
    const store = testStore();
    const saved: CompressedObservation[] = [];
    const hooks = createMemoryHooks(store, {
      injectContext: false,
      onObservationSaved: (observation) => saved.push(observation),
    });

    await hooks.beforeAgentStart?.({
      prompt: "Fix the authentication timeout",
      systemPrompt: "",
      messages: [],
    });
    assert.equal(hooks.beforeToolCall, undefined);
    await hooks.afterToolCall?.({
      toolCall: { id: "call-1", name: "edit_file", arguments: "{}" },
      args: { path: "/workspace/auth.ts" },
      result: "Updated authentication timeout",
      isError: false,
      iteration: 0,
    });

    const observations = store.listObservations("session-1", 10);
    assert.equal(observations.length, 2);
    assert.equal(saved.length, 2);
    const promptObservation = observations.find((observation) => observation.type === "conversation");
    assert.match(promptObservation?.title || "", /Fix the authentication timeout/);
    assert.equal(promptObservation?.narrative, "Fix the authentication timeout");
    assert.deepEqual(
      observations.map((observation) => observation.id).sort(),
      [saved[0]!.id, "call-1:post"].sort(),
    );
    assert.equal(store.getSession("session-1")?.observationCount, 2);
    store.close();
  });

  test("deduplicates repeated prompts and equivalent tool calls", async () => {
    const store = testStore();
    const hooks = createMemoryHooks(store, { injectContext: false });
    const prompt = { prompt: "Inspect the authentication flow", systemPrompt: "", messages: [] };
    await hooks.beforeAgentStart?.(prompt);
    await hooks.beforeAgentStart?.(prompt);
    const tool = {
      toolCall: { id: "call-a", name: "read_file", arguments: "{}" },
      args: { path: "/workspace/auth.ts" },
      result: "source",
      isError: false,
      iteration: 0,
    };
    await hooks.afterToolCall?.(tool);
    await hooks.afterToolCall?.({ ...tool, toolCall: { ...tool.toolCall, id: "call-b" } });

    assert.equal(store.listObservations("session-1", 10).length, 2);
    store.close();
  });

  test("does not save interrupted tool failures", async () => {
    const store = testStore();
    const hooks = createMemoryHooks(store, { injectContext: false });
    await hooks.afterToolCall?.({
      toolCall: { id: "call-cancelled", name: "bash", arguments: "{}" },
      args: { command: "long-task" },
      result: "Cancelled by user",
      isError: true,
      iteration: 0,
    });
    assert.equal(store.listObservations("session-1", 10).length, 0);
    store.close();
  });

  test("does not notify when no active session is configured", async () => {
    const path = `/tmp/logician-memory-hooks-${process.pid}-${++counter}.db`;
    paths.push(path);
    const store = createMemoryStore(path);
    let notices = 0;
    const hooks = createMemoryHooks(store, {
      injectContext: false,
      onObservationSaved: () => notices++,
    });

    await hooks.beforeAgentStart?.({ prompt: "hello", systemPrompt: "", messages: [] });

    assert.equal(notices, 0);
    store.close();
  });

  test("consolidates high-signal observations at a turn boundary and notifies", async () => {
    const store = testStore();
    let savedMemories = 0;
    const hooks = createMemoryHooks(store, {
      injectContext: false,
      onMemoriesSaved: (memories) => { savedMemories += memories.length; },
    });
    for (const id of ["one", "two"]) {
      await hooks.afterToolCall?.({
        toolCall: { id, name: "edit_file", arguments: "{}" },
        args: { path: "/workspace/auth.ts", edit: id },
        result: `Updated auth flow ${id}`,
        isError: false,
        iteration: 0,
      });
    }
    await hooks.shouldStopAfterTurn?.({ messages: [], iteration: 1, hadToolCalls: false });
    assert.equal(savedMemories, 1);
    assert.equal(store.list({ limit: 10 }).length, 1);
    assert.equal(store.list({ limit: 10 })[0]?.sourceObservationIds?.length, 2);
    store.close();
  });
});
