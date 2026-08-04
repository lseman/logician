import assert from "node:assert/strict";
import { unlinkSync } from "node:fs";
import { afterEach, describe, test } from "node:test";
import { createMemoryHooks as createHooks } from "../memory-hooks.js";
import type { MemoryHooksConfig } from "../memory-hooks.js";
import { createMemoryStore } from "../store.js";
import type { CompressedObservation } from "../types.js";

let counter = 0;
const paths: string[] = [];
let backgroundTasks: Promise<void>[] = [];

function createMemoryHooks(store: ReturnType<typeof createMemoryStore>, config: MemoryHooksConfig = {}) {
  return createHooks(store, {
    ...config,
    onBackgroundTask: (task) => {
      backgroundTasks.push(task);
      config.onBackgroundTask?.(task);
    },
  });
}

async function drainBackgroundTasks() {
  await Promise.all(backgroundTasks.splice(0));
}

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
  backgroundTasks = [];
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
    await hooks.afterProviderResponse?.({
      model: "test",
      content: "Authentication now uses a bounded timeout in /workspace/auth.ts.",
      toolCallCount: 0,
      stopReason: "stop",
      iteration: 1,
    });
    await hooks.shouldStopAfterTurn?.({ messages: [], iteration: 1, hadToolCalls: false });
    await drainBackgroundTasks();
    assert.equal(savedMemories, 1);
    assert.equal(store.list({ limit: 10 }).length, 1);
    assert.equal(store.list({ limit: 10 })[0]?.sourceObservationIds?.length, 2);
    store.close();
  });

  test("synthesizes a grounded turn episode from intent, mutations, outcome, and verification", async () => {
    const store = testStore();
    const hooks = createMemoryHooks(store, { injectContext: false, autoConsolidate: true });
    await hooks.beforeAgentStart?.({
      prompt: "Move session storage into each project folder",
      systemPrompt: "",
      messages: [],
    });
    await hooks.afterToolCall?.({
      toolCall: { id: "edit", name: "edit_file", arguments: "{}" },
      args: { path: "/workspace/src/session-store.ts" },
      result: "Updated file",
      isError: false,
      iteration: 0,
    });
    await hooks.afterToolCall?.({
      toolCall: { id: "test", name: "exec_command", arguments: "{}" },
      args: { command: "bun test session-store.test.ts" },
      result: "1 pass, 0 fail",
      isError: false,
      iteration: 1,
    });
    await hooks.afterProviderResponse?.({
      model: "test",
      content: "Session history is now stored in each workspace's .logician directory.",
      toolCallCount: 0,
      stopReason: "stop",
      iteration: 2,
    });
    await hooks.shouldStopAfterTurn?.({ messages: [], iteration: 2, hadToolCalls: false });
    await drainBackgroundTasks();

    const episode = store.listObservations("session-1", 10).find((item) => item.id.startsWith("episode:"));
    assert.ok(episode);
    assert.equal(episode.type, "implementation");
    assert.match(episode.narrative, /Move session storage/);
    assert.match(episode.narrative, /stored in each workspace/);
    assert.ok(episode.files.includes("/workspace/src/session-store.ts"));
    assert.ok(episode.facts.some((fact) => fact.startsWith("Verification:")));
    assert.equal(store.list({ limit: 10 })[0]?.type, "architecture");
    const context = store.getContext("session-1", 2000, "session storage workspace");
    assert.match(context, /Session history is now stored/);
    assert.doesNotMatch(context, /Updated file/);
    store.close();
  });

  test("accepts specific model claims only when evidence and verification are grounded", async () => {
    const store = testStore();
    const hooks = createMemoryHooks(store, {
      injectContext: false,
      autoConsolidate: false,
      semanticExtractor: async () => JSON.stringify({
        kind: "bugfix",
        title: "Authentication timeout now uses a bounded retry policy",
        summary: "The authentication timeout path was corrected and its focused test completed successfully.",
        claims: [{
          text: "Authentication timeout retries are bounded by the policy implemented in auth.ts.",
          confidence: 0.96,
          status: "verified",
          evidenceEventIds: ["edit-grounded", "test-grounded"],
        }],
        outcome: "The focused authentication test passes.",
        filesRead: [],
        filesModified: ["/workspace/auth.ts"],
        concepts: ["problem-solution", "verified"],
      }),
    });
    await hooks.beforeAgentStart?.({ prompt: "Fix the authentication timeout bug", systemPrompt: "", messages: [] });
    await hooks.afterToolCall?.({
      toolCall: { id: "edit-grounded", name: "edit_file", arguments: "{}" },
      args: { path: "/workspace/auth.ts" }, result: "Updated timeout policy", isError: false, iteration: 0,
    });
    await hooks.afterToolCall?.({
      toolCall: { id: "test-grounded", name: "exec_command", arguments: "{}" },
      args: { command: "bun test auth.test.ts" }, result: "1 pass, 0 fail", isError: false, iteration: 1,
    });
    await hooks.afterProviderResponse?.({ model: "test", content: "Fixed and verified.", toolCallCount: 0, stopReason: "stop", iteration: 2 });
    await hooks.shouldStopAfterTurn?.({ messages: [], iteration: 2, hadToolCalls: false });
    await drainBackgroundTasks();

    const episode = store.listObservations("session-1", 10).find((item) => item.id.startsWith("episode:"));
    assert.equal(episode?.title, "Authentication timeout now uses a bounded retry policy");
    assert.match(episode?.facts[0] || "", /verified; confidence=0\.96; evidence=edit-grounded,test-grounded/);
    store.close();
  });

  test("rejects hallucinated model evidence and falls back to deterministic synthesis", async () => {
    const store = testStore();
    const hooks = createMemoryHooks(store, {
      injectContext: false,
      autoConsolidate: false,
      semanticExtractor: async () => ({
        kind: "implementation",
        title: "Secret deployment workflow was implemented successfully",
        summary: "A deployment workflow was supposedly added to an unrelated file and verified.",
        claims: [{ text: "Production deploys now use canaries.", confidence: 1, status: "verified", evidenceEventIds: ["invented"] }],
        filesRead: [], filesModified: ["/invented/deploy.ts"], concepts: [],
      }),
    });
    await hooks.beforeAgentStart?.({ prompt: "Update auth timeout", systemPrompt: "", messages: [] });
    await hooks.afterToolCall?.({
      toolCall: { id: "real-edit", name: "edit_file", arguments: "{}" },
      args: { path: "/workspace/auth.ts" }, result: "Updated timeout", isError: false, iteration: 0,
    });
    await hooks.afterProviderResponse?.({ model: "test", content: "Authentication timeout is now bounded.", toolCallCount: 0, stopReason: "stop", iteration: 1 });
    await hooks.shouldStopAfterTurn?.({ messages: [], iteration: 1, hadToolCalls: false });
    await drainBackgroundTasks();

    const episode = store.listObservations("session-1", 10).find((item) => item.id.startsWith("episode:"));
    assert.match(episode?.title || "", /Authentication timeout is now bounded/);
    assert.doesNotMatch(JSON.stringify(episode), /canaries|invented/);
    store.close();
  });

  test("honors a model skip for turns without durable knowledge", async () => {
    const store = testStore();
    const hooks = createMemoryHooks(store, {
      injectContext: false,
      autoConsolidate: false,
      semanticExtractor: async () => ({ skip: true }),
    });
    await hooks.beforeAgentStart?.({ prompt: "Thanks", systemPrompt: "", messages: [] });
    await hooks.afterProviderResponse?.({ model: "test", content: "You're welcome.", toolCallCount: 0, stopReason: "stop", iteration: 0 });
    await hooks.shouldStopAfterTurn?.({ messages: [], iteration: 0, hadToolCalls: false });
    await drainBackgroundTasks();
    assert.equal(store.listObservations("session-1", 10).some((item) => item.id.startsWith("episode:")), false);
    store.close();
  });

  test("runs semantic extraction in the background and serializes turns", async () => {
    const store = testStore();
    let releaseFirst!: () => void;
    const firstGate = new Promise<void>((resolve) => { releaseFirst = resolve; });
    const started: string[] = [];
    const hooks = createMemoryHooks(store, {
      injectContext: false,
      autoConsolidate: false,
      semanticExtractor: async ({ userPrompt }) => {
        const intent = JSON.parse(userPrompt).user_intent as string;
        started.push(intent);
        if (intent === "first turn") await firstGate;
        return { skip: true };
      },
    });

    await hooks.beforeAgentStart?.({ prompt: "first turn", systemPrompt: "", messages: [] });
    const firstResult = hooks.shouldStopAfterTurn?.({ messages: [], iteration: 0, hadToolCalls: false });
    assert.equal(firstResult, undefined);
    await Promise.resolve();

    await hooks.beforeAgentStart?.({ prompt: "second turn", systemPrompt: "", messages: [] });
    const secondResult = hooks.shouldStopAfterTurn?.({ messages: [], iteration: 1, hadToolCalls: false });
    assert.equal(secondResult, undefined);
    await Promise.resolve();
    assert.deepEqual(started, ["first turn"]);

    releaseFirst();
    await drainBackgroundTasks();
    assert.deepEqual(started, ["first turn", "second turn"]);
    store.close();
  });
});
