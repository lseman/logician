import { describe, test } from "node:test";
import assert from "node:assert/strict";
import { createMemoryStore } from "../store.js";
import {
  remember,
  recall,
  searchObservations,
  forget,
  listMemories,
  consolidate,
  getContext,
  autoForget,
  autoTierMemories,
} from "../hook-adapter.js";

let counter = 0;
function dbPath(): string {
  return `/tmp/logician-memory-test-${++counter}-${Date.now()}.db`;
}

// ── Memory CRUD ────────────────────────────────────────────────────────────

describe("createMemoryStore — memories", () => {
  test("creates and retrieves a memory entry", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("test content", {
      sessionIds: ["sess-1"],
      strength: 7,
      concepts: ["hello", "world"],
    });

    assert.equal(entry.content, "test content");
    assert.equal(entry.strength, 7);
    assert.ok(entry.concepts.includes("hello"));
    assert.ok(entry.concepts.includes("world"));
    assert.equal(entry.type, "fact");

    const retrieved = store.get(entry.id)!;
    assert.equal(retrieved.content, "test content");
    assert.equal(retrieved.id, entry.id);

    store.close();
  });

  test("auto-extracts concepts from content", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("fix auth error in login flow with #auth and #api", {});
    assert.ok(entry.concepts.includes("auth"));
    assert.ok(entry.concepts.includes("error"));
    assert.ok(entry.concepts.includes("login"));
    assert.ok(entry.concepts.includes("api"));

    store.close();
  });

  test("auto-assigns strength based on keywords", () => {
    const store = createMemoryStore(dbPath());
    const errEntry = store.create("fix error in auth module");
    assert.equal(errEntry.strength, 8);

    const todoEntry = store.create("todo next refactor database layer");
    assert.equal(todoEntry.strength, 4);

    store.close();
  });

  test("lists memories sorted by strength", () => {
    const store = createMemoryStore(dbPath());
    store.create("low priority task", { strength: 1 });
    store.create("critical fix bug now", { strength: 9 });
    store.create("medium priority todo", { strength: 5 });

    const all = store.list();
    assert.equal(all.length, 3);
    assert.ok(all[0].strength >= all[all.length - 1].strength);

    store.close();
  });

  test("filters by type", () => {
    const store = createMemoryStore(dbPath());
    store.create("error crash fix", { type: "bug" });
    store.create("auth pattern decided", { type: "pattern" });
    store.create("just a fact", { type: "fact" });

    const bugs = store.list({ type: "bug" });
    assert.equal(bugs.length, 1);
    assert.equal(bugs[0].type, "bug");

    store.close();
  });

  test("filters by minStrength", () => {
    const store = createMemoryStore(dbPath());
    store.create("low priority", { strength: 2 });
    store.create("high priority fix bug crash", { strength: 9 });
    store.create("medium priority", { strength: 5 });

    const high = store.list({ minStrength: 6 });
    assert.equal(high.length, 1);
    assert.equal(high[0].strength, 9);

    store.close();
  });

  test("filters by concept", () => {
    const store = createMemoryStore(dbPath());
    store.create("auth error in login", { concepts: ["auth", "error"] });
    store.create("database connection issue", { concepts: ["database", "connection"] });
    store.create("auth and database combined", { concepts: ["auth", "database"] });

    const authMemories = store.list({ concepts: ["auth"] });
    assert.equal(authMemories.length, 2);

    const both = store.list({ concepts: ["auth", "database"] });
    assert.equal(both.length, 1);
    assert.equal(both[0].content, "auth and database combined");

    store.close();
  });

  test("text search across content and concepts", () => {
    const store = createMemoryStore(dbPath());
    store.create("authentication error in login flow", { concepts: ["auth"] });
    store.create("database connection timeout issue", { concepts: ["database"] });
    store.create("api endpoint returns 404 not found", { concepts: ["api"] });

    const auth = store.list({ search: "authentication" });
    assert.ok(auth.length >= 1);
    assert.ok(auth.some(m => m.content.toLowerCase().includes("authentication")));

    const notFound = store.list({ search: "not found" });
    assert.ok(notFound.length >= 1);

    store.close();
  });

  test("updates a memory entry", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("original content", { type: "fact" });

    const updated = store.update(entry.id, {
      content: "updated content",
      strength: 8,
    })!;
    assert.equal(updated.content, "updated content");
    assert.equal(updated.strength, 8);

    const retrieved = store.get(entry.id)!;
    assert.equal(retrieved.content, "updated content");

    store.close();
  });

  test("deletes a memory entry", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("to be deleted", { type: "fact" });

    assert.equal(store.remove(entry.id), true);
    assert.equal(store.get(entry.id), null);
    assert.equal(store.remove(entry.id), false);

    store.close();
  });

  test("recall formats as markdown", () => {
    const store = createMemoryStore(dbPath());
    store.create("markdown memory one", { strength: 8, type: "pattern" });
    store.create("markdown memory two", { strength: 6, type: "bug" });

    const md = store.recall({ limit: 10 }, { format: "markdown" });
    assert.ok(md.includes("#"));
    assert.ok(md.includes("[8/10]"));
    assert.ok(md.includes("markdown memory one"));

    store.close();
  });

  test("recall formats as system-prompt", () => {
    const store = createMemoryStore(dbPath());
    store.create("prompt memory content", { strength: 7, type: "pattern" });

    const formatted = store.recall({ limit: 10 }, { format: "system-prompt" });
    assert.ok(formatted.includes("# pattern [7/10]"));
    assert.ok(formatted.includes("prompt memory content"));

    store.close();
  });

  test("handles empty query gracefully", () => {
    const store = createMemoryStore(dbPath());
    const listResult = store.list();
    assert.deepStrictEqual(listResult, []);

    const recallResult = store.recall({});
    assert.equal(recallResult, "");

    store.close();
  });
});

// ── Sessions ───────────────────────────────────────────────────────────────

describe("createMemoryStore — sessions", () => {
  test("creates and retrieves a session", () => {
    const store = createMemoryStore(dbPath());
    const session = store.createSession("sess-1", {
      project: "my-project",
      cwd: "/home/user/my-project",
      model: "claude-sonnet-4-20250514",
      tags: ["frontend", "typescript"],
      firstPrompt: "Build a login page",
    });

    assert.equal(session.id, "sess-1");
    assert.equal(session.project, "my-project");
    assert.equal(session.status, "active");
    assert.equal(session.observationCount, 0);
    assert.deepStrictEqual(session.tags, ["frontend", "typescript"]);
    assert.equal(session.firstPrompt, "Build a login page");

    const retrieved = store.getSession("sess-1");
    assert.ok(retrieved);
    assert.equal(retrieved!.id, "sess-1");
    assert.equal(retrieved!.project, "my-project");

    store.close();
  });

  test("lists sessions with filters", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "proj-a" });
    store.createSession("sess-2", { project: "proj-b" });
    store.createSession("sess-3", { project: "proj-a" });

    const projA = store.listSessions({ project: "proj-a" });
    assert.equal(projA.length, 2);

    store.close();
  });

  test("updates session status", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    const updated = store.updateSession("sess-1", {
      status: "completed",
      summary: "Completed the task",
    })!;
    assert.equal(updated.status, "completed");
    assert.equal(updated.summary, "Completed the task");

    store.close();
  });
});

// ── Observations ───────────────────────────────────────────────────────────

describe("createMemoryStore — observations", () => {
  test("captures an observation with synthetic compression", () => {
    const store = createMemoryStore(dbPath());
    const comp = store.observe({
      id: "obs-1",
      sessionId: "sess-1",
      timestamp: new Date().toISOString(),
      hookType: "post_tool_use",
      toolName: "bash",
      toolInput: { command: "ls -la" },
      toolOutput: "file1.txt\nfile2.txt",
      raw: { tool_name: "bash", tool_input: { command: "ls -la" }, tool_output: "file1.txt" },
    });

    assert.ok(comp);
    assert.equal(comp.id, "obs-1");
    assert.equal(comp.type, "command_run");
    assert.equal(comp.importance, 5);
    assert.ok(comp.concepts.length >= 0);

    store.close();
  });

  test("errors get high importance", () => {
    const store = createMemoryStore(dbPath());
    const comp = store.observe({
      id: "obs-err",
      sessionId: "sess-1",
      timestamp: new Date().toISOString(),
      hookType: "post_tool_failure",
      toolName: "bash",
      toolOutput: "Error: command failed with exit code 1",
      raw: { tool_name: "bash", error: "Error: command failed" },
    });

    assert.ok(comp);
    assert.equal(comp.importance, 8);
    assert.equal(comp.type, "error");

    store.close();
  });

  test("lists observations for a session", () => {
    const store = createMemoryStore(dbPath());
    store.observe({ id: "obs-1", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", raw: { tool_name: "bash" } });
    store.observe({ id: "obs-2", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "prompt_submit", raw: { prompt: "hello" } });

    const obs = store.listObservations("sess-1");
    assert.equal(obs.length, 2);

    store.close();
  });

  test("searches observations by content", () => {
    const store = createMemoryStore(dbPath());
    store.observe({ id: "obs-1", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", toolName: "bash", toolOutput: "file1.txt", raw: { tool_name: "bash", tool_output: "file1.txt" } });
    store.observe({ id: "obs-2", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", toolName: "bash", toolOutput: "file2.txt", raw: { tool_name: "bash", tool_output: "file2.txt" } });

    const results = store.searchObservations("file", 10);
    assert.ok(results.length >= 1);

    store.close();
  });

  test("updates session observation count", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    store.observe({ id: "obs-1", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", raw: { tool_name: "test" } });

    const session = store.getSession("sess-1");
    assert.equal(session!.observationCount, 1);

    store.close();
  });
});

// ── Consolidation ──────────────────────────────────────────────────────────

describe("consolidate", () => {
  test("creates memories from multiple observations", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    store.observe({ id: "obs-1", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", toolName: "write", toolOutput: "wrote src/foo.ts", raw: { tool_name: "write" } });
    store.observe({ id: "obs-2", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", toolName: "edit", toolOutput: "edited src/foo.ts", raw: { tool_name: "edit" } });
    store.observe({ id: "obs-3", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", toolName: "write", toolOutput: "wrote src/bar.ts", raw: { tool_name: "write" } });

    const memories = store.consolidate("sess-1");
    assert.ok(memories.length >= 1);
    assert.ok(memories.some(m => m.sourceObservationIds?.length! > 1));

    store.close();
  });

  test("returns empty for few observations", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    store.observe({ id: "obs-1", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", raw: { tool_name: "test" } });

    const memories = store.consolidate("sess-1");
    assert.deepStrictEqual(memories, []);

    store.close();
  });
});

// ── Context Injection ──────────────────────────────────────────────────────

describe("getContext", () => {
  test("includes session summary", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test", summary: "Built a login page with auth" });

    const ctx = store.getContext("sess-1");
    assert.ok(ctx.includes("Built a login page with auth"));

    store.close();
  });

  test("includes high-importance observations", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    store.observe({
      id: "obs-err",
      sessionId: "sess-1",
      timestamp: new Date().toISOString(),
      hookType: "post_tool_failure",
      toolOutput: "Error: authentication failed",
      raw: { tool_name: "auth", error: "Error: authentication failed" },
    });

    const ctx = store.getContext("sess-1");
    assert.ok(ctx.includes("authentication failed") || ctx.includes("Error"));

    store.close();
  });

  test("returns empty string when no context available", () => {
    const store = createMemoryStore(dbPath());
    const ctx = store.getContext("nonexistent");
    assert.equal(ctx, "");

    store.close();
  });
});

// ── Hook Adapter Functions ─────────────────────────────────────────────────

describe("hook adapter functions", () => {
  test("remember function works", () => {
    const store = createMemoryStore(dbPath());
    const id = remember(store, "important decision about #caching", {
      type: "pattern",
      strength: 8,
    });

    assert.ok(typeof id === "string");
    const mem = store.get(id)!;
    assert.equal(mem.content, "important decision about #caching");
    assert.equal(mem.strength, 8);
    assert.equal(mem.type, "pattern");
    assert.ok(mem.concepts.includes("caching"));

    store.close();
  });

  test("recall function works", () => {
    const store = createMemoryStore(dbPath());
    store.create("cache strategy decided: LRU with TTL of 5 minutes", { strength: 7, type: "pattern" });

    const result = recall(store, "cache strategy", 5, "text");
    assert.ok(result.includes("LRU"));
    assert.ok(!result.includes("No memories found"));

    store.close();
  });

  test("searchObservations function works", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    store.observe({
      id: "obs-1",
      sessionId: "sess-1",
      timestamp: new Date().toISOString(),
      hookType: "post_tool_use",
      toolName: "bash",
      toolOutput: "Error: database connection timeout",
      raw: { tool_name: "bash", tool_output: "Error: database connection timeout" },
    });

    const result = searchObservations(store, "database", 5);
    assert.ok(result.includes("database connection timeout"));
    assert.ok(!result.includes("No observations found"));

    store.close();
  });

  test("forget function works", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("temporary note", { type: "fact" });

    const result = forget(store, entry.id);
    assert.ok(result.includes("deleted"));
    assert.equal(store.get(entry.id), null);

    store.close();
  });

  test("listMemories function works", () => {
    const store = createMemoryStore(dbPath());
    store.create("memory one", { type: "pattern", strength: 7 });
    store.create("memory two", { type: "bug", strength: 9 });

    const result = listMemories(store, { minStrength: 6 });
    assert.ok(result.includes("memory one"));
    assert.ok(result.includes("memory two"));

    store.close();
  });

  test("consolidate function works", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    store.observe({ id: "obs-1", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", toolName: "write", raw: { tool_name: "write" } });
    store.observe({ id: "obs-2", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", toolName: "write", raw: { tool_name: "write" } });
    store.observe({ id: "obs-3", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", toolName: "write", raw: { tool_name: "write" } });

    const ids = consolidate(store, "sess-1");
    assert.ok(ids.length >= 1);

    store.close();
  });

  test("getContext function works", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test", summary: "Test summary" });

    const ctx = getContext(store, "sess-1");
    assert.ok(ctx.includes("Test summary"));

    store.close();
  });

  test("close does not throw", () => {
    const store = createMemoryStore(dbPath());
    store.create("test", { type: "fact" });
    assert.doesNotThrow(() => store.close());
  });

  test("autoForget function works", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    // Create very old, low-importance observations (error hookType = importance 8, so use post_tool_use = 5)
    const oldDate = new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(); // 10 days ago
    store.observe({ id: "obs-1", sessionId: "sess-1", timestamp: oldDate, hookType: "post_tool_use", raw: { tool_name: "bash" } });
    store.observe({ id: "obs-2", sessionId: "sess-1", timestamp: oldDate, hookType: "post_tool_use", raw: { tool_name: "bash" } });

    // TTL = 5 days, minImportance = 6 (so observations with importance 5 get deleted)
    const result = autoForget(store, { ttlMs: 1000 * 60 * 60 * 24 * 5, minImportance: 6 });
    assert.ok(result.deleted >= 2);

    store.close();
  });

  test("autoTierMemories function works", () => {
    const store = createMemoryStore(dbPath());
    const mem = store.create("test memory", { type: "fact" });
    store.trackAccess(mem.id);

    const tiered = autoTierMemories(store);
    assert.ok(typeof tiered[mem.id] === "string");
    assert.ok(["hot", "warm", "cold", "archived"].includes(tiered[mem.id]!));

    store.close();
  });
});

// ── Dedup ───────────────────────────────────────────────────────────────────

describe("dedup", () => {
  test("detects duplicate observations within window", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    const result1 = store.dedupCheck("sess-1", "bash", { command: "ls -la" });
    assert.equal(result1, false); // first time, not a dup

    store.dedupRecord("sess-1", "bash", { command: "ls -la" });

    const result2 = store.dedupCheck("sess-1", "bash", { command: "ls -la" });
    assert.equal(result2, true); // duplicate

    store.close();
  });

  test("allows different inputs", () => {
    const store = createMemoryStore(dbPath());

    const r1 = store.dedupCheck("sess-1", "bash", { command: "ls -la" });
    const r2 = store.dedupCheck("sess-1", "bash", { command: "pwd" });
    assert.equal(r1, false);
    assert.equal(r2, false); // different input

    store.close();
  });

  test("allows different sessions", () => {
    const store = createMemoryStore(dbPath());

    const r1 = store.dedupCheck("sess-1", "bash", { command: "ls" });
    const r2 = store.dedupCheck("sess-2", "bash", { command: "ls" });
    assert.equal(r1, false);
    assert.equal(r2, false); // different session

    store.close();
  });
});

// ── Sliding Window ──────────────────────────────────────────────────────────

describe("sliding window", () => {
  test("caps observations per session", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    // Add more than the cap (observe already caps at 200, so we need 201+)
    for (let i = 0; i < 250; i++) {
      store.observe({
        id: `obs-${i}`,
        sessionId: "sess-1",
        timestamp: new Date(Date.now() - (250 - i) * 1000).toISOString(),
        hookType: "post_tool_use",
        toolName: "bash",
        toolOutput: `output ${i}`,
        raw: { tool_name: "bash", tool_output: `output ${i}` },
      });
    }

    // observe() already caps at 200, so manual cap returns 0
    const evicted = store.slidingWindowCap("sess-1", 200);
    // After observe caps, the second call should return 0 or a small number

    const obs = store.listObservations("sess-1");
    assert.ok(obs.length <= 200);

    store.close();
  });

  test("does nothing when under cap", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    store.observe({ id: "obs-1", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", raw: { tool_name: "bash" } });
    store.observe({ id: "obs-2", sessionId: "sess-1", timestamp: new Date().toISOString(), hookType: "post_tool_use", raw: { tool_name: "bash" } });

    const evicted = store.slidingWindowCap("sess-1", 200);
    assert.equal(evicted, 0);

    store.close();
  });
});

// ── Access Tracker ──────────────────────────────────────────────────────────

describe("access tracker", () => {
  test("tracks access count", () => {
    const store = createMemoryStore(dbPath());
    const mem = store.create("test memory", { type: "fact", strength: 7 });

    const stats1 = store.getAccessStats(mem.id);
    assert.equal(stats1?.accessCount, 0);

    store.trackAccess(mem.id);
    store.trackAccess(mem.id);

    const stats2 = store.getAccessStats(mem.id);
    assert.equal(stats2?.accessCount, 2);
    assert.ok(stats2?.lastAccessed.length > 0);

    store.close();
  });

  test("returns null for non-existent entity", () => {
    const store = createMemoryStore(dbPath());
    const stats = store.getAccessStats("nonexistent");
    assert.equal(stats, null);

    store.close();
  });
});

// ── Working Memory Tiers ────────────────────────────────────────────────────

describe("working memory tiers", () => {
  test("defaults to cold tier", () => {
    const store = createMemoryStore(dbPath());
    const mem = store.create("test memory", { type: "fact" });

    const tier = store.getWorkingMemoryTier(mem.id);
    assert.equal(tier, "cold");

    store.close();
  });

  test("sets tier explicitly", () => {
    const store = createMemoryStore(dbPath());
    const mem = store.create("test memory", { type: "fact" });

    store.setWorkingMemoryTier(mem.id, "hot");
    assert.equal(store.getWorkingMemoryTier(mem.id), "hot");

    store.setWorkingMemoryTier(mem.id, "warm");
    assert.equal(store.getWorkingMemoryTier(mem.id), "warm");

    store.close();
  });

  test("auto-tier based on access time", () => {
    const store = createMemoryStore(dbPath());

    // Create memory and track recent access (should be "hot")
    const mem1 = store.create("recent memory", { type: "fact" });
    store.trackAccess(mem1.id);

    const tiered = store.autoTierMemories();
    assert.equal(tiered[mem1.id], "hot");

    store.close();
  });
});

// ── Auto-Forget ─────────────────────────────────────────────────────────────

describe("auto-forget", () => {
  test("deletes old low-importance observations", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    // Create a very old, low-importance observation (post_tool_use = importance 5)
    const oldDate = new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString(); // 60 days ago
    store.observe({
      id: "old-obs-1",
      sessionId: "sess-1",
      timestamp: oldDate,
      hookType: "post_tool_use",
      toolName: "bash",
      toolOutput: "old output",
      raw: { tool_name: "bash", tool_output: "old output" },
    });

    // Verify it exists
    const before = store.listObservations("sess-1");
    assert.ok(before.some(o => o.id === "old-obs-1"));

    // Auto-forget: 30 day TTL, minImportance 6 (so importance 5 gets deleted)
    const result = store.autoForget(30 * 24 * 60 * 60 * 1000, 6, 10);
    assert.ok(result.deleted >= 1);

    // Verify deleted
    const after = store.listObservations("sess-1");
    assert.ok(!after.some(o => o.id === "old-obs-1"));

    store.close();
  });

  test("does not delete high-importance observations", () => {
    const store = createMemoryStore(dbPath());
    store.createSession("sess-1", { project: "test" });

    const oldDate = new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString();
    store.observe({
      id: "old-high-importance",
      sessionId: "sess-1",
      timestamp: oldDate,
      hookType: "post_tool_failure",
      toolName: "bash",
      toolOutput: "Error: critical failure",
      raw: { tool_name: "bash", error: "Error: critical failure" },
    });

    const result = store.autoForget(1000 * 60 * 5, 8, 10);
    // Should not have deleted this high-importance observation
    const obs = store.listObservations("sess-1");
    assert.ok(obs.some(o => o.id === "old-high-importance"));

    store.close();
  });
});

// ── Memory Relations ─────────────────────────────────────────────────────

test("relate creates a relation between two memories", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const m1 = store.create("Config pattern: use environment variables");
  const m2 = store.create("Config pattern: prefer .env files");
  const rel = store.relate(m1.id, m2.id, "related_to");
  assert.ok(rel);
  assert.equal(rel.type, "related_to");
  assert.ok(rel.confidence >= 0);
  assert.ok(rel.confidence <= 1);
  assert.equal(rel.sourceId, m1.id);
  assert.equal(rel.targetId, m2.id);
  store.close();
});

test("relate returns null when memory not found", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const m = store.create("Some memory");
  const rel = store.relate(m.id, "non-existent", "related_to");
  assert.equal(rel, null);
  store.close();
});

test("getRelations returns relations for a memory", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const m1 = store.create("Memory one");
  const m2 = store.create("Memory two");
  store.relate(m1.id, m2.id, "supports");
  const rels = store.getRelations(m1.id);
  assert.ok(rels.length >= 1);
  assert.equal(rels[0].targetId, m2.id);
  store.close();
});

test("getRelatedMemories traverses via BFS", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const m1 = store.create("Memory one");
  const m2 = store.create("Memory two");
  const m3 = store.create("Memory three");
  store.relate(m1.id, m2.id, "related_to");
  store.relate(m2.id, m3.id, "related_to");
  const related = store.getRelatedMemories(m1.id, 2);
  assert.ok(related.length >= 1);
  // m2 should be at hop 1
  assert.ok(related.some(r => r.memory.id === m2.id && r.hop === 1));
  // m3 should be at hop 2
  assert.ok(related.some(r => r.memory.id === m3.id && r.hop === 2));
  store.close();
});

test("evolve creates a new version of a memory", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const m1 = store.create("Old content");
  const result = store.evolve(m1.id, "New and improved content", "Updated title");
  assert.ok(result);
  assert.equal(result?.previousId, m1.id);
  assert.ok(result?.memory.id !== m1.id);
  assert.equal(result?.memory.title, "Updated title");
  assert.equal(result?.memory.content, "New and improved content");
  assert.equal(result?.memory.version, 2);
  // Old memory should not be latest (use getAny since get() only returns latest)
  const old = store.getAny(m1.id);
  assert.ok(old);
  assert.equal(old?.isLatest, false);
  // New memory should be latest
  const latest = store.get(result?.memory.id!);
  assert.ok(latest);
  assert.equal(latest?.isLatest, true);
  store.close();
});

test("evolve returns null when memory not found", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const result = store.evolve("non-existent", "new content");
  assert.equal(result, null);
  store.close();
});

test("removeRelation deletes a relation", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const m1 = store.create("Memory one");
  const m2 = store.create("Memory two");
  const rel = store.relate(m1.id, m2.id, "related_to");
  assert.ok(rel);
  const deleted = store.removeRelation(rel.id);
  assert.ok(deleted);
  const rels = store.getRelations(m1.id);
  assert.equal(rels.length, 0);
  store.close();
});

// ── Retention Scoring ──────────────────────────────────────────────────

test("computeRetentionScore calculates a score", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const m = store.create("Architecture pattern: use dependency injection", { type: "architecture" });
  const score = store.computeRetentionScore(m.id);
  assert.ok(score);
  assert.ok(score.score >= 0);
  assert.ok(score.score <= 1);
  assert.equal(score.type, "architecture");
  store.close();
});

test("computeRetentionScore returns null for missing memory", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const score = store.computeRetentionScore("non-existent");
  assert.equal(score, null);
  store.close();
});

test("rescoreAll returns scores for all memories", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  store.create("Architecture pattern 1");
  store.create("Bug fix pattern 2");
  store.create("Fact: database is PostgreSQL");
  const scores = store.rescoreAll();
  assert.ok(scores.length >= 3);
  // Sorted descending by score
  for (let i = 1; i < scores.length; i++) {
    assert.ok(scores[i].score <= scores[i - 1].score);
  }
  store.close();
});

test("listByRetentionScore returns limited scores", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  for (let i = 0; i < 10; i++) {
    store.create(`Memory ${i}`);
  }
  const scores = store.listByRetentionScore({}, 5);
  assert.equal(scores.length, 5);
  store.close();
});

test("architecture type has higher base salience than fact", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const arch = store.create("Architecture: microservices pattern", { type: "architecture" });
  const fact = store.create("Fact: database is PostgreSQL", { type: "fact" });
  const archScore = store.computeRetentionScore(arch.id);
  const factScore = store.computeRetentionScore(fact.id);
  assert.ok(archScore);
  assert.ok(factScore);
  // Architecture (0.9) should score higher than fact (0.5) when no decay
  assert.ok(archScore.score > factScore.score);
  store.close();
});

// ── File Context Index ─────────────────────────────────────────────────

test("getFileContext returns observations mentioning a file", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  store.observe({
    id: "obs-1",
    sessionId: "sess-1",
    timestamp: new Date().toISOString(),
    hookType: "file_read",
    toolName: "read_file",
    toolOutput: "Reading src/app.ts",
    raw: { tool_name: "read_file", file_path: "src/app.ts" },
  });
  const ctx = store.getFileContext("src/app.ts");
  assert.ok(ctx);
  assert.equal(ctx.file, "src/app.ts");
  assert.ok(ctx.observations.length >= 1);
  store.close();
});

test("getFileContext returns null for non-matching file", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  store.observe({
    id: "obs-1",
    sessionId: "sess-1",
    timestamp: new Date().toISOString(),
    hookType: "file_read",
    toolName: "read_file",
    toolOutput: "Reading src/app.ts",
    raw: { tool_name: "read_file", file_path: "src/app.ts" },
  });
  const ctx = store.getFileContext("nonexistent/file.ts");
  assert.equal(ctx, null);
  store.close();
});

test("getFilesContext returns contexts for multiple files", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  store.observe({
    id: "obs-1",
    sessionId: "sess-1",
    timestamp: new Date().toISOString(),
    hookType: "file_read",
    toolName: "read_file",
    toolOutput: "Reading src/app.ts",
    raw: { tool_name: "read_file", file_path: "src/app.ts" },
  });
  const ctxs = store.getFilesContext(["src/app.ts", "nonexistent.ts"]);
  assert.ok(ctxs.length >= 1);
  assert.equal(ctxs[0].file, "src/app.ts");
  store.close();
});

test("rebuildFileIndex counts observations with file refs", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  // No file refs
  store.observe({
    id: "obs-1",
    sessionId: "sess-1",
    timestamp: new Date().toISOString(),
    hookType: "command_run",
    toolName: "bash",
    toolOutput: "echo hello",
    raw: { tool_name: "bash" },
  });
  // With file refs
  store.observe({
    id: "obs-2",
    sessionId: "sess-1",
    timestamp: new Date().toISOString(),
    hookType: "file_read",
    toolName: "read_file",
    toolOutput: "Reading src/app.ts",
    raw: { tool_name: "read_file", file_path: "src/app.ts" },
  });
  const count = store.rebuildFileIndex();
  assert.ok(count >= 1);
  store.close();
});

// ── Export/Import ──────────────────────────────────────────────────────

test("exportData produces a valid ExportData", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  store.create("Memory one");
  store.create("Memory two");
  const data = store.exportData();
  assert.equal(data.version, 1);
  assert.ok(data.exportedAt);
  assert.ok(data.sessions.length >= 1);
  assert.ok(data.memories.length >= 2);
  assert.ok(Array.isArray(data.observations));
  assert.ok(Array.isArray(data.relations));
  store.close();
});

test("importData imports with skip on conflict", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const mem = store.create("Original memory");
  const exportData = store.exportData();

  // Modify the exported data
  const importData = {
    ...exportData,
    memories: ["New memory"].map((c, i) => ({
      ...mem,
      id: `imported-${i}`,
      content: c,
      title: c.slice(0, 200),
    })),
  };

  const result = store.importData(importData as any);
  assert.ok(result.imported >= 1);
  store.close();
});

test("importData imports with update on conflict", () => {
  const store = createMemoryStore(dbPath());
  store.createSession("sess-1", { project: "test" });
  const mem = store.create("Original content");
  const originalId = mem.id;

  // Import with the same session ID (should update)
  const importData = {
    version: 1,
    onConflict: "update",
    sessions: [{
      id: "sess-1",
      project: "updated-project",
      cwd: "/updated/cwd",
      startedAt: mem.createdAt,
      status: "active" as const,
      observationCount: 0,
    }],
    observations: [],
    memories: [],
  } as any;

  const result = store.importData(importData as any);
  assert.ok(result.imported >= 1);

  // Verify session was updated
  const updatedSession = store.getSession("sess-1");
  assert.equal(updatedSession?.project, "updated-project");
  store.close();
});
