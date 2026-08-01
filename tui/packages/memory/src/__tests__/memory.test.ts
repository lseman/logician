import { describe, test } from "node:test";
import assert from "node:assert/strict";
import { createMemoryStore } from "../store.js";
import { remember, recall, forget } from "../hook-adapter.js";

let counter = 0;
function dbPath(): string {
  return `/tmp/logician-memory-test-${++counter}-${Date.now()}.db`;
}

describe("createMemoryStore", () => {
  test("creates and retrieves a memory entry", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("test content", {
      source: "test",
      sessionId: "sess-1",
      importance: 7,
      tags: ["hello", "world"],
    });

    assert.equal(entry.content, "test content");
    assert.equal(entry.importance, 7);
    assert.deepStrictEqual(entry.tags, ["hello", "world"]);
    assert.equal(entry.source, "test");
    assert.equal(entry.sessionId, "sess-1");

    const retrieved = store.get(entry.id)!;
    assert.equal(retrieved.content, "test content");
    assert.equal(retrieved.id, entry.id);

    store.close();
  });

  test("auto-tags from #hashtags", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("fix bug with #auth and #api endpoints", {
      autoTags: true,
    });
    assert.ok(entry.tags.includes("auth"));
    assert.ok(entry.tags.includes("api"));

    store.close();
  });

  test("auto-assigns importance based on keywords", () => {
    const store = createMemoryStore(dbPath());
    const errEntry = store.create("fix error in auth module");
    assert.equal(errEntry.importance, 7);

    const todoEntry = store.create("todo next refactor database layer");
    assert.equal(todoEntry.importance, 4);

    store.close();
  });

  test("lists memories sorted by importance", () => {
    const store = createMemoryStore(dbPath());
    store.create("low priority task", { importance: 1 });
    store.create("critical fix bug now", { importance: 9 });
    store.create("medium priority todo", { importance: 5 });

    const all = store.list();
    assert.equal(all.length, 3);
    assert.ok(all[0].importance >= all[all.length - 1].importance);

    store.close();
  });

  test("filters by source", () => {
    const store = createMemoryStore(dbPath());
    store.create("tool call result", { source: "tool:bash" });
    store.create("manual memory", { source: "manual" });
    store.create("another tool", { source: "tool:bash" });

    const tools = store.list({ source: "tool:bash" });
    assert.equal(tools.length, 2);
    assert.ok(tools.every((m) => m.source === "tool:bash"));

    store.close();
  });

  test("filters by minImportance", () => {
    const store = createMemoryStore(dbPath());
    store.create("low priority", { importance: 2 });
    store.create("high priority fix bug crash", { importance: 9 });
    store.create("medium priority", { importance: 5 });

    const high = store.list({ minImportance: 6 });
    assert.equal(high.length, 1);
    assert.equal(high[0].importance, 9);

    store.close();
  });

  test("text search via LIKE", () => {
    const store = createMemoryStore(dbPath());
    store.create("authentication error in login flow", { source: "test" });
    store.create("database connection timeout issue", { source: "test" });
    store.create("api endpoint returns 404 not found", { source: "test" });

    const auth = store.list({ search: "authentication" });
    assert.ok(auth.length >= 1);
    assert.ok(auth.some((m) => m.content.toLowerCase().includes("authentication")));

    const notFound = store.list({ search: "not found" });
    assert.ok(notFound.length >= 1);

    store.close();
  });

  test("updates a memory entry", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("original content", { source: "test" });

    const updated = store.update(entry.id, {
      content: "updated content",
      importance: 8,
    })!;
    assert.equal(updated.content, "updated content");
    assert.equal(updated.importance, 8);

    const retrieved = store.get(entry.id)!;
    assert.equal(retrieved.content, "updated content");

    store.close();
  });

  test("deletes a memory entry", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("to be deleted", { source: "test" });

    assert.equal(store.delete(entry.id), true);
    assert.equal(store.get(entry.id), null);
    assert.equal(store.delete(entry.id), false);

    store.close();
  });

  test("recall formats as markdown", () => {
    const store = createMemoryStore(dbPath());
    store.create("markdown memory one", { importance: 8, source: "test" });
    store.create("markdown memory two", { importance: 6, source: "api" });

    const md = store.recall({ limit: 10 }, { format: "markdown" });
    assert.ok(md.includes("#"));
    assert.ok(md.includes("[8/10]"));
    assert.ok(md.includes("markdown memory one"));

    store.close();
  });

  test("recall formats as system-prompt", () => {
    const store = createMemoryStore(dbPath());
    store.create("prompt memory content", { importance: 7, source: "auth" });

    const formatted = store.recall({ limit: 10 }, { format: "system-prompt" });
    assert.ok(formatted.includes("# auth [7/10]"));
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

  test("close does not throw", () => {
    const store = createMemoryStore(dbPath());
    store.create("test", { source: "test" });
    assert.doesNotThrow(() => store.close());
  });

  test("remember function works", () => {
    const store = createMemoryStore(dbPath());
    const id = remember(store, "important decision about #caching", "decision", "sess-1", 8);

    assert.ok(typeof id === "string");
    const mem = store.get(id)!;
    assert.equal(mem.content, "important decision about #caching");
    assert.equal(mem.importance, 8);
    assert.ok(mem.tags.includes("caching"));

    store.close();
  });

  test("recall function works", () => {
    const store = createMemoryStore(dbPath());
    store.create("cache strategy decided: LRU with TTL of 5 minutes", { importance: 7, source: "decision" });

    const result = recall(store, "cache strategy", 5, "text");
    assert.ok(result.includes("LRU"));
    assert.ok(!result.includes("No memories found"));

    store.close();
  });

  test("forget function works", () => {
    const store = createMemoryStore(dbPath());
    const entry = store.create("temporary note", { source: "test" });

    const result = forget(store, entry.id);
    assert.ok(result.includes("deleted"));
    assert.equal(store.get(entry.id), null);

    store.close();
  });
});
