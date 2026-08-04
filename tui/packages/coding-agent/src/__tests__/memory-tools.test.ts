import assert from "node:assert/strict";
import { test } from "node:test";
import { createMemoryStore } from "@logician/memory";
import { createMemoryGetTool, createMemorySearchTool } from "../tools/memory-tools.ts";

void test("memory_get expands selected IDs and reports missing entries", async () => {
	const store = createMemoryStore(`/tmp/logician-memory-tool-${process.pid}-${Date.now()}.db`);
	store.setCurrentWorkspace("/workspace");
	const memory = store.create("Retries use bounded exponential backoff", { strength: 8 });
	const tool = createMemoryGetTool(() => store);
	const result = await tool.execute({ ids: [memory.id, "missing"] }, { cwd: "/workspace" });
	const text = typeof result === "string" ? result : result.content;
	assert.match(String(text), /bounded exponential backoff/);
	assert.match(String(text), /Missing or out of scope: missing/);
	store.close();
});

void test("memory_search returns compact IDs that can be expanded", async () => {
	const store = createMemoryStore(`/tmp/logician-memory-search-tool-${process.pid}-${Date.now()}.db`);
	store.setCurrentWorkspace("/workspace");
	const memory = store.create("Parser delimiters preserve escaped separators", { strength: 8 });
	store.update(memory.id, { title: "Delimiter convention" });
	const tool = createMemorySearchTool(() => store);
	const result = await tool.execute({ query: "parser delimiters", limit: 5 }, { cwd: "/workspace" });
	const text = typeof result === "string" ? result : result.content;
	assert.match(String(text), new RegExp(memory.id));
	assert.match(String(text), /Delimiter convention/);
	assert.match(String(text), /memory_get/);
	store.close();
});
