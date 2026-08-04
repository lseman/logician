import assert from "node:assert/strict";
import { test } from "node:test";
import { createMemoryStore } from "@logician/memory";
import { createMemoryGetTool } from "../tools/memory-tools.ts";

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
