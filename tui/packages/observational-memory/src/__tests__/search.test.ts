import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { ExtensionEventBus } from "@logician/agent-core/hooks/extensions";
import { ConsolidationPipeline } from "../consolidation/pipeline.ts";
import { registerConsolidationHooks } from "../integration/hooks.ts";
import {
	createMemorySearchTool,
	createRecallTool,
} from "../integration/tools.ts";
import { searchMemoryStore } from "../retrieval/search.ts";
import { FilePersistence } from "../storage/persistence.ts";
import { MemoryStoreImpl } from "../storage/store.ts";

const observation = {
	id: "aaaaaaaaaaaa",
	content: "The deployment uses a blue green release strategy",
	timestamp: "2026-07-24T00:00:00.000Z",
	relevance: "high" as const,
	sourceEntryIds: ["bbbbbbbbbbbb"],
	tokenCount: 12,
};

function createStore(): MemoryStoreImpl {
	const dir = mkdtempSync(join(tmpdir(), "observational-memory-search-"));
	return new MemoryStoreImpl({
		persistence: new FilePersistence({ path: join(dir, "memory.json") }),
	});
}

void test("memory search ranks matching active observations and excludes drops", () => {
	const store = createStore();
	store.recordObservations(
		[
			observation,
			{
				...observation,
				id: "cccccccccccc",
				content: "Database migrations run before deployment",
				relevance: "medium",
			},
		],
		"source",
	);
	assert.equal(searchMemoryStore(store, "blue green")[0]?.id, "aaaaaaaaaaaa");
	store.recordDrops(["aaaaaaaaaaaa"], "source");
	assert.ok(
		!searchMemoryStore(store, "blue green").some(
			(match) => match.id === "aaaaaaaaaaaa",
		),
	);
});

void test("memory search tool validates queries and bounds results", () => {
	const store = createStore();
	store.recordObservations([observation], "source");
	const search = createMemorySearchTool(store);
	assert.equal(search("").status, "invalid_query");
	assert.equal(search("deployment", 100).matches.length, 1);
});

void test("recall resolves source entries at call time", () => {
	const store = createStore();
	store.recordObservations([observation], "source");
	let sourceContent = "first";
	const recall = createRecallTool({
		memoryStore: store,
		sourceEntries: () => [
			{
				id: "bbbbbbbbbbbb",
				type: "message",
				origin: "user",
				timestamp: "2026-07-24T00:00:00.000Z",
				content: sourceContent,
			},
		],
	});
	sourceContent = "live source evidence";
	const result = recall("aaaaaaaaaaaa");
	assert.equal(result.status, "ok");
	assert.match(result.content ?? "", /live source evidence/);
});

void test("before_agent_start injects only query-relevant bounded memory", async () => {
	const store = createStore();
	store.recordObservations([observation], "source");
	const bus = new ExtensionEventBus();
	const pipeline = new ConsolidationPipeline({ model: "unused", apiKey: "" });
	const unsubscribe = registerConsolidationHooks({
		extensionBus: bus,
		memoryStore: store,
		pipeline,
		options: { memoryContextMaxTokens: 100 },
	});
	const matching = await bus.emit({
		type: "before_agent_start",
		prompt: "How is the blue green deployment configured?",
		systemPrompt: "base",
	});
	assert.match(matching?.systemPrompt ?? "", /observational-memory/);
	assert.match(matching?.systemPrompt ?? "", /aaaaaaaaaaaa/);
	const unrelated = await bus.emit({
		type: "before_agent_start",
		prompt: "Explain typography",
		systemPrompt: "base",
	});
	assert.equal(unrelated, undefined);
	unsubscribe();
});
