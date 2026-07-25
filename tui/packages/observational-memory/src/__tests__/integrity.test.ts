import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { ExtensionEventBus } from "@logician/agent-core/hooks/extensions";
import {
	ConsolidationPipeline,
	maxDropCountForPool,
	selectDropCandidates,
} from "../consolidation/pipeline.ts";
import { parseObservations } from "../consolidation/observer.ts";
import { parseReflections } from "../consolidation/reflector.ts";
import { registerConsolidationHooks } from "../integration/hooks.ts";
import { searchMemoryStore } from "../retrieval/search.ts";
import { FilePersistence } from "../storage/persistence.ts";
import { MemoryStoreImpl } from "../storage/store.ts";
import type { Observation, Reflection } from "../types.ts";

function observation(
	id: string,
	content: string,
	options: Partial<Observation> = {},
): Observation {
	return {
		id,
		content,
		timestamp: "2026-07-24T00:00:00.000Z",
		relevance: "medium",
		sourceEntryIds: ["source-1"],
		tokenCount: 10,
		...options,
	};
}

function createStore(): { store: MemoryStoreImpl; dir: string } {
	const dir = mkdtempSync(join(tmpdir(), "observational-memory-integrity-"));
	return {
		dir,
		store: new MemoryStoreImpl({
			persistence: new FilePersistence({ path: join(dir, "memory.json") }),
			observationsPoolTargetTokens: 10,
		}),
	};
}

void test("observer computes IDs and token counts locally", () => {
	const parsed = parseObservations(
		{
			observations: [
				{
					id: "ffffffffffff",
					content: "  Keep   IDs local  ",
					timestamp: "2026-07-24T00:00:00.000Z",
					relevance: "high",
					sourceEntryIds: ["source-1", "invented"],
					tokenCount: 999,
				},
			],
		},
		["source-1"],
	);
	assert.equal(parsed?.length, 1);
	assert.notEqual(parsed?.[0]?.id, "ffffffffffff");
	assert.equal(parsed?.[0]?.content, "Keep IDs local");
	assert.deepEqual(parsed?.[0]?.sourceEntryIds, ["source-1"]);
	assert.notEqual(parsed?.[0]?.tokenCount, 999);
});

void test("reflector rejects invented support IDs and computes record fields", () => {
	assert.equal(
		parseReflections(
			{
				reflections: [
					{
						content: "A durable project preference",
						supportingObservationIds: ["invented-id"],
					},
				],
			},
			["aaaaaaaaaaaa"],
		)?.length ?? 0,
		0,
	);
	const accepted = parseReflections(
		{
			reflections: [
				{
					content: "A durable project preference",
					supportingObservationIds: ["aaaaaaaaaaaa"],
				},
			],
		},
		["aaaaaaaaaaaa"],
	);
	assert.equal(accepted?.length, 1);
	assert.match(accepted?.[0]?.id ?? "", /^[a-f0-9]{12}$/);
	assert.ok((accepted?.[0]?.tokenCount ?? 0) > 0);
});

void test("store rejects dangling reflection provenance", () => {
	const { store, dir } = createStore();
	try {
		store.recordReflections(
			[
				{
					id: "bbbbbbbbbbbb",
					content: "Unsupported reflection",
					supportingObservationIds: ["aaaaaaaaaaaa"],
					tokenCount: 4,
				},
			],
			"source-1",
		);
		assert.equal(store.getReflections().length, 0);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

void test("drop selection validates IDs and enforces pool pressure bound", () => {
	const observations = [
		observation("aaaaaaaaaaaa", "Old low-value detail", {
			relevance: "low",
			timestamp: "2026-01-01T00:00:00.000Z",
		}),
		observation("bbbbbbbbbbbb", "Current critical constraint", {
			relevance: "critical",
			timestamp: "2026-07-24T00:00:00.000Z",
		}),
	];
	const maxDrops = maxDropCountForPool(observations, 10);
	assert.equal(maxDrops, 1);
	assert.deepEqual(
		selectDropCandidates(
			["invented-id", "bbbbbbbbbbbb", "aaaaaaaaaaaa"],
			observations,
			[],
			maxDrops,
		),
		["aaaaaaaaaaaa"],
	);
});

void test("knowledge graph expands a reflection match to supporting evidence", () => {
	const { store, dir } = createStore();
	try {
		const source = observation(
			"aaaaaaaaaaaa",
			"Production releases use two alternating environments.",
		);
		const reflection: Reflection = {
			id: "bbbbbbbbbbbb",
			content: "The deployment convention is blue-green.",
			supportingObservationIds: [source.id],
			tokenCount: 8,
		};
		store.recordObservations([source], "source-1");
		store.recordReflections([reflection], "source-1");
		const matches = searchMemoryStore(store, "deployment convention");
		assert.equal(matches[0]?.id, reflection.id);
		assert.ok(matches.some((item) => item.id === source.id));
		assert.equal(store.fold().knowledgeGraph?.edges.length, 1);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

void test("durable progress, diagnostics, and graph survive reload", () => {
	const { store, dir } = createStore();
	const memoryPath = join(dir, "memory.json");
	try {
		const source = observation("aaaaaaaaaaaa", "Persist the memory graph.");
		store.recordObservations([source], "source-1");
		store.setProgress({ observationCoverageId: "source-1" });
		store.setDiagnostics({
			lastStage: "observer",
			lastRunAt: "2026-07-24T00:00:00.000Z",
		});
		store.flush();

		const restored = new MemoryStoreImpl({
			persistence: new FilePersistence({ path: memoryPath }),
		});
		restored.load();
		assert.equal(restored.getProgress().observationCoverageId, "source-1");
		assert.equal(restored.getDiagnostics().lastStage, "observer");
		assert.equal(restored.fold().knowledgeGraph?.nodes.length, 1);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

void test("pipeline cancellation aborts an in-flight worker", async () => {
	const originalFetch = globalThis.fetch;
	let markFetchStarted: (() => void) | undefined;
	const fetchStarted = new Promise<void>((resolve) => {
		markFetchStarted = resolve;
	});
	globalThis.fetch = ((
		_input: Parameters<typeof fetch>[0],
		init?: Parameters<typeof fetch>[1],
	) => {
		markFetchStarted?.();
		return new Promise<Response>((_resolve, reject) => {
			const signal = init?.signal;
			signal?.addEventListener(
				"abort",
				() => reject(new DOMException("Aborted", "AbortError")),
				{ once: true },
			);
		});
	}) as unknown as typeof fetch;
	const pipeline = new ConsolidationPipeline({ model: "test", apiKey: "" });
	try {
		const run = pipeline.maybeLaunch({
			observeDue: true,
			reflectDue: false,
			observations: [],
			reflections: [],
			sourceEntries: [
				{ id: "source-1", role: "user", content: "remember this" },
			],
		});
		await fetchStarted;
		pipeline.cancel();
		await run;
		assert.equal(pipeline.getStatus().inFlight, false);
		assert.equal(pipeline.getStatus().lastError, undefined);
	} finally {
		globalThis.fetch = originalFetch;
	}
});

void test("empty observer output does not advance durable source coverage", async () => {
	const { store, dir } = createStore();
	const originalFetch = globalThis.fetch;
	let calls = 0;
	globalThis.fetch = (async () => {
		calls++;
		const observations =
			calls === 1
				? []
				: [
						{
							content: "The project retains memory in its working directory.",
							timestamp: "2026-07-24T00:00:00.000Z",
							relevance: "high",
							sourceEntryIds: ["source-1"],
						},
					];
		return new Response(
			JSON.stringify({
				choices: [
					{
						message: {
							tool_calls: [
								{
									function: {
										name: "record_observations",
										arguments: JSON.stringify({ observations }),
									},
								},
							],
						},
					},
				],
			}),
			{ status: 200, headers: { "content-type": "application/json" } },
		);
	}) as unknown as typeof fetch;
	const pipeline = new ConsolidationPipeline({ model: "test", apiKey: "" });
	const bus = new ExtensionEventBus();
	const unsubscribe = registerConsolidationHooks({
		extensionBus: bus,
		memoryStore: store,
		pipeline,
		getSourceEntries: () => [
			{
				id: "source-1",
				role: "user",
				content: "Remember the project-local memory decision.",
				tokenCount: 2,
			},
		],
		options: { observeAfterTokens: 1, reflectAfterTokens: 100 },
	});
	try {
		await bus.emit({
			type: "turn_end",
			turnIndex: 0,
			stopReason: "stop",
			message: { role: "assistant", content: "done" },
			toolResults: [],
		});
		await waitFor(() => calls === 1 && !pipeline.getStatus().inFlight);
		assert.equal(store.getProgress().observationCoverageId, undefined);

		await bus.emit({
			type: "turn_end",
			turnIndex: 1,
			stopReason: "stop",
			message: { role: "assistant", content: "done" },
			toolResults: [],
		});
		await waitFor(() => store.getActiveObservations().length === 1);
		assert.equal(store.getProgress().observationCoverageId, "source-1");
	} finally {
		unsubscribe();
		globalThis.fetch = originalFetch;
		rmSync(dir, { recursive: true, force: true });
	}
});

async function waitFor(predicate: () => boolean): Promise<void> {
	for (let attempts = 0; attempts < 100; attempts++) {
		if (predicate()) return;
		await new Promise<void>((resolve) => setImmediate(resolve));
	}
	throw new Error("condition was not reached");
}
