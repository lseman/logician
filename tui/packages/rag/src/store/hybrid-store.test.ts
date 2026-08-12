import { afterEach, describe, test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { RAGChunk } from "../types.ts";
import { BM25Scorer, HybridVectorStore } from "./hybrid-store.ts";

const temporaryDirectories: string[] = [];

afterEach(() => {
	for (const directory of temporaryDirectories.splice(0)) {
		rmSync(directory, { recursive: true, force: true });
	}
});

describe("BM25Scorer", () => {
	test("scores only documents containing query terms", () => {
		const scorer = new BM25Scorer();
		scorer.addChunk(1, ["authentication", "retry"]);
		scorer.addChunk(2, ["database", "migration"]);
		scorer.recomputeAvgLen(4);

		assert.deepEqual(
			scorer.topK(["authentication"], 5).map(hit => hit.id),
			[1],
		);
	});
});

describe("HybridVectorStore", () => {
	function createStore(): {
		store: HybridVectorStore;
		restoreEnvironment: () => void;
	} {
		const dataHome = mkdtempSync(join(tmpdir(), "logician-rag-test-"));
		temporaryDirectories.push(dataHome);
		const previousDataHome = process.env.XDG_DATA_HOME;
		process.env.XDG_DATA_HOME = dataHome;
		return {
			store: new HybridVectorStore("/workspace", {
				dbName: `hybrid-${crypto.randomUUID()}`,
				dimension: 2,
			}),
			restoreEnvironment: () => {
				if (previousDataHome === undefined) delete process.env.XDG_DATA_HOME;
				else process.env.XDG_DATA_HOME = previousDataHome;
			},
		};
	}

	test("recovers sparse-only candidates and enforces filters before fusion", async () => {
		const { store, restoreEnvironment } = createStore();
		try {
			const chunks: RAGChunk[] = Array.from({ length: 10 }, (_, index) => ({
				id: `dense-${index}`,
				documentId: `dense-doc-${index}`,
				text:
					index === 0
						? "quasar quasar excluded"
						: `unrelated document ${index}`,
				metadata: { scope: index === 0 ? "excluded" : "allowed" },
				vector: [1, index / 100],
			}));
			chunks.push({
				id: "sparse-target",
				documentId: "sparse-doc",
				text: "quasar lease renewal protocol",
				metadata: { scope: "allowed" },
				vector: [-1, 0],
			});
			await store.add(chunks);

			const hits = await store.searchHybrid("quasar", [1, 0], 1, {
				filter: { scope: "allowed" },
				denseWeight: 0.2,
				sparseWeight: 0.8,
			});

			assert.equal(hits[0]?.chunk.id, "sparse-target");
		} finally {
			store.close();
			restoreEnvironment();
		}
	});

	test("keeps lexical indexes for previously ingested documents", async () => {
		const { store, restoreEnvironment } = createStore();
		try {
			await store.add([
				{
					id: "first:chunk-0",
					documentId: "first",
					text: "quasar renewal protocol",
					metadata: {},
					vector: [-1, 0],
				},
			]);
			await store.add([
				{
					id: "second:chunk-0",
					documentId: "second",
					text: "database migration guide",
					metadata: {},
					vector: [1, 0],
				},
			]);

			const hits = await store.searchHybrid("quasar", [1, 0], 1, {
				denseWeight: 0.1,
				sparseWeight: 0.9,
			});
			assert.equal(hits[0]?.chunk.documentId, "first");
		} finally {
			store.close();
			restoreEnvironment();
		}
	});
});
