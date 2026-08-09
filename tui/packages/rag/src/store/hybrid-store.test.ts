import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, test } from "node:test";
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
	test("recovers sparse-only candidates and enforces filters before fusion", async () => {
		const dataHome = mkdtempSync(join(tmpdir(), "logician-rag-test-"));
		temporaryDirectories.push(dataHome);
		const previousDataHome = process.env.XDG_DATA_HOME;
		process.env.XDG_DATA_HOME = dataHome;
		const store = new HybridVectorStore("/workspace", {
			dbName: `hybrid-${Date.now()}`,
			dimension: 2,
		});
		try {
			const chunks: RAGChunk[] = Array.from({ length: 10 }, (_, index) => ({
				id: `dense-${index}`,
				documentId: `dense-doc-${index}`,
				text: index === 0 ? "quasar quasar excluded" : `unrelated document ${index}`,
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
			});

			assert.equal(hits[0]?.chunk.id, "sparse-target");
		} finally {
			store.close();
			if (previousDataHome === undefined) delete process.env.XDG_DATA_HOME;
			else process.env.XDG_DATA_HOME = previousDataHome;
		}
	});
});
