import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { FilePersistence } from "../persistence.ts";
import { MemoryStoreImpl } from "../store.ts";
import type { Observation } from "../types.ts";

function observation(id: string, content: string): Observation {
	return {
		id,
		content,
		timestamp: "2026-07-11T00:00:00.000Z",
		relevance: "high",
		sourceEntryIds: ["source"],
		tokenCount: 10,
	};
}

void test("store emits only durable, deduplicated observations", () => {
	const dir = mkdtempSync(join(tmpdir(), "memory-store-"));
	try {
		const store = new MemoryStoreImpl({
			persistence: new FilePersistence({ path: join(dir, "memory.json") }),
		});
		const counts: number[] = [];
		store.subscribe((event) => {
			if (event.type === "observations_added") counts.push(event.observations.length);
		});
		store.recordObservations([observation("aaaaaaaaaaaa", "Use immutable state")], "one");
		store.recordObservations([observation("bbbbbbbbbbbb", "  use  immutable STATE ")], "two");
		assert.deepEqual(counts, [1]);
		assert.equal(store.getActiveObservations().length, 1);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

void test("drop tombstones survive persistence reload", () => {
	const dir = mkdtempSync(join(tmpdir(), "memory-store-"));
	const memoryPath = join(dir, "memory.json");
	try {
		const store = new MemoryStoreImpl({
			persistence: new FilePersistence({ path: memoryPath }),
		});
		store.recordObservations([observation("aaaaaaaaaaaa", "Persistent fact")], "one");
		store.recordDrops(["aaaaaaaaaaaa"], "one");
		const restored = new MemoryStoreImpl({
			persistence: new FilePersistence({ path: memoryPath }),
		});
		restored.load();
		assert.equal(restored.isDropped("aaaaaaaaaaaa"), true);
		assert.equal(restored.getActiveObservations().length, 0);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});
