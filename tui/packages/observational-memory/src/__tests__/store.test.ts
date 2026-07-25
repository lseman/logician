import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { FilePersistence } from "../storage/persistence.ts";
import { MemoryStoreImpl } from "../storage/store.ts";
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
		store.flush();
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

void test("corrupt primary memory recovers from the last atomic backup", () => {
	const dir = mkdtempSync(join(tmpdir(), "memory-store-"));
	const memoryPath = join(dir, "memory.json");
	try {
		const store = new MemoryStoreImpl({
			persistence: new FilePersistence({ path: memoryPath }),
		});
		store.recordObservations([observation("aaaaaaaaaaaa", "First durable fact")], "one");
		store.flush();
		store.recordObservations([observation("bbbbbbbbbbbb", "Second durable fact")], "two");
		store.flush();
		writeFileSync(memoryPath, "{truncated", "utf8");

		const restored = new MemoryStoreImpl({
			persistence: new FilePersistence({ path: memoryPath }),
		});
		restored.load();
		assert.deepEqual(
			restored.getActiveObservations().map((item) => item.id),
			["aaaaaaaaaaaa"],
		);
		assert.equal(restored.getDiagnostics().recoveredFromBackup, true);
		assert.match(
			restored.getDiagnostics().lastPersistenceError ?? "",
			/JSON|Unexpected|position/i,
		);
		assert.equal(
			readdirSync(dir).some((entry) => entry.includes(".tmp-")),
			false,
		);
		restored.flush();
		assert.equal(restored.getDiagnostics().recoveredFromBackup, undefined);
		const repaired = new MemoryStoreImpl({
			persistence: new FilePersistence({ path: memoryPath }),
		});
		repaired.load();
		assert.equal(repaired.getDiagnostics().lastPersistenceError, undefined);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});
