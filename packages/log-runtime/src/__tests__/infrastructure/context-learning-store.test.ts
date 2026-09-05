import { expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ContextLearningStore } from "../../runtime/context/context-learning-store.ts";

test("context learning persists atomically and is scoped by workspace", () => {
	const root = mkdtempSync(join(tmpdir(), "logician-context-learning-"));
	const first = new ContextLearningStore("/workspace/one", root);
	const same = new ContextLearningStore("/workspace/one", root);
	const other = new ContextLearningStore("/workspace/two", root);
	const state = {
		version: 1 as const,
		sources: {
			memory: { utility: 0.8, uses: 3, terms: {} },
		},
	};

	first.save(state);
	expect(same.path).toBe(first.path);
	expect(other.path).not.toBe(first.path);
	expect(same.load()).toEqual(state);
	expect(other.load()).toBeUndefined();
});
