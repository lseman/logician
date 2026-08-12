import { describe, test } from "bun:test";
import assert from "node:assert/strict";
import { selectContextCandidates } from "../retrieval/context-selector.js";

function candidate(
	id: string,
	score: number,
	similarityText: string,
	tokens = 10,
	sourceKey = id,
) {
	return { id, score, similarityText, tokens, sourceKey, recency: 0 };
}

describe("selectContextCandidates", () => {
	test("uses budget for novel evidence instead of near-duplicates", () => {
		const selected = selectContextCandidates(
			[
				candidate(
					"retry-a",
					10,
					"authentication retries use exponential backoff",
				),
				candidate(
					"retry-b",
					9.8,
					"authentication retries use exponential backoff policy",
				),
				candidate(
					"breaker",
					9.5,
					"authentication failures open the circuit breaker",
				),
			],
			{ budget: 20 },
		);

		assert.deepEqual(
			selected.map(item => item.id),
			["retry-a", "breaker"],
		);
	});

	test("respects token and item limits", () => {
		const selected = selectContextCandidates(
			[
				candidate("a", 3, "alpha", 8),
				candidate("b", 2, "beta", 8),
				candidate("c", 1, "gamma", 8),
			],
			{ budget: 16, maxItems: 1 },
		);

		assert.equal(selected.length, 1);
		assert.ok(selected.reduce((sum, item) => sum + item.tokens, 0) <= 16);
	});

	test("uses stable IDs to break otherwise equal ties", () => {
		const inputs = [candidate("z", 1, "same"), candidate("a", 1, "same")];
		assert.equal(selectContextCandidates(inputs, { budget: 10 })[0]?.id, "a");
	});
});
