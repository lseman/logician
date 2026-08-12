import { describe, expect, test } from "bun:test";
import {
	diagnoseRetrieval,
	fuseRankedHits,
	selectDiverseHits,
} from "./retrieval.ts";
import type { SearchHit } from "./types.ts";

const hit = (
	id: string,
	text: string,
	documentId: string,
	score = 1,
): SearchHit => ({
	chunk: { id, text, documentId, metadata: {} },
	score,
});

describe("advanced retrieval", () => {
	test("rank fusion rewards independent route agreement", () => {
		const fused = fuseRankedHits([
			{
				name: "original",
				hits: [hit("a", "alpha", "d1"), hit("b", "beta", "d2")],
			},
			{
				name: "rewrite",
				hits: [hit("b", "beta", "d2"), hit("c", "gamma", "d3")],
			},
		]);
		expect(fused[0].chunk.id).toBe("b");
		expect(fused[0].retrievalRoutes).toEqual(["original", "rewrite"]);
	});

	test("diversification prefers complementary evidence", () => {
		const selected = selectDiverseHits(
			[
				hit("a", "alpha beta gamma repeated evidence", "d1", 1),
				hit("b", "alpha beta gamma repeated evidence again", "d1", 0.99),
				hit("c", "independent counterexample with different facts", "d2", 0.9),
			],
			2,
		);
		expect(selected.map(value => value.chunk.id)).toEqual(["a", "c"]);
	});

	test("diagnostics expose empty retrieval as insufficient", () => {
		const diagnostics = diagnoseRetrieval(["query"], [], []);
		expect(diagnostics.insufficientEvidence).toBe(true);
		expect(diagnostics.reasons).toContain("no_candidates");
	});
});
