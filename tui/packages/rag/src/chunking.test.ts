import { describe, expect, test } from "bun:test";
import { parentChildChunk } from "./chunking.ts";

describe("parentChildChunk", () => {
	test("parentOverlap shares child context between adjacent parents", () => {
		const text = Array.from(
			{ length: 20 },
			(_, index) => `sentence-${index.toString().padStart(2, "0")} content.`,
		).join(" ");
		const withoutOverlap = parentChildChunk(text, {
			childSize: 60,
			childOverlap: 0,
			parentSize: 180,
			parentOverlap: 0,
		});
		const withOverlap = parentChildChunk(text, {
			childSize: 60,
			childOverlap: 0,
			parentSize: 180,
			parentOverlap: 60,
		});

		expect(withoutOverlap.parents[0]?.childIds).not.toContain(
			withoutOverlap.parents[1]?.childIds[0],
		);
		expect(withOverlap.parents[0]?.childIds.at(-1)).toBe(
			withOverlap.parents[1]?.childIds[0],
		);
	});
});
