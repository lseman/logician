import { test } from "bun:test";
import assert from "node:assert/strict";
import { resolveOutcome } from "../runtime/tasks/outcome-resolution.ts";

void test("structured done is authoritative", () => {
	assert.deepEqual(
		resolveOutcome({
			declared: { status: "done", summary: "verified", ts: 1 },
			structuredOutcomeRequired: true,
		}),
		{ status: "completed", summary: "verified", source: "structured" },
	);
});

void test("undeclared tool-bearing stop is blocked", () => {
	const decision = resolveOutcome({
		declared: null,
		structuredOutcomeRequired: true,
	});
	assert.equal(decision.status, "blocked");
	assert.equal(decision.source, "runtime");
});
