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

void test("undeclared tool-bearing stop completes with warning", () => {
	const decision = resolveOutcome({
		declared: null,
		structuredOutcomeRequired: true,
	});
	assert.equal(decision.status, "completed");
	assert.equal(decision.source, "runtime");
	assert.ok(decision.summary?.includes("without a declared task outcome"));
});
