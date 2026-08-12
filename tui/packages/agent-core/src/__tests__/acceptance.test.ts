import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	type AcceptanceConfig,
	formatAcceptancePrompt,
	parseAcceptanceReport,
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
	stripAcceptanceReport,
	validateAcceptanceInput,
} from "../agent/guards/acceptance-contract.ts";

function reportFence(report: Record<string, unknown>): string {
	const bt = "`";
	return (
		"\n\n" +
		bt +
		bt +
		bt +
		"acceptance-report\n" +
		JSON.stringify(report, null, 2) +
		"\n" +
		bt +
		bt +
		bt
	);
}

void test("resolve with no config returns none", () => {
	const resolved = resolveEffectiveAcceptance({ explicit: undefined });
	assert.equal(resolved.level, "none");
	assert.equal(resolved.explicit, false);
	assert.equal(resolved.criteria.length, 0);
});

void test("resolve with criteria returns checked", () => {
	const config: AcceptanceConfig = { criteria: ["fix the bug"] };
	const resolved = resolveEffectiveAcceptance({ explicit: config });
	assert.equal(resolved.level, "checked");
	assert.equal(resolved.explicit, true);
	assert.equal(resolved.criteria[0].must, "fix the bug");
	assert.ok(shouldRunAcceptanceFinalization(resolved));
});

void test("resolve with verify returns verified", () => {
	const config: AcceptanceConfig = {
		verify: [{ id: "t1", command: "npm test" }],
	};
	const resolved = resolveEffectiveAcceptance({ explicit: config });
	assert.equal(resolved.level, "verified");
});

void test("resolve with review returns reviewed", () => {
	const config: AcceptanceConfig = { review: { agent: "reviewer" } };
	const resolved = resolveEffectiveAcceptance({ explicit: config });
	assert.equal(resolved.level, "reviewed");
});

void test("normalize string criteria with global evidence", () => {
	const config: AcceptanceConfig = {
		criteria: ["fix bug"],
		evidence: ["changed-files"],
	};
	const resolved = resolveEffectiveAcceptance({ explicit: config });
	assert.equal(resolved.criteria[0].id, "criterion-1");
	assert.deepEqual(resolved.criteria[0].evidence, ["changed-files"]);
});

void test("normalize gate criteria with custom evidence", () => {
	const config: AcceptanceConfig = {
		criteria: [{ id: "c1", must: "pass tests", evidence: ["tests-added"] }],
	};
	const resolved = resolveEffectiveAcceptance({ explicit: config });
	assert.equal(resolved.criteria[0].id, "c1");
	assert.deepEqual(resolved.criteria[0].evidence, ["tests-added"]);
	assert.equal(resolved.criteria[0].severity, "required");
});

void test("validate rejects empty criteria", () => {
	const errors = validateAcceptanceInput({ criteria: [""] });
	assert.ok(errors.length > 0);
});

void test("validate rejects unknown keys", () => {
	const errors = validateAcceptanceInput({ foo: "bar" } as AcceptanceConfig);
	assert.ok(errors.some(e => e.includes("foo")));
});

void test("validate accepts minimal config", () => {
	const errors = validateAcceptanceInput({ criteria: ["do something"] });
	assert.equal(errors.length, 0);
});

void test("validate rejects maxFinalizationTurns out of range", () => {
	const errors = validateAcceptanceInput({
		criteria: ["do something"],
		maxFinalizationTurns: 15,
	});
	assert.ok(errors.some(e => e.includes("10")));
});

void test("format returns empty for none level", () => {
	const resolved = resolveEffectiveAcceptance({ explicit: undefined });
	assert.equal(formatAcceptancePrompt(resolved), "");
});

void test("format includes criteria", () => {
	const config: AcceptanceConfig = { criteria: ["fix the bug"] };
	const resolved = resolveEffectiveAcceptance({ explicit: config });
	const prompt = formatAcceptancePrompt(resolved);
	assert.ok(prompt.includes("Acceptance Contract"));
	assert.ok(prompt.includes("fix the bug"));
	assert.ok(prompt.includes("acceptance-report"));
});

void test("parse accepts well-formed report", () => {
	const report = {
		criteriaSatisfied: [
			{ id: "c1", status: "satisfied" as const, evidence: "changed 3 files" },
		],
		changedFiles: ["a.ts", "b.ts"],
		commandsRun: [
			{ command: "npm test", result: "passed" as const, summary: "all pass" },
		],
		residualRisks: [],
	};
	const output = `Here is my answer.${reportFence(report)}`;
	const result = parseAcceptanceReport(output);
	assert.ok(result.report);
	assert.equal(result.report?.changedFiles?.length, 2);
});

void test("parse rejects malformed report", () => {
	const output = "```acceptance-report\n{bad json\n```";
	const result = parseAcceptanceReport(output);
	assert.ok(result.error);
});

void test("parse rejects missing report", () => {
	const result = parseAcceptanceReport("Just plain text.");
	assert.ok(result.error);
});

void test("strip removes acceptance report", () => {
	const report = {
		criteriaSatisfied: [
			{ id: "c1", status: "satisfied" as const, evidence: "done" },
		],
	};
	const output = `Answer.${reportFence(report)}`;
	const stripped = stripAcceptanceReport(output);
	assert.ok(!stripped.includes("acceptance-report"));
});
