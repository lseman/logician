import assert from "node:assert/strict";
import { test } from "node:test";
import { FakeBackend } from "./fake-backend.ts";
import { AgentLoop } from "../core/loop.ts";
import {
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
	formatAcceptancePrompt,
	parseAcceptanceReport,
	validateAcceptanceInput,
	stripAcceptanceReport,
} from "../core/acceptance-contract.ts";
import type { AcceptanceConfig } from "../core/types.ts";

function reportFence(report: Record<string, unknown>): string {
	const bt = "`";
	return "\n\n" + bt + bt + bt + "acceptance-report\n" + JSON.stringify(report, null, 2) + "\n" + bt + bt + bt;
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
	const config: AcceptanceConfig = { verify: [{ id: "t1", command: "npm test" }] };
	const resolved = resolveEffectiveAcceptance({ explicit: config });
	assert.equal(resolved.level, "verified");
});

void test("resolve with review returns reviewed", () => {
	const config: AcceptanceConfig = { review: { agent: "reviewer" } };
	const resolved = resolveEffectiveAcceptance({ explicit: config });
	assert.equal(resolved.level, "reviewed");
});

void test("normalize string criteria with global evidence", () => {
	const config: AcceptanceConfig = { criteria: ["fix bug"], evidence: ["changed-files"] };
	const resolved = resolveEffectiveAcceptance({ explicit: config });
	assert.equal(resolved.criteria[0].id, "criterion-1");
	assert.deepEqual(resolved.criteria[0].evidence, ["changed-files"]);
});

void test("normalize gate criteria with custom evidence", () => {
	const config: AcceptanceConfig = { criteria: [{ id: "c1", must: "pass tests", evidence: ["tests-added"] }] };
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
	assert.ok(errors.some((e) => e.includes("foo")));
});

void test("validate accepts minimal config", () => {
	const errors = validateAcceptanceInput({ criteria: ["do something"] });
	assert.equal(errors.length, 0);
});

void test("validate rejects maxFinalizationTurns out of range", () => {
	const errors = validateAcceptanceInput({ criteria: ["do something"], maxFinalizationTurns: 15 });
	assert.ok(errors.some((e) => e.includes("10")));
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
		criteriaSatisfied: [{ id: "c1", status: "satisfied" as const, evidence: "changed 3 files" }],
		changedFiles: ["a.ts", "b.ts"],
		commandsRun: [{ command: "npm test", result: "passed" as const, summary: "all pass" }],
		residualRisks: [],
	};
	const output = "Here is my answer." + reportFence(report);
	const result = parseAcceptanceReport(output);
	assert.ok(result.report);
	assert.equal(result.report!.changedFiles?.length, 2);
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
	const report = { criteriaSatisfied: [{ id: "c1", status: "satisfied" as const, evidence: "done" }] };
	const output = "Answer." + reportFence(report);
	const stripped = stripAcceptanceReport(output);
	assert.ok(!stripped.includes("acceptance-report"));
});

void test("loop with acceptance parses report from output", async () => {
	const config: AcceptanceConfig = { criteria: ["fix the bug"], evidence: ["changed-files"] };
	const report = {
		criteriaSatisfied: [{ id: "criterion-1", status: "satisfied" as const, evidence: "modified file.ts" }],
		changedFiles: ["file.ts"],
		commandsRun: [{ command: "npm test", result: "passed" as const, summary: "all pass" }],
		residualRisks: [],
		noStagedFiles: false,
	};
	const backend = new FakeBackend([
		(msgs, opts) => {
			opts.callbacks?.onTextStart?.();
			if (opts.callbacks?.onDelta) opts.callbacks.onDelta("done");
			opts.callbacks?.onTextEnd?.();
			return { content: "I fixed the bug." + reportFence(report), toolCalls: [], stopReason: "stop" };
		},
	]);
	const loop = new AgentLoop({
		config: {
			baseUrl: "http://fake", model: "fake", systemPrompt: "test",
			acceptance: config, runtimeHooksEnabled: false, proactiveCompactionEnabled: false,
			continuationEnabled: false, maxIterations: 1,
			tools: [{ name: "noop", description: "noop", parameters: { type: "object", properties: {} }, execute: async () => "ok" }],
		},
		backend,
	});
	const messages = await loop.run("fix the bug");
	const ledger = loop.acceptanceLedger;
	assert.ok(ledger);
	assert.equal(ledger.status, "passed");
	assert.ok(ledger.report);
	assert.equal(ledger.config?.level, "checked");
});

void test("loop with verify commands runs them", async () => {
	const config: AcceptanceConfig = { criteria: ["do the thing"], verify: [{ id: "v1", command: "echo hello", cwd: "/tmp" }] };
	const report = {
		criteriaSatisfied: [{ id: "criterion-1", status: "satisfied" as const, evidence: "completed" }],
		commandsRun: [{ command: "echo hello", result: "passed" as const, summary: "hello" }],
		residualRisks: [],
	};
	const backend = new FakeBackend([
		(msgs, opts) => {
			opts.callbacks?.onTextStart?.();
			if (opts.callbacks?.onDelta) opts.callbacks.onDelta("done");
			opts.callbacks?.onTextEnd?.();
			return { content: "Done." + reportFence(report), toolCalls: [], stopReason: "stop" };
		},
	]);
	const loop = new AgentLoop({
		config: {
			baseUrl: "http://fake", model: "fake", systemPrompt: "test",
			acceptance: config, runtimeHooksEnabled: false, proactiveCompactionEnabled: false,
			continuationEnabled: false, maxIterations: 1,
			tools: [{ name: "noop", description: "noop", parameters: { type: "object", properties: {} }, execute: async () => "ok" }],
		},
		backend,
	});
	const messages = await loop.run("do the thing");
	const ledger = loop.acceptanceLedger;
	assert.ok(ledger);
	assert.ok(ledger.verification?.length > 0);
});

void test("loop without acceptance has not-required status", async () => {
	const backend = new FakeBackend([
		(msgs, opts) => {
			opts.callbacks?.onTextStart?.();
			if (opts.callbacks?.onDelta) opts.callbacks.onDelta("done");
			opts.callbacks?.onTextEnd?.();
			return { content: "done", toolCalls: [], stopReason: "stop" };
		},
	]);
	const loop = new AgentLoop({
		config: {
			baseUrl: "http://fake", model: "fake", systemPrompt: "test",
			runtimeHooksEnabled: false, proactiveCompactionEnabled: false,
			continuationEnabled: false, maxIterations: 1,
			tools: [{ name: "noop", description: "noop", parameters: { type: "object", properties: {} }, execute: async () => "ok" }],
		},
		backend,
	});
	const messages = await loop.run("do the thing");
	assert.ok(loop.acceptanceLedger);
	assert.equal(loop.acceptanceLedger!.status, "not-required");
});

void test("loop with acceptance but no report times out in self-review", async () => {
	const config: AcceptanceConfig = { criteria: ["fix the bug"] };
	const backend = new FakeBackend([
		(msgs, opts) => {
			opts.callbacks?.onTextStart?.();
			if (opts.callbacks?.onDelta) opts.callbacks.onDelta("I did the thing.");
			opts.callbacks?.onTextEnd?.();
			return { content: "I did the thing.", toolCalls: [], stopReason: "stop" };
		},
	]);
	const loop = new AgentLoop({
		config: {
			baseUrl: "http://fake", model: "fake", systemPrompt: "test",
			acceptance: config, runtimeHooksEnabled: false, proactiveCompactionEnabled: false,
			continuationEnabled: false, maxIterations: 1,
			tools: [{ name: "noop", description: "noop", parameters: { type: "object", properties: {} }, execute: async () => "ok" }],
		},
		backend,
	});
	const messages = await loop.run("fix the bug");
	const ledger = loop.acceptanceLedger;
	assert.ok(ledger);
	assert.ok(ledger.status === "failed" || ledger.status === "timeout");
});

void test("loop with evidence-only acceptance no report fails or times out", async () => {
	const config: AcceptanceConfig = { evidence: ["changed-files"] };
	const backend = new FakeBackend([
		(msgs, opts) => {
			opts.callbacks?.onTextStart?.();
			if (opts.callbacks?.onDelta) opts.callbacks.onDelta("I did the thing.");
			opts.callbacks?.onTextEnd?.();
			return { content: "I did the thing.", toolCalls: [], stopReason: "stop" };
		},
	]);
	const loop = new AgentLoop({
		config: {
			baseUrl: "http://fake", model: "fake", systemPrompt: "test",
			acceptance: config, runtimeHooksEnabled: false, proactiveCompactionEnabled: false,
			continuationEnabled: false, maxIterations: 1,
			tools: [{ name: "noop", description: "noop", parameters: { type: "object", properties: {} }, execute: async () => "ok" }],
		},
		backend,
	});
	const messages = await loop.run("do the thing");
	const ledger = loop.acceptanceLedger;
	assert.ok(ledger);
	assert.ok(ledger.status === "failed" || ledger.status === "timeout");
});
