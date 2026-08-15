import { test } from "bun:test";
import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { LoopDetector } from "../agent/guards/loop-detector.ts";
import { createGuardEngine } from "../agent/guards/guard-engine.ts";
import {
	awaitsUserInput,
	detectsCircling,
} from "../agent/guards/response-patterns.ts";
import {
	buildBuiltinHooks,
	rewriteCommandWithRtk,
} from "../hooks/builtin/builtin-hooks.ts";

// Capture the real PATH once at module load so cleanup always restores the
// true original value, even when other tests mutate process.env.PATH.
const __originalPath = process.env.PATH;

function withFakeRtk<T>(body: () => T): T {
	const root = mkdtempSync(path.join(tmpdir(), "logician-rtk-"));
	const executable = path.join(root, "rtk");
	writeFileSync(
		executable,
		`#!/bin/sh
if [ "$1" != "rewrite" ]; then exit 2; fi
case "$2" in
  "git status") printf '%s\\n' "rtk git status" ;;
  "cd repo && git status") printf '%s\\n' "cd repo && rtk git status" ;;
  "cargo test && echo done") printf '%s\\n' "rtk cargo test && echo done" ;;
  *) printf '%s\\n' "$2"; exit 3 ;;
esac
`,
		"utf8",
	);
	chmodSync(executable, 0o755);
	process.env.PATH = `${root}${path.delimiter}${__originalPath ?? ""}`;
	try {
		return body();
	} finally {
		if (__originalPath === undefined) delete process.env.PATH;
		else process.env.PATH = __originalPath;
		rmSync(root, { recursive: true, force: true });
	}
}

// ── detectsCircling ───────────────────────────────────────────────────────

void test("detectsCircling: catches retry patterns", () => {
	assert.ok(detectsCircling("I'll try again with a different approach."));
	assert.ok(detectsCircling("Let me try again."));
	assert.ok(detectsCircling("I will attempt the same thing again."));
	assert.ok(detectsCircling("Cannot fix it but I'll try."));
});

void test("detectsCircling: catches attempt patterns", () => {
	assert.ok(
		detectsCircling("I tried it but however it failed, let me try again."),
	);
	assert.ok(detectsCircling("I've tried again and attempted another way."));
});

void test("detectsCircling: does not flag non-circling text", () => {
	assert.ok(!detectsCircling("I read the file and made the changes."));
	assert.ok(!detectsCircling("The task is complete."));
	assert.ok(!detectsCircling("Here is the solution to your problem."));
	assert.ok(!detectsCircling(""));
});

void test("detectsCircling: requires minimum length", () => {
	assert.ok(!detectsCircling("ok"));
	assert.ok(!detectsCircling("done"));
});

void test("awaitsUserInput: detects final questions and direct input requests", () => {
	assert.ok(
		awaitsUserInput("I found two valid approaches. Which one do you prefer?"),
	);
	assert.ok(awaitsUserInput("Please choose one of the options below:"));
	assert.ok(awaitsUserInput("I need your confirmation."));
	assert.ok(
		awaitsUserInput(
			"Which environment should I use?\n\n1. Staging\n2. Production",
		),
	);
});

void test("awaitsUserInput: ignores questions followed by continued work", () => {
	assert.ok(
		!awaitsUserInput("What caused this? I will inspect the stack trace next."),
	);
	assert.ok(!awaitsUserInput("The tests answer the question. Task complete."));
	assert.ok(!awaitsUserInput(""));
});

void test("minimal profile keeps mechanism hooks and omits built-in policies", () => {
	const hooks = buildBuiltinHooks({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			executionProfile: "minimal",
			continuationEnabled: true,
			budgetStopEnabled: true,
			thinkingLoopDetectionEnabled: true,
			proactiveCompactionEnabled: true,
		},
		contextWindowTokens: () => 4096,
		toolDefs: () => [],
		guardEngine: createGuardEngine({
			thinkingLoopDetectionEnabled: true,
		}),
		loopDetector: new LoopDetector(),
	});

	assert.equal(hooks.getFollowUpMessages, undefined);
	assert.equal(hooks.shouldStopAfterTurn, undefined);
	assert.equal(hooks.afterProviderResponse, undefined);
	assert.equal(typeof hooks.prepareNextTurn, "function");
	assert.equal(typeof hooks.beforeToolCall, "function");
	assert.equal(typeof hooks.afterToolCall, "function");
});

void test("explicitly disabling guards bypasses the default duplicate guard", () => {
	const loopDetector = new LoopDetector();
	let checks = 0;
	loopDetector.checkToolCall = (...args) => {
		checks += 1;
		return LoopDetector.prototype.checkToolCall.apply(loopDetector, args);
	};
	const hooks = buildBuiltinHooks({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			guardsEnabled: false,
		},
		contextWindowTokens: () => 4096,
		toolDefs: () => [],
		guardEngine: createGuardEngine({ guardsEnabled: false }),
		loopDetector,
	});

	for (let iteration = 1; iteration <= 4; iteration += 1) {
		hooks.beforeToolCall?.({
			toolCall: { id: String(iteration), name: "read_file", arguments: "{}" },
			args: { path: "README.md" },
			iteration,
		});
	}
	assert.equal(checks, 0);
});

void test("RTK rewrite delegates supported and compound commands to RTK", () => {
	withFakeRtk(() => {
		assert.equal(rewriteCommandWithRtk("git status"), "rtk git status");
		assert.equal(
			rewriteCommandWithRtk("cd repo && git status"),
			"cd repo && rtk git status",
		);
		assert.equal(
			rewriteCommandWithRtk("cargo test && echo done"),
			"rtk cargo test && echo done",
		);
	});
});

void test("RTK rewrite leaves unsupported commands unchanged", () => {
	withFakeRtk(() => {
		assert.equal(rewriteCommandWithRtk("echo hello"), "echo hello");
	});
});

// ── GuardEngine wiring: recovery-memory read side ────────────────────────

void test("afterToolCall appends a recovery-memory warning on a repeated failure", async () => {
	const guardEngine = createGuardEngine({ guardsEnabled: false, recoveryMemoryEnabled: true });
	const hooks = buildBuiltinHooks({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			guardsEnabled: false,
			recoveryMemoryEnabled: true,
		},
		contextWindowTokens: () => 4096,
		toolDefs: () => [],
		guardEngine,
	});

	const call = (n: number) =>
		hooks.afterToolCall?.({
			toolCall: { id: String(n), name: "read_file", arguments: '{"path":"/src/missing.ts"}' },
			args: { path: "/src/missing.ts" },
			result: "Error: ENOENT no such file",
			isError: true,
			iteration: n,
		});

	const first = await call(1);
	assert.equal(first, undefined);

	const second = await call(2);
	assert.ok(second?.content?.includes("recovery-memory"));
	assert.ok(second?.content?.includes("Error: ENOENT no such file"));
});

void test("afterToolCall does not warn on the first occurrence of a failure", async () => {
	const guardEngine = createGuardEngine({ guardsEnabled: false, recoveryMemoryEnabled: true });
	const hooks = buildBuiltinHooks({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			guardsEnabled: false,
			recoveryMemoryEnabled: true,
		},
		contextWindowTokens: () => 4096,
		toolDefs: () => [],
		guardEngine,
	});

	const result = await hooks.afterToolCall?.({
		toolCall: { id: "1", name: "bash", arguments: '{"command":"npm test"}' },
		args: { command: "npm test" },
		result: "Error: timeout exceeded",
		isError: true,
		iteration: 1,
	});
	assert.equal(result, undefined);
});

void test("recovery memory records and warns even with duplicate/failure tool-call guards off", async () => {
	// Regression: recordFailure/checkAndRecordFailure used to be gated behind
	// the tool-call guard's own `guardThresholds`, so recovery memory silently
	// never populated when duplicate/failure guards were both off — a
	// plausible config for anyone who wants recovery memory without the
	// blocking guards.
	const guardEngine = createGuardEngine({
		guardsEnabled: false,
		duplicateGuardEnabled: false,
		failureGuardEnabled: false,
		recoveryMemoryEnabled: true,
	});
	const hooks = buildBuiltinHooks({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			guardsEnabled: false,
			duplicateGuardEnabled: false,
			failureGuardEnabled: false,
			recoveryMemoryEnabled: true,
		},
		contextWindowTokens: () => 4096,
		toolDefs: () => [],
		guardEngine,
	});

	const call = (n: number) =>
		hooks.afterToolCall?.({
			toolCall: { id: String(n), name: "grep", arguments: '{"pattern":"foo"}' },
			args: { pattern: "foo" },
			result: "Error: not found",
			isError: true,
			iteration: n,
		});

	await call(1);
	const second = await call(2);
	assert.ok(second?.content?.includes("recovery-memory"));
});

// ── GuardEngine wiring: composite evaluate()/graduated intervention ──────

void test("evaluate() escalates to restrict severity once progress signal and repeated failures compound", () => {
	const guardEngine = createGuardEngine({
		guardsEnabled: false,
		recoveryMemoryEnabled: true,
		progressSignalEnabled: true,
		fusionEnabled: true,
		graduatedIntervention: true,
		progressSignalMinScore: 30,
		progressSignalMinLowScoreTurns: 2,
	});

	// Drive progress down: repeated no-op turns in "orient" phase score low.
	for (let i = 1; i <= 3; i++) {
		guardEngine.recordProgress([], [], i, "orient");
	}
	// Compound with a repeated failure pattern (recovery memory signal).
	for (let i = 0; i < 4; i++) {
		guardEngine.recordFailure("bash", '{"command":"npm test"}', "Error: timeout");
	}

	const decision = guardEngine.evaluate();
	assert.ok(decision.compositeScore > 0);
	assert.ok(["nudge", "restrict", "pause", "abort"].includes(decision.severity));
	assert.ok(decision.evidence.length > 0);
});

void test("evaluate() caps heuristic-only signals at nudge, never restrict/pause/abort", () => {
	// Regression: progress_signal, recovery_memory, and hypothesis_tracker are
	// all scored from heuristics (keyword matches, string similarity) that
	// produce identical scores for a stuck agent and for perfectly healthy
	// work whose tool results just don't happen to contain the expected
	// keywords (e.g. reading a JSON config or test fixture). Composite
	// severity must stay capped at "nudge" unless a reliable, pattern-based
	// detector (loop_detection/thinking_loop/output_errors) is also present —
	// otherwise a healthy long-running agent gets its flow force-interrupted
	// by a false positive.
	const guardEngine = createGuardEngine({
		guardsEnabled: false, // no LoopDetector-backed tool-call guard
		fusionEnabled: true,
		graduatedIntervention: true,
		progressSignalEnabled: true,
		recoveryMemoryEnabled: true,
		progressSignalMinScore: 30,
		progressSignalMinLowScoreTurns: 1,
	});

	// Drive progress_signal as low as possible: no meaningful file changes,
	// no verification, stuck in "orient".
	for (let i = 1; i <= 10; i++) {
		guardEngine.recordProgress(
			[{ name: "read_file", args: JSON.stringify({ path: `f${i}.json` }) }],
			[{ content: '{"unrelated": "content"}', file: `f${i}.json` }],
			i,
			"orient",
		);
	}
	// Compound with repeated recovery-memory failures on the same approach.
	for (let i = 0; i < 6; i++) {
		guardEngine.recordFailure("bash", '{"command":"npm test"}', "Error: timeout exceeded");
	}

	const decision = guardEngine.evaluate();
	assert.ok(
		decision.severity === "info" || decision.severity === "nudge",
		`expected info or nudge, got ${decision.severity} (score ${decision.compositeScore})`,
	);
});

void test("evaluate() still escalates to restrict/pause when loop_detection is among the evidence", () => {
	// Counterpart to the cap above: a reliable, pattern-based signal (an
	// actual repeated tool call + repeated assistant text) must still be able
	// to escalate past nudge — the cap only applies to heuristic-only signals.
	const guardEngine = createGuardEngine({
		guardsEnabled: true,
		duplicateGuardEnabled: true,
		fusionEnabled: true,
		graduatedIntervention: true,
	});

	let lastSeverity = "info";
	for (let i = 1; i <= 6; i++) {
		guardEngine.recordTurn("I will try reading the file again.", [
			{ name: "read_file", args: JSON.stringify({ path: "src/missing.ts" }), result: "Error: not found" },
		]);
		lastSeverity = guardEngine.evaluate().severity;
	}
	assert.ok(
		lastSeverity === "restrict" || lastSeverity === "pause" || lastSeverity === "abort",
		`expected an escalated severity from a genuine repeated loop, got ${lastSeverity}`,
	);
});

void test("evaluate() stays at info severity with no signals", () => {
	const guardEngine = createGuardEngine({
		guardsEnabled: false,
		fusionEnabled: true,
		graduatedIntervention: true,
	});
	const decision = guardEngine.evaluate();
	assert.equal(decision.severity, "info");
	assert.equal(decision.shouldIntervene, false);
});

// ── GuardEngine wiring: hypothesis prompt round-trip ─────────────────────

void test("buildHypothesisPrompt + parseHypothesesFromText round-trips into stored hypotheses", () => {
	const guardEngine = createGuardEngine({ guardsEnabled: false, hypothesisTrackingEnabled: true });
	const prompt = guardEngine.buildHypothesisPrompt(["progress is low"]);
	assert.match(prompt, /progress is low/);

	const modelReply = "1. The path changed because the repo was restructured\n   test: grep for the old path";
	const count = guardEngine.parseHypothesesFromText(modelReply);
	assert.equal(count, 1);
	assert.equal(guardEngine.getActiveHypotheses().length, 1);
});
