import { test } from "bun:test";
import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { BudgetTracker } from "../../../core/hooks/builtin/budget.ts";
import {
	buildBuiltinHooks,
	COMPACTION_COOLDOWN_TURNS,
	rewriteCommandWithRtk,
} from "../../../core/hooks/builtin/builtin-hooks.ts";
import { HarnessInterventionController } from "../../../core/policy/intervention-controller.ts";
import { LoopDetector } from "../../../core/guards/loop-detector.ts";
import { awaitsUserInput } from "../../../core/guards/response-patterns.ts";

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

// ── awaitsUserInput: detects final questions and direct input requests ──────

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
			proactiveCompactionEnabled: true,
		},
		contextWindowTokens: () => 4096,
		toolDefs: () => [],
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

// ── beforeToolCall/afterToolCall: duplicate + failure-loop guards ─────────

void test("afterToolCall records failures against the LoopDetector when guards are armed", async () => {
	const loopDetector = new LoopDetector({ failureThreshold: 2 });
	const hooks = buildBuiltinHooks({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			guardsEnabled: true,
			failureGuardEnabled: true,
			executionProfile: "autonomous",
		},
		contextWindowTokens: () => 4096,
		toolDefs: () => [],
		loopDetector,
	});

	const call = (n: number) =>
		hooks.afterToolCall?.({
			toolCall: {
				id: String(n),
				name: "bash",
				arguments: '{"command":"npm test"}',
			},
			args: { command: "npm test" },
			result: "Error: timeout exceeded",
			isError: true,
			iteration: n,
		});

	await call(1);
	await call(2);

	const decision = hooks.beforeToolCall?.({
		toolCall: { id: "3", name: "bash", arguments: '{"command":"npm test"}' },
		args: { command: "npm test" },
		iteration: 3,
	});
	assert.equal((await decision)?.isError, true);
});

// ── Cross-rebuild state: harness rebuilds the hooks object every loop
// iteration (withExtensionRuntime), so interventions/budget/compaction-cooldown
// state must be threaded through explicitly or it silently resets every call.

void test("compaction cooldown carries its last-fired iteration across rebuilds when the cooldown box is shared", async () => {
	const config = {
		baseUrl: "http://fake",
		model: "fake",
		executionProfile: "autonomous" as const,
		proactiveCompactionEnabled: true,
	};
	const compactionCooldown = { lastTurn: -COMPACTION_COOLDOWN_TURNS };
	const build = () =>
		buildBuiltinHooks({
			config,
			contextWindowTokens: () => 100,
			toolDefs: () => [],
			loopDetector: new LoopDetector(),
			compactionCooldown,
		});
	const messages = [{ role: "user" as const, content: "hello" }];

	// Iteration 1: cooldown passes (1 - (-3) = 4 >= 3), records lastTurn = 1.
	await build().prepareNextTurn?.({
		messages,
		iteration: 1,
		hadToolCalls: false,
	});
	assert.equal(compactionCooldown.lastTurn, 1);

	// Iteration 2, a fresh rebuild (simulating refreshNextTurnConfig): still
	// within the 3-turn cooldown (2 - 1 = 1 < 3), so lastTurn must NOT advance.
	await build().prepareNextTurn?.({
		messages,
		iteration: 2,
		hadToolCalls: false,
	});
	assert.equal(
		compactionCooldown.lastTurn,
		1,
		"cooldown should still be in effect one iteration later",
	);

	// Iteration 4: now past the cooldown (4 - 1 = 3 >= 3) — fires and advances.
	await build().prepareNextTurn?.({
		messages,
		iteration: 4,
		hadToolCalls: false,
	});
	assert.equal(compactionCooldown.lastTurn, 4);
});

void test("compaction cooldown resets every rebuild when the cooldown box is NOT shared (documents the bug the fix prevents)", async () => {
	const config = {
		baseUrl: "http://fake",
		model: "fake",
		executionProfile: "autonomous" as const,
		proactiveCompactionEnabled: true,
	};
	const messages = [{ role: "user" as const, content: "hello" }];
	// Each call below builds its own fresh `{ lastTurn: -COMPACTION_COOLDOWN_TURNS }`
	// box, mimicking what buildBuiltinHooks falls back to when no
	// compactionCooldown is passed — the pre-fix bug where a per-iteration
	// hooks rebuild (withExtensionRuntime, via refreshNextTurnConfig)
	// silently resets the cooldown instead of carrying it forward.
	const runOnFreshBox = async (iteration: number) => {
		const cooldown = { lastTurn: -COMPACTION_COOLDOWN_TURNS };
		await buildBuiltinHooks({
			config,
			contextWindowTokens: () => 100,
			toolDefs: () => [],
			loopDetector: new LoopDetector(),
			compactionCooldown: cooldown,
		}).prepareNextTurn?.({ messages, iteration, hadToolCalls: false });
		return cooldown.lastTurn;
	};

	// If the cooldown carried over (as the shared-box test above proves it
	// does when shared), iteration 2 would still be gated (2 - 1 = 1 < 3)
	// and lastTurn would stay at 1. Each call here starts its own box
	// instead, so both see `iteration - (-3) >= 3`, pass the gate, and
	// record their own iteration — proving the cooldown never survives
	// a rebuild unless the box itself is threaded through.
	const lastTurnAfterIteration1 = await runOnFreshBox(1);
	const lastTurnAfterIteration2 = await runOnFreshBox(2);
	assert.equal(lastTurnAfterIteration1, 1);
	assert.equal(
		lastTurnAfterIteration2,
		2,
		"a fresh box every rebuild means iteration 2 also looks like the cooldown just started",
	);
});

void test("budget-stop tracks consecutive low-progress turns across rebuilds when the tracker is shared", () => {
	const config = {
		baseUrl: "http://fake",
		model: "fake",
		executionProfile: "autonomous" as const,
		budgetStopEnabled: true,
	};
	const budget = new BudgetTracker({
		diminishingFloor: 500,
		minContinuations: 1,
	});
	const build = () =>
		buildBuiltinHooks({
			config,
			contextWindowTokens: () => 100_000,
			toolDefs: () => [],
			loopDetector: new LoopDetector(),
			budget,
		});

	// Same token count each call → zero delta both times → stalls on the 2nd.
	const messages = [{ role: "user" as const, content: "x" }];
	const first = build().shouldStopAfterTurn?.({
		messages,
		iteration: 1,
		hadToolCalls: false,
	});
	const second = build().shouldStopAfterTurn?.({
		messages,
		iteration: 2,
		hadToolCalls: false,
	});

	assert.equal(first, false);
	assert.equal(second, true);
});

void test("intervention escalation persists across rebuilds when the controller is shared", async () => {
	const config = {
		baseUrl: "http://fake",
		model: "fake",
		executionProfile: "autonomous" as const,
		guardsEnabled: true,
	};
	const interventions = new HarnessInterventionController();
	const events: Array<{ type: string; attempt?: number }> = [];
	const loopDetector = new LoopDetector({ duplicateThreshold: 1 });

	const build = () =>
		buildBuiltinHooks({
			config,
			contextWindowTokens: () => 4096,
			toolDefs: () => [],
			loopDetector,
			emitEvent: e => events.push(e as { type: string; attempt?: number }),
			interventions,
		});

	const call = { id: "1", name: "read_file", arguments: '{"path":"a.txt"}' };
	const args = { path: "a.txt" };

	// Trip the duplicate guard twice across two rebuilds — same incident key
	// ("loop"/"tool_call_guard"/"duplicate"), so attempt should escalate to 2.
	build().beforeToolCall?.({ toolCall: call, args, iteration: 1 });
	build().beforeToolCall?.({ toolCall: call, args, iteration: 2 });

	const interventionEvents = events.filter(
		e => e.type === "harness_intervention",
	);
	assert.equal(interventionEvents.length, 2);
	assert.equal(interventionEvents[0]?.attempt, 1);
	assert.equal(interventionEvents[1]?.attempt, 2);
});
