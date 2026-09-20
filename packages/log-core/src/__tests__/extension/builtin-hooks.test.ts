import { test } from "bun:test";
import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { LoopDetector } from "../../control/guards/loop-detector.ts";
import { HarnessInterventionController } from "../../control/policy/intervention-controller.ts";
import { ProgressTracker } from "../../control/policy/progress-tracker.ts";
import {
	buildBuiltinHooks,
	COMPACTION_COOLDOWN_TURNS,
	rewriteCommandWithRtk,
} from "../../runtime/hooks/builtin/builtin-hooks.ts";
import type { Message } from "../../system/types/types-messages.ts";

// Capture the real PATH once at module load so cleanup always restores the
// true original value, even when other tests mutate process.env.PATH.
const __originalPath = process.env.PATH;

async function withFakeRtk<T>(body: () => Promise<T>): Promise<T> {
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
		return await body();
	} finally {
		if (__originalPath === undefined) delete process.env.PATH;
		else process.env.PATH = __originalPath;
		rmSync(root, { recursive: true, force: true });
	}
}

void test("minimal profile keeps mechanism hooks and omits built-in policies", () => {
	const hooks = buildBuiltinHooks({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			executionProfile: "minimal",
			continuationEnabled: true,
			progressStopEnabled: true,
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
			toolCall: { id: String(iteration), name: "read", arguments: "{}" },
			args: { path: "README.md" },
			iteration,
		});
	}
	assert.equal(checks, 0);
});

void test("RTK rewrite delegates supported and compound commands to RTK", async () => {
	await withFakeRtk(async () => {
		assert.equal(await rewriteCommandWithRtk("git status"), "rtk git status");
		assert.equal(
			await rewriteCommandWithRtk("cd repo && git status"),
			"cd repo && rtk git status",
		);
		assert.equal(
			await rewriteCommandWithRtk("cargo test && echo done"),
			"rtk cargo test && echo done",
		);
	});
});

void test("RTK rewrite leaves unsupported commands unchanged", async () => {
	await withFakeRtk(async () => {
		assert.equal(await rewriteCommandWithRtk("echo hello"), "echo hello");
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

void test("progress-stop tracks repeated evidence-free turns across hook rebuilds", () => {
	const config = {
		baseUrl: "http://fake",
		model: "fake",
		executionProfile: "autonomous" as const,
		progressStopEnabled: true,
	};
	const progress = new ProgressTracker({
		minimumChecks: 1,
		stalledChecks: 1,
	});
	const build = () =>
		buildBuiltinHooks({
			config,
			contextWindowTokens: () => 100_000,
			toolDefs: () => [],
			loopDetector: new LoopDetector(),
			progress,
		});

	// No successful tool evidence or task-state transition → stalls on the 2nd.
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

	const call = { id: "1", name: "read", arguments: '{"path":"a.txt"}' };
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

// ── Batch-loop redirect: detection must inject a corrective message the
// model actually sees, not just a TUI event ───────────────────────────────

void test("batch-loop detection injects a corrective redirect message for the next turn", async () => {
	const events: Array<{ type: string; kind?: string; cause?: string }> = [];
	const hooks = buildBuiltinHooks({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			executionProfile: "autonomous",
			guardsEnabled: true,
			proactiveCompactionEnabled: false,
		},
		contextWindowTokens: () => 4096,
		toolDefs: () => [],
		loopDetector: new LoopDetector({ batchThreshold: 2 }),
		emitEvent: e => events.push(e as never),
	});

	const call = { id: "1", name: "read", arguments: '{"path":"a.txt"}' };
	const args = { path: "a.txt" };
	const transcript = [{ role: "user" as const, content: "do it" }];
	// One harness loop iteration: the turn's tool call, then prepareNextTurn
	// (which runs after every turn and clears the batch accumulator).
	const runTurn = async (iteration: number, messages: Message[]) => {
		await hooks.beforeToolCall?.({ toolCall: call, args, iteration });
		await hooks.afterToolCall?.({
			toolCall: call,
			args,
			result: "line A",
			isError: false,
			iteration,
		});
		return hooks.prepareNextTurn?.({
			messages,
			iteration: iteration + 1,
			hadToolCalls: true,
		});
	};

	// Two identical tool-call batches in a row trips the batch threshold.
	const afterTurn1 = await runTurn(1, transcript);
	assert.equal(afterTurn1, undefined, "no redirect below the threshold");
	const result = await runTurn(2, transcript);

	// The corrective user message is appended — the model sees it next turn.
	assert.ok(result, "prepareNextTurn must return prepared messages");
	assert.equal(result.messages.length, transcript.length + 1);
	const injected = result.messages.at(-1)!;
	assert.equal(injected.role, "user");
	assert.match(String(injected.content), /^\[loop-redirect:batch_loop\]/);
	assert.match(String(injected.content), /`read`/);
	assert.match(String(injected.content), /a\.txt/);

	// The intervention event still fires for the TUI.
	const intervention = events.find(
		e => e.type === "harness_intervention" && e.kind === "loop",
	);
	assert.ok(intervention, "loop intervention event expected");
	assert.equal(intervention.cause, "batch-loop");

	// A third identical batch must NOT re-inject: detection fires exactly
	// once per episode (at the threshold), never on every subsequent turn.
	const again = await runTurn(3, result.messages);
	assert.equal(again, undefined, "no second redirect in the same episode");
});

void test("changed batches and tool-less turns do not inject a loop redirect", async () => {
	const hooks = buildBuiltinHooks({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			executionProfile: "autonomous",
			guardsEnabled: true,
			proactiveCompactionEnabled: false,
		},
		contextWindowTokens: () => 4096,
		toolDefs: () => [],
		loopDetector: new LoopDetector({ batchThreshold: 2 }),
	});

	const transcript = [{ role: "user" as const, content: "do it" }];

	// Turn 1: read a.txt. Turn 2: read b.txt — different batch, no loop.
	// prepareNextTurn interleaves, as in the harness.
	for (const [id, p] of [
		["1", "a.txt"],
		["2", "b.txt"],
	] as const) {
		const call = { id, name: "read", arguments: `{"path":"${p}"}` };
		await hooks.beforeToolCall?.({
			toolCall: call,
			args: { path: p },
			iteration: 1,
		});
		await hooks.afterToolCall?.({
			toolCall: call,
			args: { path: p },
			result: "ok",
			isError: false,
			iteration: 1,
		});
		const result = await hooks.prepareNextTurn?.({
			messages: transcript,
			iteration: 2,
			hadToolCalls: true,
		});
		assert.equal(result, undefined, "changed batch must not redirect");
	}

	// A turn with no tool calls never records a batch.
	const plain = await hooks.prepareNextTurn?.({
		messages: transcript,
		iteration: 3,
		hadToolCalls: false,
	});
	assert.equal(plain, undefined);
});
