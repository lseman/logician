import { test } from "bun:test";
import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { LoopDetector } from "../agent/guards/loop-detector.ts";
import { awaitsUserInput } from "../agent/guards/response-patterns.ts";
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
			toolCall: { id: String(n), name: "bash", arguments: '{"command":"npm test"}' },
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
