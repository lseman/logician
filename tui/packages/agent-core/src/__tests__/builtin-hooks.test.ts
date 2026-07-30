import assert from "node:assert/strict";
import { test } from "node:test";
import {
	awaitsUserInput,
	detectsCircling,
} from "../agent/guards/response-patterns.ts";
import { LoopDetector } from "../agent/guards/loop-detector.ts";
import { buildBuiltinHooks } from "../hooks/builtin/builtin-hooks.ts";

// ── detectsCircling ───────────────────────────────────────────────────────

void test("detectsCircling: catches retry patterns", () => {
	assert.ok(detectsCircling("I'll try again with a different approach."));
	assert.ok(detectsCircling("Let me try again."));
	assert.ok(detectsCircling("I've tried reading the file again."));
	assert.ok(detectsCircling("I will try to read the file next."));
	assert.ok(detectsCircling("Let me try another approach."));
	assert.ok(detectsCircling("I'll try to read the file."));
	assert.ok(detectsCircling("I'm going to try again next."));
});

void test("detectsCircling: catches attempt patterns", () => {
	assert.ok(detectsCircling("I attempted to fix it but it didn't work."));
	assert.ok(detectsCircling("I tried a different method yet."));
	assert.ok(detectsCircling("I've tried again and attempted another way."));
	assert.ok(detectsCircling("I tried to read again next."));
	assert.ok(detectsCircling("I tried to fix it next."));
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
	assert.ok(awaitsUserInput("I found two valid approaches. Which one do you prefer?"));
	assert.ok(awaitsUserInput("Please choose one of the options below:"));
	assert.ok(awaitsUserInput("I need your confirmation."));
	assert.ok(awaitsUserInput("Which environment should I use?\n\n1. Staging\n2. Production"));
});

void test("awaitsUserInput: ignores questions followed by continued work", () => {
	assert.ok(!awaitsUserInput("What caused this? I will inspect the stack trace next."));
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
		loopDetector: new LoopDetector(),
	});

	assert.equal(hooks.getFollowUpMessages, undefined);
	assert.equal(hooks.shouldStopAfterTurn, undefined);
	assert.equal(hooks.afterProviderResponse, undefined);
	assert.equal(typeof hooks.prepareNextTurn, "function");
	assert.equal(typeof hooks.beforeToolCall, "function");
	assert.equal(typeof hooks.afterToolCall, "function");
});
