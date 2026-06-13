import assert from "node:assert/strict";
import { test } from "node:test";
import { declaresStop, detectsCircling } from "../builtin-hooks.ts";

// ── declaresStop ──────────────────────────────────────────────────────────

void test("declaresStop: recognizes completion", () => {
	assert.ok(declaresStop("I've completed the task."));
	assert.ok(declaresStop("All done."));
	assert.ok(declaresStop("Everything is complete."));
	assert.ok(declaresStop("The work is finished."));
	assert.ok(declaresStop("All tasks are done."));
});

void test("declaresStop: recognizes blocked", () => {
	assert.ok(declaresStop("I can't access the file."));
	assert.ok(declaresStop("I am unable to proceed."));
	assert.ok(declaresStop("I'm blocked."));
	assert.ok(declaresStop("I don't have access."));
	assert.ok(declaresStop("Unable to proceed."));
});

void test("declaresStop: recognizes stuck variants", () => {
	assert.ok(declaresStop("I am truly stuck on this."));
	assert.ok(declaresStop("I don't know how to proceed."));
	assert.ok(declaresStop("I can't make progress."));
	assert.ok(declaresStop("I'm completely confused."));
	assert.ok(declaresStop("I don't know what else to try."));
	assert.ok(declaresStop("I don't have the ability to finish."));
});

void test("declaresStop: does NOT false-positive on incidental phrases", () => {
	assert.ok(!declaresStop("I cannot stress enough how important this is."));
	assert.ok(!declaresStop("The task is complex but I'll try harder."));
	assert.ok(!declaresStop("I can see the file exists."));
	// "I'll try again" is circling but NOT a stop declaration.
	assert.ok(!declaresStop("I will try again with a different approach."));
	// "I can't see the file" is not a stop — it's an observation.
	assert.ok(!declaresStop("I can't see the file contents clearly."));
});

void test("declaresStop: only checks the tail line", () => {
	// "I can't proceed" appears mid-text but the tail says "All done." — tail "all done" IS a stop signal.
	// So this should actually return true because "All done" matches /ball\s+doned/.
	assert.ok(declaresStop("I can't proceed with step 1.\n\nAll done."));
	// Tail line has the blocked signal
	assert.ok(declaresStop("I tried reading, editing, and searching.\n\nI can't make progress."));
	// Neutral tail — no stop signal
	assert.ok(!declaresStop("I tried reading, editing, and searching.\n\nLet me keep going."));
});

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
