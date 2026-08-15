import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	awaitsUserInput,
	detectsCircling,
	looksComplete,
	looksNonCommittal,
} from "../agent/guards/response-patterns.ts";

// ── looksNonCommittal ──────────────────────────────────────────────────────

void test("looksNonCommittal detects hedging patterns", () => {
	assert.ok(
		looksNonCommittal("I'm not sure what to do next, let me think about it"),
	);
	assert.ok(
		looksNonCommittal(
			"I need to figure out the root cause, let me first check the logs",
		),
	);
	assert.ok(looksNonCommittal("This requires more thought"));
	assert.ok(looksNonCommittal("Let me reconsider my approach here"));
	assert.ok(
		looksNonCommittal("At this point I am not sure what is happening"),
	);
});

void test("looksNonCommittal returns false for short text", () => {
	assert.equal(looksNonCommittal("check"), false);
	assert.equal(looksNonCommittal("ok"), false);
	assert.equal(looksNonCommittal(""), false);
});

void test("looksNonCommittal returns false for decisive text", () => {
	assert.equal(looksNonCommittal("I found the bug in auth.ts line 42"), false);
	assert.equal(looksNonCommittal("Running the test now"), false);
});

void test("looksNonCommittal returns false for legitimate single-clause planning language", () => {
	// These are normal agent planning statements, not hedging loops — the
	// sharpened patterns require a compounding second clause to trigger.
	assert.equal(looksNonCommittal("Let me think about the approach"), false);
	assert.equal(looksNonCommittal("I'll try reading the file first"), false);
	assert.equal(
		looksNonCommittal("I'm going to analyze the problem"),
		false,
	);
});

// ── looksComplete ──────────────────────────────────────────────────────────

void test("looksComplete detects completion phrases", () => {
	assert.ok(looksComplete("Task complete"));
	assert.ok(looksComplete("All done"));
	assert.ok(looksComplete("Finished successfully"));
	assert.ok(looksComplete("Nothing else to do"));
	assert.ok(looksComplete("That's all done"));
	assert.ok(looksComplete("done"));
});

void test("looksComplete returns false for incomplete text", () => {
	assert.equal(looksComplete("I'm working on it"), false);
	assert.equal(looksComplete("Let me try"), false);
	assert.equal(looksComplete(""), false);
});

// ── awaitsUserInput ────────────────────────────────────────────────────────

void test("awaitsUserInput detects trailing questions", () => {
	assert.ok(awaitsUserInput("What should I do next?"));
	assert.ok(awaitsUserInput("Should I proceed?"));
	assert.ok(awaitsUserInput("Can you help?"));
});

void test("awaitsUserInput detects request phrases at end", () => {
	assert.ok(awaitsUserInput("Please answer the question"));
	assert.ok(awaitsUserInput("Let me know which option"));
	assert.ok(awaitsUserInput("I need your confirmation"));
});

void test("awaitsUserInput returns false for questions mid-text", () => {
	assert.equal(
		awaitsUserInput("I checked the file. The issue is on line 5."),
		false,
	);
	assert.equal(awaitsUserInput("The function works as expected"), false);
	assert.equal(awaitsUserInput(""), false);
});

void test("awaitsUserInput handles question with choice list", () => {
	assert.ok(awaitsUserInput("Which option?\n- A\n- B\n- C"));
});

// ── detectsCircling ────────────────────────────────────────────────────────

void test("detectsCircling detects retry intent", () => {
	assert.ok(detectsCircling("I'll try again"));
	assert.ok(detectsCircling("Let me try again"));
	assert.ok(detectsCircling("I will attempt the same thing again"));
	assert.ok(detectsCircling("Cannot fix it but I'll try"));
});

void test("detectsCircling detects failed-then-retry pattern", () => {
	assert.ok(
		detectsCircling(
			"I tried it but however it failed, let me try again",
		),
	);
	assert.ok(detectsCircling("Cannot fix it but I'll try"));
});

void test("detectsCircling returns false for decisive text", () => {
	assert.equal(detectsCircling("I found the fix and applied it"), false);
	assert.equal(detectsCircling("The test passes now"), false);
	assert.equal(detectsCircling("short"), false);
});

void test("detectsCircling returns false for legitimate first-attempt language", () => {
	assert.equal(detectsCircling("I will try reading the file first"), false);
	assert.equal(detectsCircling("Let me attempt the bash command"), false);
});

void test("detectsCircling returns false for short text", () => {
	assert.equal(detectsCircling(""), false);
	assert.equal(detectsCircling("ok"), false);
});
