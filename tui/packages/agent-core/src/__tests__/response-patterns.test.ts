import assert from "node:assert/strict";
import { test } from "node:test";
import {
	looksNonCommittal,
	looksComplete,
	awaitsUserInput,
	detectsCircling,
} from "../core/guards/response-patterns.ts";

// ── looksNonCommittal ──────────────────────────────────────────────────────

void test("looksNonCommittal detects hedging patterns", () => {
	assert.ok(looksNonCommittal("I need to check the source code"));
	assert.ok(looksNonCommittal("Let me think about this"));
	assert.ok(looksNonCommittal("I'm going to analyze the problem"));
	assert.ok(looksNonCommittal("I'll try to investigate"));
	assert.ok(looksNonCommittal("But I need to verify"));
	assert.ok(looksNonCommittal("This requires further analysis"));
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
	assert.equal(awaitsUserInput("I checked the file. The issue is on line 5."), false);
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
	assert.ok(detectsCircling("I tried it again"));
	assert.ok(detectsCircling("I will attempt to fix it"));
});

void test("detectsCircling detects failed-then-retry pattern", () => {
	assert.ok(detectsCircling("I tried X but it did not work"));
	assert.ok(detectsCircling("I attempted it again"));
	assert.ok(detectsCircling("Cannot fix it but I'll try"));
});

void test("detectsCircling returns false for decisive text", () => {
	assert.equal(detectsCircling("I found the fix and applied it"), false);
	assert.equal(detectsCircling("The test passes now"), false);
	assert.equal(detectsCircling("short"), false);
});

void test("detectsCircling returns false for short text", () => {
	assert.equal(detectsCircling(""), false);
	assert.equal(detectsCircling("ok"), false);
});
