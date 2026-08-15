import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	awaitsUserInput,
	looksComplete,
} from "../agent/guards/response-patterns.ts";

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


