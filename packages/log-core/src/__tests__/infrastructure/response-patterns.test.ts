import { test } from "bun:test";
import assert from "node:assert/strict";
import { awaitsUserInput } from "../../core/guards/response-patterns.ts";

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

void test("awaitsUserInput ignores a trailing self-directed reasoning question", () => {
	assert.equal(
		awaitsUserInput(
			"The first approach failed. Let me check the types, or should I try another approach?",
		),
		false,
	);
});

void test("awaitsUserInput handles question with choice list", () => {
	assert.ok(awaitsUserInput("Which option?\n- A\n- B\n- C"));
});
