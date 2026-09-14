import { test } from "bun:test";
import assert from "node:assert/strict";
import { ask } from "../../capabilities/ask/index.ts";

void test("ask sends canonical multi-question requests", async () => {
	const result = await ask.execute(
		{
			questions: [
				{
					id: "scope",
					header: "Scope",
					question: "Choose a scope",
					choices: [{ value: "small", label: "Small" }],
				},
				{
					id: "tests",
					header: "Tests",
					question: "Choose validation",
					choices: [{ value: "full", label: "Full suite" }],
				},
			],
		},
		{
			onQuestionRequest: async ({ questions }) => {
				assert.deepEqual(
					questions.map(question => question.id),
					["scope", "tests"],
				);
				return '{"scope":"small","tests":"full"}';
			},
		},
	);
	assert.equal(result, 'User responded: {"scope":"small","tests":"full"}');
});

void test("ask rejects removed single-question arguments", async () => {
	const prepared = ask.prepareArguments?.({
		question: "Continue?",
		choices: [{ value: "yes", label: "Yes" }],
	});
	assert.deepEqual(prepared, { questions: undefined });
	assert.match(
		String(await ask.execute(prepared ?? {}, {})),
		/requires at least one choice/,
	);
});

void test("ask passes multi and recommended through, and only when recommended matches a real choice", async () => {
	const result = await ask.execute(
		{
			questions: [
				{
					id: "tags",
					question: "Which apply?",
					multi: true,
					recommended: "bug",
					choices: [
						{ value: "bug", label: "Bug" },
						{ value: "perf", label: "Performance" },
					],
				},
				{
					id: "scope",
					question: "Which scope?",
					recommended: "does-not-exist",
					choices: [{ value: "small", label: "Small" }],
				},
			],
		},
		{
			onQuestionRequest: async ({ questions }) => {
				assert.equal(questions[0]?.multi, true);
				assert.equal(questions[0]?.recommended, "bug");
				// A recommended value with no matching choice is dropped rather
				// than passed through as a dangling reference.
				assert.equal(questions[1]?.recommended, undefined);
				return "ok";
			},
		},
	);
	assert.equal(result, "User responded: ok");
});

void test("ask with allowOther appends a reserved free-text choice", async () => {
	const result = await ask.execute(
		{
			questions: [
				{
					id: "reason",
					question: "Why?",
					allowOther: true,
					choices: [{ value: "a", label: "Option A" }],
				},
			],
		},
		{
			onQuestionRequest: async ({ questions }) => {
				const choices = questions[0]?.choices ?? [];
				assert.equal(choices.length, 2);
				const other = choices[1];
				assert.equal(other?.isFreeText, true);
				assert.equal(other?.label, "Other (type your own)");
				return "custom answer";
			},
		},
	);
	assert.equal(result, "User responded: custom answer");
});

void test("ask with allowOther and no other choices is still valid (free-text-only question)", async () => {
	const result = await ask.execute(
		{
			questions: [
				{ id: "reason", question: "Why?", allowOther: true, choices: [] },
			],
		},
		{
			onQuestionRequest: async ({ questions }) => {
				assert.equal(questions[0]?.choices.length, 1);
				return "typed answer";
			},
		},
	);
	assert.equal(result, "User responded: typed answer");
});

void test("ask requires stable unique question ids", async () => {
	const result = await ask.execute(
		{
			questions: [
				{
					id: "same",
					question: "First?",
					choices: [{ value: "a", label: "A" }],
				},
				{
					id: "same",
					question: "Second?",
					choices: [{ value: "b", label: "B" }],
				},
			],
		},
		{},
	);
	assert.match(String(result), /ids must be unique/);
});
