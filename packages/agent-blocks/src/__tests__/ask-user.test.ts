import { test } from "bun:test";
import assert from "node:assert/strict";
import { ask_user } from "../interaction/ask-user/index.ts";

void test("ask_user sends canonical multi-question requests", async () => {
	const result = await ask_user.execute(
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

void test("ask_user rejects removed single-question arguments", async () => {
	const prepared = ask_user.prepareArguments?.({
		question: "Continue?",
		choices: [{ value: "yes", label: "Yes" }],
	});
	assert.deepEqual(prepared, { questions: undefined });
	assert.match(
		String(await ask_user.execute(prepared ?? {}, {})),
		/requires at least one choice/,
	);
});

void test("ask_user requires stable unique question ids", async () => {
	const result = await ask_user.execute(
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
