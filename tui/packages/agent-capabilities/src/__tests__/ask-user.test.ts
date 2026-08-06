import assert from "node:assert/strict";
import { test } from "node:test";
import { ask_user } from "../interaction/ask-user/index.ts";

void test("ask_user forwards a tabbed questionnaire and returns structured answers", async () => {
	let received: unknown;
	const result = await ask_user.execute(
		{
			questions: [
				{
					id: "scope",
					header: "Scope",
					question: "Choose a scope",
					choices: [
						{
							value: "small",
							label: "Focused",
							description: "Keep the change narrow.",
						},
					],
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
			onQuestionRequest: async questionnaire => {
				received = questionnaire;
				return JSON.stringify({ scope: "small", tests: "full" });
			},
		},
	);

	assert.deepEqual(received, {
		questions: [
			{
				id: "scope",
				header: "Scope",
				question: "Choose a scope",
				choices: [
					{
						value: "small",
						label: "Focused",
						description: "Keep the change narrow.",
					},
				],
			},
			{
				id: "tests",
				header: "Tests",
				question: "Choose validation",
				choices: [{ value: "full", label: "Full suite" }],
			},
		],
	});
	assert.equal(result, 'User responded: {"scope":"small","tests":"full"}');
});

void test("ask_user preserves the legacy single-question shape", async () => {
	const prepared = ask_user.prepareArguments?.({
		question: "Continue?",
		choices: [{ value: "yes", label: "Yes" }],
	});
	const result = await ask_user.execute(prepared ?? {}, {
		onQuestionRequest: async ({ questions }) => {
			assert.equal(questions[0]?.id, "answer");
			return "yes";
		},
	});
	assert.equal(result, "User responded: yes");
});
