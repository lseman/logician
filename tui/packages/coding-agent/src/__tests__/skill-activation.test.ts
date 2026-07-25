import assert from "node:assert/strict";
import { test } from "node:test";
import {
	formatActivatedSkills,
	formatSkillActivationNotice,
	SkillActivationSession,
	selectSkillsForPrompt,
} from "../skill-activation.ts";
import type { Skill } from "../skills.ts";

function skill(
	name: string,
	description: string,
	metadata: Partial<Skill> = {},
): Skill {
	return {
		name,
		displayName: name,
		description,
		content: `Instructions for ${name}.`,
		filePath: `/skills/${name}/SKILL.md`,
		baseDir: `/skills/${name}`,
		slashName: name,
		disableModelInvocation: false,
		source: "user",
		...metadata,
	};
}

void test("selects a specialist from routing metadata", () => {
	const review = skill(
		"typescript-code-review",
		"Review TypeScript code for correctness, type safety, and maintainability.",
		{ triggers: ["TypeScript code review", "review TS", "type safety"] },
	);
	const debugging = skill(
		"typescript-debugging",
		"Diagnose TypeScript errors, exceptions, and module failures.",
		{ triggers: ["TypeScript type error", "TS exception", "module failure"] },
	);

	const result = selectSkillsForPrompt(
		[debugging, review],
		"Please review this TypeScript service for correctness.",
	);
	assert.deepEqual(result.map(({ skill: selected }) => selected.name), [
		"typescript-code-review",
	]);
	assert.match(result[0].reason, /matched|relevant/);
});

void test("explicit skill references force activation", () => {
	const ariadne = skill("ariadne", "Query the codebase graph.");
	const other = skill("typescript-debugging", "Diagnose TypeScript errors.");

	assert.equal(
		selectSkillsForPrompt([other, ariadne], "Use $ariadne before reading.")[0]
			.skill.name,
		"ariadne",
	);
	assert.equal(
		selectSkillsForPrompt([other, ariadne], "Run /ariadne for this.")[0]
			.skill.name,
		"ariadne",
	);
});

void test("hidden and contraindicated skills do not auto-activate", () => {
	const hidden = skill("release", "Release production builds.", {
		triggers: ["release production"],
		disableModelInvocation: true,
	});
	const local = skill("local-files", "Inspect local project files.", {
		triggers: ["inspect project files"],
		whenNotToUse: ["remote-only task"],
	});

	assert.deepEqual(
		selectSkillsForPrompt(
			[hidden, local],
			"Inspect project files for this remote-only task, then release production.",
		),
		[],
	);
});

void test("does not activate on weak generic overlap", () => {
	const skillList = [
		skill("ariadne", "Use for review, context, graph traversal, and code analysis."),
		skill("typescript-router", "Route TypeScript tasks to engineering skills."),
	];
	assert.deepEqual(
		selectSkillsForPrompt(skillList, "Please update the README title."),
		[],
	);
});

void test("formats full selected skill bodies for the turn", () => {
	const selected = skill("typescript-debugging", "Diagnose TypeScript errors.");
	const formatted = formatActivatedSkills([
		{ skill: selected, score: 20, reason: "trigger" },
	]);
	assert.match(formatted, /<activated-skills>/);
	assert.match(formatted, /Instructions for typescript-debugging\./);
	assert.match(formatted, /<skill name="typescript-debugging"/);
});

void test("formats concise human-facing activation reasons", () => {
	const selected = skill("typescript-debugging", "Diagnose TypeScript errors.", {
		displayName: "TypeScript Debugging",
	});
	assert.equal(
		formatSkillActivationNotice([
			{ skill: selected, score: 20, reason: "matched “TypeScript error”" },
		]),
		"TypeScript Debugging · matched “TypeScript error”",
	);
	assert.equal(
		formatSkillActivationNotice([
			{ skill: selected, score: 100, reason: "explicitly requested" },
			{
				skill: skill("ariadne", "Trace code dependencies."),
				score: 100,
				reason: "explicitly requested",
			},
		]),
		"TypeScript Debugging · explicitly requested  +  ariadne · explicitly requested",
	);
});

void test("continued activations are identified as continued", () => {
	const selected = skill("typescript-debugging", "Diagnose TypeScript errors.", {
		triggers: ["TypeScript error"],
	});
	const session = new SkillActivationSession();
	const initial = session.select([selected], "Diagnose this TypeScript error.");
	session.continueWith(initial);

	assert.equal(
		session.select([selected], "continue")[0].reason,
		"continuing from the previous turn",
	);
});
