import { describe, expect, test } from "bun:test";
import {
	buildJudgeRequest,
	JUDGED_CONTENT_MAX_CHARS,
	parseJudgeVerdicts,
} from "../../ttsr/judge.ts";
import { TtsrManager } from "../../ttsr/ttsr-manager.ts";
import type { TtsrRule } from "../../types/ttsr.ts";

const judged: TtsrRule = {
	name: "judged",
	path: "rules/judged.md",
	description: "",
	content: "Do not do the thing.",
	conditions: [],
	question: "Does the output do the thing?",
	scope: ["text"],
	interruptMode: "always",
};

describe("parseJudgeVerdicts", () => {
	test("reads yes answers from a JSON object wrapped in prose", () => {
		expect(
			parseJudgeVerdicts(
				'Sure:\n```json\n{"q0":"no","q1":"Yes","q2":true}\n```',
				3,
			),
		).toEqual([1, 2]);
	});

	test("fails closed on anything unparseable", () => {
		for (const reply of ["yes", "{q0: yes}", '["yes"]', "", '{"q0":"maybe"}']) {
			expect(parseJudgeVerdicts(reply, 1)).toEqual([]);
		}
	});

	test("ignores answers beyond the asked questions", () => {
		expect(parseJudgeVerdicts('{"q0":"no","q5":"yes"}', 1)).toEqual([]);
	});
});

describe("buildJudgeRequest", () => {
	test("numbers questions and caps the output", () => {
		const { user } = buildJudgeRequest(
			{
				content: "x".repeat(JUDGED_CONTENT_MAX_CHARS + 10),
				context: { source: "text" },
				subject: "reply",
			},
			[
				{ rule: judged, question: "First?" },
				{ rule: judged, question: "Second?" },
			],
		);
		expect(user).toContain("q0: First?\nq1: Second?");
		expect(user).toContain("[… output truncated for judging]");
	});
});

describe("TtsrManager judged rules", () => {
	test("registers question-only rules without stream matching them", () => {
		const manager = new TtsrManager({ builtinRules: false });
		expect(manager.addRule(judged)).toBe(true);
		expect(manager.hasJudgedRules()).toBe(true);
		expect(
			manager.checkDelta("the output does the thing", { source: "text" }),
		).toEqual([]);
	});

	test("judgedCandidates respects scope and the repeat gate", async () => {
		const manager = new TtsrManager({ builtinRules: false });
		manager.addRule(judged);
		expect(await manager.judgedCandidates("out", { source: "tool" })).toEqual(
			[],
		);
		const candidates = await manager.judgedCandidates("out", {
			source: "text",
		});
		expect(candidates.map(c => c.question)).toEqual([
			judged.question as string,
		]);

		expect(manager.claim([judged]).map(r => r.name)).toEqual(["judged"]);
		expect(manager.claim([judged])).toEqual([]);
		expect(await manager.judgedCandidates("out", { source: "text" })).toEqual(
			[],
		);
	});

	test("judge: false turns judged rules off", async () => {
		const manager = new TtsrManager({ builtinRules: false, judge: false });
		manager.addRule(judged);
		expect(manager.hasJudgedRules()).toBe(false);
		expect(await manager.judgedCandidates("out", { source: "text" })).toEqual(
			[],
		);
	});
});
