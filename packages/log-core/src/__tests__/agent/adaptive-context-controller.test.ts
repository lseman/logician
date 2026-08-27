import { describe, expect, test } from "bun:test";
import { AdaptiveContextController } from "../../system/context/adaptive-context-controller.ts";

const estimate = (messages: readonly unknown[]) => messages.length * 10;

describe("AdaptiveContextController", () => {
	test("packs relevant context at message granularity within the budget", () => {
		const controller = new AdaptiveContextController(estimate);
		const plan = controller.buildContext({
			history: [{ role: "user", content: "fix authentication" }],
			objective: "fix the authentication token validator",
			maxInjectedTokens: 20,
			contributions: [
				{
					source: "unrelated",
					messages: [{ role: "system", content: "CSS colors" }],
				},
				{
					source: "repository-map",
					messages: [
						{ role: "system", content: "authentication token validator" },
						{ role: "system", content: "authentication tests" },
						{ role: "system", content: "extra authentication history" },
					],
				},
			],
		});

		expect(plan.budget).toEqual({ limit: 20, used: 20 });
		expect(plan.messages.map(message => message.content)).toEqual([
			"fix authentication",
			"authentication token validator",
			"authentication tests",
		]);
		expect(plan.sources[0]).toEqual({
			source: "repository-map",
			messages: 2,
			estimatedTokens: 20,
			included: true,
		});
	});

	test("learns from outcomes while preserving declared priority", () => {
		const controller = new AdaptiveContextController(estimate, {
			learningRate: 1,
			learningWeight: 4,
		});
		const contributions = [
			{
				source: "memory",
				messages: [{ role: "system" as const, content: "memory" }],
			},
			{
				source: "graph",
				messages: [{ role: "system" as const, content: "graph" }],
			},
		];
		const first = controller.buildContext({
			history: [],
			maxInjectedTokens: 10,
			contributions,
		});
		expect(first.sources[0]?.source).toBe("memory");
		expect(controller.recordOutcome(first.id, { success: false })).toBe(true);

		const second = controller.buildContext({
			history: [],
			maxInjectedTokens: 10,
			contributions,
		});
		expect(second.sources[0]?.source).toBe("graph");
		expect(controller.recordOutcome(second.id, { success: true })).toBe(true);
		expect(controller.recordOutcome(second.id, { success: true })).toBe(false);
	});

	test("deduplicates history and credits explicitly useful sources", () => {
		const controller = new AdaptiveContextController(estimate);
		const existing = { role: "user" as const, content: "same" };
		const plan = controller.buildContext({
			history: [existing],
			contributions: [
				{ source: "memory", messages: [existing] },
				{ source: "skills", systemPrompt: "skill prompt" },
			],
		});
		expect(plan.messages).toEqual([existing]);
		expect(plan.systemPrompt).toBe("skill prompt");
		expect(
			controller.recordOutcome(plan.id, {
				success: true,
				usefulSources: ["skills"],
			}),
		).toBe(true);
	});
});
