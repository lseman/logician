import { describe, expect, test } from "bun:test";
import { ContextEngine } from "../../system/context/context-engine.ts";

describe("ContextEngine", () => {
	test("deduplicates contributions and attributes their usage", () => {
		const engine = new ContextEngine(messages => messages.length * 10);
		const snapshot = engine.assemble({
			history: [{ role: "user", content: "existing" }],
			contributions: [
				{
					source: "skills",
					messages: [
						{ role: "user", content: "existing" },
						{ role: "user", content: "new" },
					],
				},
			],
		});
		expect(snapshot.messages.map(message => message.content)).toEqual([
			"existing",
			"new",
		]);
		expect(snapshot.sources).toEqual([
			{ source: "skills", messages: 1, estimatedTokens: 10, included: true },
		]);
	});

	test("uses priority for system-prompt overrides and injected budgets", () => {
		const engine = new ContextEngine(messages => messages.length * 10);
		const snapshot = engine.assemble({
			history: [],
			baseSystemPrompt: "base",
			maxInjectedTokens: 10,
			contributions: [
				{
					source: "low",
					priority: 0,
					systemPrompt: "low",
					messages: [{ role: "user", content: "low" }],
				},
				{
					source: "high",
					priority: 1,
					systemPrompt: "high",
					messages: [{ role: "user", content: "high" }],
				},
			],
		});
		expect(snapshot.systemPrompt).toBe("high");
		expect(snapshot.messages).toEqual([{ role: "user", content: "high" }]);
		expect(snapshot.sources[1].included).toBe(false);
	});
});
