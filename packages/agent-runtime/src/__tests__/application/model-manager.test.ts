import { expect, test } from "bun:test";
import type { AgentConfig } from "@logician/agent-core";
import { RuntimeModelManager } from "../../application/runtime/model-manager.ts";

test("model manager presents and selects configured endpoints", () => {
	const config = {
		baseUrl: "http://default",
		model: "alpha",
		models: [
			{ name: "Alpha", model: "alpha" },
			{ name: "Beta", model: "beta", url: "http://beta" },
		],
		systemPrompt: "",
		tools: [],
	} satisfies AgentConfig;
	let endpoint: [string, string] | undefined;
	const manager = new RuntimeModelManager(
		() => config,
		() =>
			({
				models: {
					model: config.model,
					setEndpoint: (model: string, url: string) => {
						endpoint = [model, url];
					},
				},
			}) as never,
	);
	expect(manager.options().map(option => option.active)).toEqual([true, false]);
	expect(manager.selectOption("1:Beta")).toEqual({
		model: "beta",
		url: "http://beta",
	});
	expect(endpoint).toEqual(["beta", "http://beta"]);
});
