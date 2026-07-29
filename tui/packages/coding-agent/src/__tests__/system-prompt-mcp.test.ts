import assert from "node:assert/strict";
import { test } from "node:test";
import type { Tool } from "@logician/agent-core";
import { buildDefaultSystemPrompt } from "../system-prompt.ts";

void test("system prompt preserves MCP guidance and prefers context tools", () => {
	const fullDescription =
		"Execute repository commands while retaining only relevant output, avoiding large raw command results in the model context.";
	const tool: Tool = {
		name: "ctx_execute",
		label: "MCP: Context execute",
		description: fullDescription,
		promptSnippet: fullDescription,
		promptGuidelines: [
			"Use ctx_execute for repository commands with potentially large output.",
		],
		parameters: { type: "object", properties: {} },
		execute: async () => "ok",
	};

	const prompt = buildDefaultSystemPrompt("/workspace", [tool]);

	assert.match(prompt, new RegExp(fullDescription));
	assert.match(
		prompt,
		/Use ctx_execute for repository commands with potentially large output\./,
	);
	assert.match(prompt, /MCP tool workflow:/);
	assert.match(
		prompt,
		/Prefer ctx_execute for repository exploration, searches, and commands/,
	);
});
