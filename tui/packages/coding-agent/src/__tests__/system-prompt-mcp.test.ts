import assert from "node:assert/strict";
import { test } from "node:test";
import type { Tool } from "@logician/agent-core";
import { buildDefaultSystemPrompt } from "../system-prompt.ts";

void test("system prompt makes MCP the primary tool-selection workflow", () => {
	const fullDescription =
		"Execute repository commands while retaining only relevant output, avoiding large raw command results in the model context.";
	const mcpTool: Tool = {
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
	const grepTool: Tool = {
		name: "grep",
		label: "Grep",
		description: "Search file contents.",
		parameters: { type: "object", properties: {} },
		execute: async () => "ok",
	};
	const bashTool: Tool = {
		name: "bash",
		label: "Bash",
		description: "Run a shell command.",
		parameters: { type: "object", properties: {} },
		execute: async () => "ok",
	};

	const prompt = buildDefaultSystemPrompt("/workspace", [
		grepTool,
		bashTool,
		mcpTool,
	]);

	assert.match(prompt, new RegExp(fullDescription));
	assert.match(
		prompt,
		/Use ctx_execute for repository commands with potentially large output\./,
	);
	assert.match(prompt, /MCP-first tool workflow:/);
	assert.match(prompt, /MCP tools currently available: ctx_execute\./);
	assert.match(
		prompt,
		/For repository exploration, search, command execution, or large-output inspection, prefer ctx_execute over raw grep\/find\/bash/,
	);
	assert.match(
		prompt,
		/Before choosing grep, find, bash, git, web, or generic file tools/,
	);
	assert.match(prompt, /try the closest MCP alternative before abandoning MCP/);
	assert.ok(
		prompt.indexOf("MCP-first tool workflow:") <
			prompt.indexOf("Default coding-agent workflow:"),
	);
});

void test("system prompt omits MCP policy when no MCP tools are available", () => {
	const localTool: Tool = {
		name: "grep",
		label: "Grep",
		description: "Search file contents.",
		parameters: { type: "object", properties: {} },
		execute: async () => "ok",
	};

	const prompt = buildDefaultSystemPrompt("/workspace", [localTool]);

	assert.doesNotMatch(prompt, /MCP-first tool workflow:/);
	assert.match(prompt, /Use local list_files, find, grep/);
});
