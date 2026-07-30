import assert from "node:assert/strict";
import { test } from "node:test";
import type { Tool } from "@logician/agent-core";
import { buildDefaultSystemPrompt } from "../context/system-prompt.ts";

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
	const fffGrepTool: Tool = {
		name: "fff__grep",
		label: "MCP: Grep (fff)",
		description:
			"Search repository file contents using the fff index and return focused matches.",
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
		fffGrepTool,
	]);

	assert.match(prompt, new RegExp(fullDescription));
	assert.match(
		prompt,
		/Use ctx_execute for repository commands with potentially large output\./,
	);
	assert.match(prompt, /MCP-first tool workflow:/);
	assert.match(
		prompt,
		/MCP tools currently available: ctx_execute, fff__grep\./,
	);
	assert.match(
		prompt,
		/For repository content or symbol search, use fff__grep before local grep, rg, git grep, or a shell search pipeline\./,
	);
	assert.match(
		prompt,
		/For repository commands whose output may be large, use ctx_execute before bash/,
	);
	assert.match(
		prompt,
		/Before choosing grep, find, bash, git, web, or generic file tools/,
	);
	assert.match(prompt, /try the closest MCP alternative before abandoning MCP/);
	assert.match(
		prompt,
		/Local list_files, find, grep, read_file, git status\/diff, and bash are fallback tools\./,
	);
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
	assert.match(
		prompt,
		/Local list_files, find, grep, read_file, git status\/diff, and bash are fallback tools\./,
	);
});
