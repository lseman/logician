import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	allocateMcpToolName,
	createMcpTool,
	encodeMcpMessage,
	formatMcpToolResult,
	tryDecodeMcpMessage,
} from "../mcp/client.ts";

void test("MCP tools expose concise names and retain qualified hook aliases", () => {
	const client = {
		name: "plugin_context-mode_context-mode",
		callTool: async () => ({}),
	};
	const description =
		"Gather without flooding context. This full routing guidance must remain visible beyond eighty characters.";
	const tool = createMcpTool(client as never, {
		name: "ctx_batch_execute",
		description,
		inputSchema: { type: "object", properties: {} },
	}) as {
		name: string;
		promptSnippet: string;
		hookAliases: string[];
	};

	assert.equal(tool.name, "ctx_batch_execute");
	assert.deepEqual(tool.hookAliases, [
		"mcp__plugin_context-mode_context-mode__ctx_batch_execute",
	]);
	assert.equal(tool.promptSnippet, description);
});

void test("MCP tool name allocation qualifies only collisions", () => {
	const used = new Set(["read_file", "search", "server__search"]);
	assert.equal(
		allocateMcpToolName("ctx_execute", "context-mode", used),
		"ctx_execute",
	);
	assert.equal(
		allocateMcpToolName("search", "server", used),
		"server__search__2",
	);
});

void test("stdio MCP requests use newline-delimited JSON-RPC", () => {
	const encoded = encodeMcpMessage({
		jsonrpc: "2.0",
		id: 1,
		method: "initialize",
	});
	assert.equal(encoded.toString("utf8").endsWith("\n"), true);
	assert.equal(encoded.toString("utf8").includes("Content-Length"), false);
	assert.equal(tryDecodeMcpMessage(encoded)?.message.method, "initialize");
});

void test("MCP object results use compact JSON in model context", () => {
	assert.equal(
		formatMcpToolResult({
			items: [{ path: "src/index.ts", line: 42 }],
			more: 3,
		}),
		'{"items":[{"path":"src/index.ts","line":42}],"more":3}',
	);
	assert.equal(
		formatMcpToolResult({ isError: true, code: "failed" }),
		'Error: {"isError":true,"code":"failed"}',
	);
});
