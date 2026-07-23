import assert from "node:assert/strict";
import { test } from "node:test";
import {
	createMcpTool,
	encodeMcpMessage,
	tryDecodeMcpMessage,
} from "../mcp/client.ts";

void test("plugin MCP tool names preserve Context Mode's declared namespace", () => {
	const client = {
		name: "plugin_context-mode_context-mode",
		callTool: async () => ({}),
	};
	const tool = createMcpTool(client as never, {
		name: "ctx_batch_execute",
		description: "Gather without flooding context",
		inputSchema: { type: "object", properties: {} },
	}) as { name: string };

	assert.equal(
		tool.name,
		"mcp__plugin_context-mode_context-mode__ctx_batch_execute",
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
