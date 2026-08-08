import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { AgentHarness } from "../agent/harness.ts";
import type { AgentConfig } from "../agent/types.ts";
import { ExtensionRunner } from "../extensions/runner.ts";
import { FakeBackend, textResponse } from "./fake-backend.ts";

function extensionFile(source: string): string {
	const dir = mkdtempSync(join(tmpdir(), "logician-extension-"));
	const file = join(dir, "extension.mjs");
	writeFileSync(file, source);
	return pathToFileURL(file).href;
}

function makeConfig(): AgentConfig {
	return {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		tools: [],
	};
}

void test("ExtensionRunner registers handlers centrally and unloads extension resources", async () => {
	const runner = new ExtensionRunner({ sessionId: "s1", cwd: "/tmp" });
	await runner.load([
		{
			name: "guard",
			source: "path",
			path: extensionFile(`
				export default function(api) {
					api.on("tool_execution_start", () => ({ block: true, reason: "blocked" }));
					api.registerTool({
						name: "ext_tool",
						description: "extension tool",
						parameters: { type: "object", properties: {} },
						execute: async () => ({ content: "ok" }),
					});
					api.registerCommand({
						name: "ext_command",
						description: "extension command",
						handler: () => "ok",
					});
				}
			`),
		},
	]);

	assert.equal(runner.hasHandlers("tool_execution_start"), true);
	assert.equal(runner.getTools().length, 1);
	assert.equal(runner.getCommands().length, 1);

	const result = await runner.getHooks()?.beforeToolCall?.({
		toolCall: { id: "1", name: "bash", arguments: "{}" },
		args: {},
		iteration: 1,
	});
	assert.equal(result?.content, "blocked");
	assert.equal(result?.isError, true);

	runner.destroy();
	assert.equal(runner.hasHandlers("tool_execution_start"), false);
	assert.equal(runner.getTools().length, 0);
	assert.equal(runner.getCommands().length, 0);
});

void test("AgentHarness applies extension pre-turn context, tools, and lifecycle events", async () => {
	const seenEvents: string[] = [];
	const runner = new ExtensionRunner({ sessionId: "s2", cwd: "/tmp" });
	await runner.load([
		{
			name: "turn-ext",
			source: "path",
			path: extensionFile(`
				export default function(api) {
					api.on("before_agent_start", () => ({
						messages: [{ role: "user", content: "extension context" }],
						systemPrompt: "extension system",
					}));
					api.on("message_end", (event) => {
						api.events.emit("seen", event.context.message?.content);
					});
					api.registerTool({
						name: "ext_echo",
						description: "echo from extension",
						parameters: { type: "object", properties: {} },
						execute: async () => ({ content: "extension tool result" }),
					});
				}
			`),
		},
	]);
	runner.events.on("seen", data => {
		seenEvents.push(String(data));
	});

	const backend = new FakeBackend([
		(messages, options) => {
			assert.equal(messages[0]?.content, "extension system");
			assert.ok(messages.some(m => m.content === "extension context"));
			assert.ok(
				options.tools?.some(
					tool =>
						(tool as { function?: { name?: string } }).function?.name ===
						"ext_echo",
				),
			);
			return {
				content: "calling tool",
				toolCalls: [{ id: "call_1", name: "ext_echo", arguments: "{}" }],
				stopReason: "stop",
			};
		},
		() => textResponse("done"),
	]);
	const harness = new AgentHarness({
		config: makeConfig(),
		backend,
		extensionRunner: runner,
		maxIterations: 3,
	});

	await harness.prompt("hello");
	assert.ok(harness.messages.some(m => m.content === "extension tool result"));
	assert.ok(seenEvents.includes("extension tool result"));
});
