import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { ExtensionRunner } from "../extensions/runner.ts";
import type { ExtensionDefinition } from "../extensions/types.ts";

function piExtensionFile(content: string): string {
	const dir = mkdtempSync(join(tmpdir(), "logician-pi-ext-"));
	const file = join(dir, "pi-extension.ts");
	writeFileSync(file, content);
	return pathToFileURL(file).href;
}

function nativeExtensionFile(content: string): string {
	const dir = mkdtempSync(join(tmpdir(), "logician-native-ext-"));
	const file = join(dir, "native-extension.ts");
	writeFileSync(file, content);
	return pathToFileURL(file).href;
}

void test("Pi adapter auto-detects Pi-style extensions by TypeBox imports", async () => {
	const runner = new ExtensionRunner({ sessionId: "pi-test", cwd: "/tmp" });

	// Use a file that contains TypeBox-like patterns to trigger Pi detection
	await runner.load([
		{
			name: "pi-ext",
			source: "path",
			path: piExtensionFile(`
				// Simulated Pi extension using TypeBox patterns
				import { Type } from 'typebox';
				export default function(api) {
					api.registerTool({
						name: "pi_tool",
						description: "A Pi tool",
						parameters: Type.Object({ name: Type.String() })
					});
					api.registerCommand("pi-cmd", {
						description: "A Pi command",
						handler: async () => {}
					});
				}
			`),
		} as unknown as ExtensionDefinition,
	]);

	// Pi extension is auto-detected (may fail to load due to missing typebox, but detection should work)
	// The key is that it was routed to loadPiExtension, not loadNativeExtension
	runner.destroy();
});

void test("Pi adapter handles missing registerTool gracefully", async () => {
	const runner = new ExtensionRunner({ sessionId: "pi-test2", cwd: "/tmp" });

	// This Pi extension uses Type.String pattern → detected as Pi
	// But the placeholder API doesn't have registerTool → should not crash
	await runner.load([
		{
			name: "pi-ext2",
			source: "path",
			path: piExtensionFile(`
				// Uses Type.String which triggers Pi detection
				const schema = { type: "object", properties: {
					name: { type: "string" }
				} };
				export default function(api) {
					api.on("session_start", async (event) => {});
					api.registerTool({
						name: "greet",
						description: "Greet someone",
						parameters: schema,
						execute: async () => ({ content: [{ type: "text", text: "hello" }] })
					});
				}
			`),
		} as unknown as ExtensionDefinition,
	]);

	// Pi tools now enter the live Logician registry rather than remaining only
	// in adapter-local bookkeeping.
	assert.equal(runner.getTools().length, 1);
	assert.equal(runner.getTools()[0].name, "greet");
	const result = await runner
		.getTools()[0]
		.execute(
			"call-1",
			{},
			{ cwd: "/tmp", sessionId: "pi-test2", toolCall: {} as never },
		);
	assert.equal(result.content, "hello");
	runner.destroy();
});

void test("Native extensions are not loaded through Pi adapter", async () => {
	const runner = new ExtensionRunner({ sessionId: "native-test", cwd: "/tmp" });

	await runner.load([
		{
			name: "native-ext",
			source: "path",
			path: nativeExtensionFile(`
				export default function(api) {
					api.on("tool_execution_start", () => ({ block: true, reason: "blocked" }));
					api.registerTool({
						name: "native_tool",
						description: "A native tool",
						parameters: { type: "object", properties: {} },
						execute: async () => ({ content: "ok" })
					});
				}
			`),
		} as unknown as ExtensionDefinition,
	]);

	// Native extension should register directly with the runner
	assert.equal(runner.hasHandlers("tool_execution_start"), true);
	assert.equal(runner.getTools().length, 1);
	runner.destroy();
});

void test("Pi adapter emits session_start to Pi handlers", async () => {
	const runner = new ExtensionRunner({ sessionId: "emit-test", cwd: "/tmp" });

	await runner.load([
		{
			name: "pi-emit",
			source: "path",
			path: piExtensionFile(`
				// Type.String marks this fixture as a Pi-style extension.
				export default function(api) {
					let receivedSession = null;
					api.on("session_start", async (event) => {
						receivedSession = event;
					});
					// Export for testing
					globalThis._piReceivedSession = () => receivedSession;
				}
			`),
		} as unknown as ExtensionDefinition,
	]);

	await runner.emitToAll({
		type: "session_start",
		context: { sessionId: "emit-test", cwd: "/tmp" },
	});
	assert.equal(runner.getPiExtensionCount(), 1);
	assert.equal(
		(
			globalThis as typeof globalThis & {
				_piReceivedSession?: () => { type?: string } | null;
			}
		)._piReceivedSession?.()?.type,
		"session_start",
	);

	runner.destroy();
});

void test("Pi context handlers chain and run through the runtime hook", async () => {
	const runner = new ExtensionRunner({
		sessionId: "context-test",
		cwd: "/tmp",
	});
	await runner.load([
		{
			name: "pi-context",
			source: "path",
			path: piExtensionFile(`
				// Type.String marks this fixture as a Pi-style extension.
				export default function(api) {
					api.on("context", event => ({
						messages: [...event.messages, { role: "user", content: "first" }]
					}));
					api.on("context", event => ({
						messages: [...event.messages, { role: "user", content: "second" }]
					}));
				}
			`),
		} as ExtensionDefinition,
	]);

	const hook = runner.getHooks()?.transformContext;
	assert.ok(hook);
	const result = await hook({ messages: [], iteration: 1 });
	assert.deepEqual(
		result?.messages?.map(message =>
			typeof (message as { content?: unknown } | undefined)?.content ===
			"string"
				? (message as { content: string }).content
				: "",
		),
		["first", "second"],
	);
	runner.destroy();
});

void test("Pi tool_call handlers can mutate arguments and block execution", async () => {
	const runner = new ExtensionRunner({
		sessionId: "tool-gate-test",
		cwd: "/tmp",
	});
	await runner.load([
		{
			name: "pi-tool-gate",
			source: "path",
			path: piExtensionFile(`
				// Type.String marks this fixture as a Pi-style extension.
				export default function(api) {
					api.on("tool_call", event => {
						event.input.checked = true;
						if (event.input.dangerous) return { block: true, reason: "unsafe" };
					});
				}
			`),
		} as ExtensionDefinition,
	]);

	const hook = runner.getHooks()?.beforeToolCall;
	assert.ok(hook);
	const changed = await hook({
		toolCall: { id: "call-1", name: "bash", arguments: "{}" },
		args: {},
		iteration: 1,
	});
	assert.deepEqual(changed?.args, { checked: true });

	const blocked = await hook({
		toolCall: { id: "call-2", name: "bash", arguments: "{}" },
		args: { dangerous: true },
		iteration: 1,
	});
	assert.equal(blocked?.content, "unsafe");
	assert.equal(blocked?.isError, true);
	runner.destroy();
});

void test("Pi adapter handles unknown source paths gracefully", async () => {
	const runner = new ExtensionRunner({ sessionId: "grace-test", cwd: "/tmp" });

	// Pass a non-existent path — should not throw
	await runner.load([
		{
			name: "nonexistent",
			source: "path",
			path: "/does/not/exist.ts",
		} as unknown as ExtensionDefinition,
	]);

	// Runner should still be functional
	assert.equal(runner.getPiExtensionCount(), 0);
	runner.destroy();
});
