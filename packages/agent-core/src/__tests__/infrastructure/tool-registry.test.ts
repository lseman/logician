import { test } from "bun:test";
import assert from "node:assert/strict";
import type { Tool } from "@logician/agent-core";
import { microCompactCompactableMessages } from "../../core/compaction/engine.ts";
import { ToolRegistry } from "../../core/tools/registry.ts";

function makeTool(overrides: Partial<Tool> & { name: string }): Tool {
	return {
		description: "test tool",
		parameters: { type: "object", properties: {} },
		execute: async () => "ok",
		...overrides,
	} as Tool;
}

function call(name: string, args: Record<string, unknown> = {}) {
	return { id: "call_1", name, arguments: JSON.stringify(args) };
}

void test("registry does not cache tools without cacheable flag", async () => {
	let executions = 0;
	const registry = new ToolRegistry();
	registry.register(
		makeTool({
			name: "counter",
			execute: async () => `run ${++executions}`,
		}),
	);

	const first = await registry.execute(call("counter"));
	const second = await registry.execute(call("counter"));
	assert.equal(first.content, "run 1");
	assert.equal(second.content, "run 2");
	assert.equal(executions, 2);
});

void test("registry caches tools that opt in via cacheable", async () => {
	let executions = 0;
	const registry = new ToolRegistry();
	registry.register(
		makeTool({
			name: "pure",
			cacheable: true,
			execute: async () => `run ${++executions}`,
		}),
	);

	const first = await registry.execute(call("pure"));
	const second = await registry.execute(call("pure"));
	assert.equal(first.content, "run 1");
	assert.equal(second.content, "run 1");
	assert.equal(executions, 1);
});

void test("registry times out hung tools", async () => {
	const registry = new ToolRegistry();
	registry.register(
		makeTool({
			name: "hang",
			timeoutMs: 50,
			execute: () => new Promise(() => {}),
		}),
	);

	const result = await registry.execute(call("hang"));
	assert.equal(result.isError, true);
	assert.match(result.content, /timed out/);
});

void test("registry aborts a tool when its timeout expires", async () => {
	let aborted = false;
	const registry = new ToolRegistry();
	registry.register(
		makeTool({
			name: "abortable",
			timeoutMs: 20,
			execute: async (_args, ctx) =>
				new Promise<string>(resolve => {
					ctx.signal?.addEventListener(
						"abort",
						() => {
							aborted = true;
							resolve("aborted");
						},
						{ once: true },
					);
				}),
		}),
	);

	const result = await registry.execute(call("abortable"));
	assert.equal(result.isError, true);
	assert.equal(aborted, true);
});

void test("registry disables caching when cache is null", async () => {
	let executions = 0;
	const registry = new ToolRegistry({ cache: null });
	registry.register(
		makeTool({
			name: "uncached",
			cacheable: true,
			execute: async () => `run ${++executions}`,
		}),
	);

	await registry.execute(call("uncached"));
	await registry.execute(call("uncached"));
	assert.equal(executions, 2);
});

void test("registry honors the configured cache size", async () => {
	let executions = 0;
	const registry = new ToolRegistry({ cacheSize: 1 });
	registry.register(
		makeTool({
			name: "bounded",
			cacheable: true,
			execute: async () => `run ${++executions}`,
		}),
	);

	await registry.execute(call("bounded", { key: "a" }));
	await registry.execute(call("bounded", { key: "b" }));
	await registry.execute(call("bounded", { key: "a" }));
	assert.equal(executions, 3);
});

void test("registry caps oversized tool results with middle truncation", async () => {
	const registry = new ToolRegistry({ maxResultChars: 1000 });
	const big = "A".repeat(600) + "MIDDLE".repeat(200) + "Z".repeat(600);
	registry.register(makeTool({ name: "flood", execute: async () => big }));

	const result = await registry.execute(call("flood"));
	assert.ok(result.content.length < big.length);
	assert.match(result.content, /truncated/);
	assert.ok(result.content.startsWith("A"));
	assert.ok(result.content.endsWith("Z"));
});

void test("registry propagates path policy to tool contexts", async () => {
	const registry = new ToolRegistry({
		cwd: "/workspace",
		allowedPaths: ["/shared"],
		allowAllPaths: true,
	});
	registry.register(
		makeTool({
			name: "context",
			execute: async (_args, ctx) =>
				JSON.stringify({
					cwd: ctx.cwd,
					allowedPaths: ctx.allowedPaths,
					allowAllPaths: ctx.allowAllPaths,
				}),
		}),
	);

	const result = await registry.execute(call("context"));
	assert.deepEqual(JSON.parse(result.content), {
		cwd: "/workspace",
		allowedPaths: ["/shared"],
		allowAllPaths: true,
	});
});

void test("micro-compaction spares recent messages and user prompts", () => {
	const oldToolResult = {
		role: "tool",
		content: "T".repeat(10_000),
	};
	const oldUserPrompt = {
		role: "user",
		content: "U".repeat(10_000),
	};
	const messages = [
		oldUserPrompt,
		oldToolResult,
		...Array.from({ length: 6 }, (_, i) => ({
			role: "tool",
			content: `${"R".repeat(10_000)}#${i}`,
		})),
	];

	const result = microCompactCompactableMessages(messages);

	// Old tool result trimmed, both head and tail preserved.
	const trimmedTool = String(result.messages[1].content);
	assert.ok(trimmedTool.length < 10_000);
	assert.match(trimmedTool, /compacted/);
	assert.ok(trimmedTool.startsWith("T"));
	assert.ok(trimmedTool.endsWith("T"));

	// Old user prompt under its 14k cap — untouched.
	assert.equal(String(result.messages[0].content).length, 10_000);

	// Recent messages untouched even when oversized.
	for (let i = 2; i < result.messages.length; i++) {
		assert.equal(String(result.messages[i].content).length, 10_002);
	}
});

void test("provider tool definitions normalize external JSON Schema dialects", () => {
	const registry = new ToolRegistry();
	registry.register({
		name: "external_tool",
		description: "External tool",
		parameters: {
			$schema: "https://json-schema.org/draft/2020-12/schema",
			$defs: { value: { type: ["string", "null"], format: "uri" } },
			type: "object",
			properties: {
				url: {
					$ref: "#/$defs/value",
					pattern:
						"^(?:(?:\\d\\d[2468][048])-02-29|\\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\\d|3[01]))$",
					propertyNames: { pattern: "^[a-z]+$" },
				},
				mode: { const: "safe" },
			},
			unevaluatedProperties: false,
			required: ["url", "missing"],
		},
		execute: async () => ({ content: "ok" }),
	});

	const definition = registry.toToolDefinitions()[0] as {
		function: { parameters: Record<string, unknown> };
	};
	assert.deepEqual(definition.function.parameters, {
		type: "object",
		properties: {
			url: { type: "string" },
			mode: { enum: ["safe"] },
		},
		required: ["url"],
	});
});
