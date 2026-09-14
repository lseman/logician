import { expect, test } from "bun:test";
import { PermissionPolicy } from "../../capabilities/tools/permissions.ts";
import { ToolRegistry } from "../../capabilities/tools/registry.ts";
import { executeToolBatch } from "../../runtime/execution/tool-batch-controller.ts";
import type { Tool } from "../../system/types/types-messages.ts";

const call = { id: "call-1", name: "transport", arguments: "{}" };
function transport(target: string): Tool {
	return {
		name: "transport",
		description: "Transport",
		parameters: {},
		resolveCall: () => ({ name: target, arguments: { command: "hello" } }),
		execute: async () => {
			throw new Error("Transport must not execute");
		},
	};
}

test("redirects are prepared before target hooks and permissions", async () => {
	let runs = 0;
	const seen: string[] = [];
	const registry = new ToolRegistry();
	registry.registerMany([
		transport("target"),
		{
			name: "target",
			description: "Target",
			parameters: {},
			prepareArguments: raw => ({
				command: String((raw as Record<string, unknown>).command).toUpperCase(),
			}),
			execute: async () => {
				runs++;
				return "ran";
			},
		},
	]);
	const output = await executeToolBatch({
		registry,
		toolCalls: [call],
		rawStopReason: "stop",
		iteration: 1,
		permissions: new PermissionPolicy({
			mode: "acceptAll",
			rules: { deny: ["target(HELLO)"] },
		}),
		hooks: {
			beforeToolCall: ({ toolCall, args }) => {
				seen.push(`${toolCall.name}:${args.command}`);
			},
		},
		emit: () => {},
	});
	expect(seen).toEqual(["target:HELLO"]);
	expect(runs).toBe(0);
	expect(output.permissionDenials).toBe(1);
	expect(output.messages[0]?.tool_call_id).toBe("call-1");
});

test("plan mode allows read-only targets and denies mutations behind a transport", async () => {
	for (const readOnly of [true, false]) {
		let runs = 0;
		const registry = new ToolRegistry();
		registry.registerMany([
			transport("target"),
			{
				name: "target",
				description: "Target",
				parameters: {},
				readOnly,
				execute: async () => {
					runs++;
					return "ran";
				},
			},
		]);
		const output = await executeToolBatch({
			registry,
			toolCalls: [call],
			rawStopReason: "stop",
			iteration: 1,
			permissions: new PermissionPolicy({ mode: "plan" }),
			emit: () => {},
		});
		expect(output.permissionDenials).toBe(readOnly ? 0 : 1);
		expect(runs).toBe(readOnly ? 1 : 0);
	}
});

test("missing targets, redirect cycles, and preparation errors fail closed", async () => {
	const registry = new ToolRegistry();
	registry.register(transport("missing"));
	expect((await registry.execute(call)).isError).toBe(true);
	registry.register(transport("transport"));
	expect((await registry.execute(call)).content).toContain("redirect cycle");
	registry.register({
		...transport("missing"),
		prepareArguments: () => {
			throw new Error("invalid payload");
		},
	});
	expect((await registry.execute(call)).content).toContain("invalid payload");
});

test("redirected sequential targets remain ordering barriers in parallel batches", async () => {
	let active = 0;
	let maxActive = 0;
	const registry = new ToolRegistry();
	registry.registerMany([
		transport("target"),
		{
			name: "target",
			description: "Target",
			parameters: {},
			executionMode: "sequential",
			execute: async () => {
				active++;
				maxActive = Math.max(maxActive, active);
				await Promise.resolve();
				active--;
				return "done";
			},
		},
	]);
	const output = await executeToolBatch({
		registry,
		toolCalls: [call, { ...call, id: "call-2" }],
		rawStopReason: "stop",
		iteration: 1,
		toolExecution: "parallel",
		emit: () => {},
	});
	expect(output.executedToolCallIds).toEqual(["call-1", "call-2"]);
	expect(maxActive).toBe(1);
});
