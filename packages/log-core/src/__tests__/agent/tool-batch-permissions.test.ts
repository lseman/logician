import { test } from "bun:test";
import assert from "node:assert/strict";
import { PermissionPolicy } from "../../capabilities/tools/permissions.ts";
import { ToolRegistry } from "../../capabilities/tools/registry.ts";
import { executeToolBatch } from "../../runtime/execution/tool-batch-controller.ts";
import type {
	AgentEvent,
	AgentHooks,
	Tool,
	ToolCall,
} from "../../system/types/types-messages.ts";

function registryWithBash(): ToolRegistry {
	const registry = new ToolRegistry({ cache: null });
	const bash: Tool = {
		name: "bash",
		label: "Bash",
		description: "run a shell command",
		readOnly: false,
		parameters: { type: "object", properties: {} },
		execute: async args => ({
			content: `ran: ${(args as { command?: string }).command}`,
		}),
	};
	registry.register(bash);
	return registry;
}

function callFor(command: string): ToolCall {
	return { id: "c1", name: "bash", arguments: JSON.stringify({ command }) };
}

async function run(
	registry: ToolRegistry,
	call: ToolCall,
	permissions: PermissionPolicy,
	onPermissionRequest?: Parameters<
		typeof executeToolBatch
	>[0]["onPermissionRequest"],
	hooks?: AgentHooks,
) {
	const events: AgentEvent[] = [];
	const batch = await executeToolBatch({
		registry,
		toolCalls: [call],
		rawStopReason: "stop",
		iteration: 1,
		permissions,
		onPermissionRequest,
		hooks,
		emit: e => {
			events.push(e);
		},
	});
	return { ...batch, events };
}

void test("permissions evaluate rewritten tool arguments", async () => {
	const permissions = new PermissionPolicy({
		mode: "acceptAll",
		rules: { deny: ["bash(rm *)"] },
	});
	const batch = await run(
		registryWithBash(),
		callFor("echo safe"),
		permissions,
		undefined,
		{
			beforeToolCall: ({ args }) => ({
				args: { ...args, command: "rm -rf /tmp/rewrite" },
			}),
		},
	);
	assert.match(String(batch.messages[0].content), /Tool call denied/);
	const start = batch.events.find(
		event => event.type === "tool_execution_start",
	);
	assert.equal(
		start?.type === "tool_execution_start" ? start.args.command : undefined,
		"rm -rf /tmp/rewrite",
	);
});

void test("denied tool call short-circuits without executing", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionPolicy({
		mode: "acceptAll",
		rules: { deny: ["bash(rm *)"] },
	});
	const batch = await run(registry, callFor("rm -rf /tmp/x"), permissions);
	assert.equal(batch.messages.length, 1);
	const content = batch.messages[0].content as string;
	assert.match(content, /Tool call denied/);
	assert.match(content, /denied by rule/);
	assert.equal(batch.permissionDenials, 1);
});

void test("plan mode denies write tools with the plan-mode reason", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionPolicy({ mode: "plan" });
	const batch = await run(registry, callFor("make build"), permissions);
	const content = batch.messages[0].content as string;
	assert.match(content, /Tool call denied/);
	assert.match(content, /plan mode/i);
});

void test("ask verdict with no handler fails closed (denied, not silently executed)", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionPolicy({ mode: "ask" });
	const batch = await run(registry, callFor("make build"), permissions);
	const content = batch.messages[0].content as string;
	assert.match(content, /Tool call denied/);
	assert.match(content, /no interactive handler/);
});

void test("ask verdict resolved 'allow' by the handler executes the tool", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionPolicy({ mode: "ask" });
	const batch = await run(
		registry,
		callFor("make build"),
		permissions,
		async () => "allow",
	);
	const content = batch.messages[0].content as string;
	assert.equal(content, "ran: make build");
	assert.equal(batch.permissionDenials, 0);
});

void test("ask verdict resolved 'deny' by the handler blocks the tool", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionPolicy({ mode: "ask" });
	const batch = await run(
		registry,
		callFor("make build"),
		permissions,
		async () => "deny",
	);
	const content = batch.messages[0].content as string;
	assert.match(content, /Tool call denied/);
	assert.match(content, /user denied/);
});

void test("ask verdict resolved 'always' persists a session allow for later calls", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionPolicy({ mode: "ask" });
	let asked = 0;
	const onPermissionRequest = async () => {
		asked++;
		return "always" as const;
	};
	const first = await run(
		registry,
		callFor("make build"),
		permissions,
		onPermissionRequest,
	);
	assert.equal(first.messages[0].content, "ran: make build");
	assert.equal(asked, 1);

	// The same call should be auto-allowed by the scoped session rule.
	const second = await run(
		registry,
		callFor("make build"),
		permissions,
		onPermissionRequest,
	);
	assert.equal(second.messages[0].content, "ran: make build");
	assert.equal(asked, 1);

	const different = await run(
		registry,
		callFor("make clean"),
		permissions,
		onPermissionRequest,
	);
	assert.equal(different.messages[0].content, "ran: make clean");
	assert.equal(asked, 2);
});

void test("acceptAll mode never invokes the permission handler", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionPolicy({ mode: "acceptAll" });
	let asked = false;
	const batch = await run(
		registry,
		callFor("make build"),
		permissions,
		async () => {
			asked = true;
			return "allow";
		},
	);
	assert.equal(asked, false);
	assert.equal(batch.messages[0].content, "ran: make build");
});

void test("permission decisions are attributed before execution", async () => {
	const { events } = await run(
		registryWithBash(),
		callFor("make build"),
		new PermissionPolicy({ mode: "ask" }),
		async () => "always",
	);
	const decisions = events.filter(e => e.type === "tool_permission_decision");
	assert.deepEqual(decisions, [
		{
			type: "tool_permission_decision",
			toolCallId: "c1",
			toolName: "bash",
			decision: "always",
			source: "user",
		},
	]);
});
