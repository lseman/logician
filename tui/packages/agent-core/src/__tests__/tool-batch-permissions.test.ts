import assert from "node:assert/strict";
import { test } from "node:test";
import type { AgentEvent, Tool, ToolCall } from "../agent/types.ts";
import type { ExtensionEvent } from "../hooks/extensions/events.ts";
import { executeToolBatch } from "../runtime/tool-batch-controller.ts";
import { PermissionManager } from "../tools/shared/permissions.ts";
import { ToolRegistry } from "../tools/shared/registry.ts";

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
	permissions: PermissionManager,
	onPermissionRequest?: Parameters<
		typeof executeToolBatch
	>[0]["onPermissionRequest"],
) {
	const events: AgentEvent[] = [];
	return executeToolBatch({
		registry,
		toolCalls: [call],
		rawStopReason: "stop",
		iteration: 1,
		permissions,
		onPermissionRequest,
		emit: e => {
			events.push(e);
		},
		emitExtension: async (_e: ExtensionEvent) => {},
	});
}

void test("denied tool call short-circuits without executing", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionManager({
		mode: "acceptAll",
		rules: { deny: ["bash(rm *)"] },
	});
	const batch = await run(registry, callFor("rm -rf /tmp/x"), permissions);
	assert.equal(batch.messages.length, 1);
	const content = batch.messages[0].content as string;
	assert.match(content, /Tool call denied/);
	assert.match(content, /denied by rule/);
});

void test("plan mode denies write tools with the plan-mode reason", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionManager({ mode: "plan" });
	const batch = await run(registry, callFor("make build"), permissions);
	const content = batch.messages[0].content as string;
	assert.match(content, /Tool call denied/);
	assert.match(content, /plan mode/i);
});

void test("ask verdict with no handler fails closed (denied, not silently executed)", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionManager({ mode: "ask" });
	const batch = await run(registry, callFor("make build"), permissions);
	const content = batch.messages[0].content as string;
	assert.match(content, /Tool call denied/);
	assert.match(content, /no interactive handler/);
});

void test("ask verdict resolved 'allow' by the handler executes the tool", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionManager({ mode: "ask" });
	const batch = await run(
		registry,
		callFor("make build"),
		permissions,
		async () => "allow",
	);
	const content = batch.messages[0].content as string;
	assert.equal(content, "ran: make build");
});

void test("ask verdict resolved 'deny' by the handler blocks the tool", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionManager({ mode: "ask" });
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
	const permissions = new PermissionManager({ mode: "ask" });
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

	// Second call with the same tool name should now be auto-allowed without
	// asking again, via the session-allow rule persisted by "always".
	const second = await run(
		registry,
		callFor("make build"),
		permissions,
		onPermissionRequest,
	);
	assert.equal(second.messages[0].content, "ran: make build");
	assert.equal(asked, 1);
});

void test("acceptAll mode never invokes the permission handler", async () => {
	const registry = registryWithBash();
	const permissions = new PermissionManager({ mode: "acceptAll" });
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
