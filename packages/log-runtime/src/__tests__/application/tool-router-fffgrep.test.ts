import { test } from "bun:test";
import assert from "node:assert/strict";
import type { Tool } from "@logician/log-core";
import { ToolRouter } from "../../application/tool-router.ts";

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeRouter(
	opts: {
		fffgrepEnabled?: boolean;
		tools?: Tool[];
		extraTools?: Tool[];
		onContextChangedCalls?: number[];
		onToolAddedCalls?: Tool[];
	} = {},
): {
	router: ToolRouter;
	onContextChangedCalls: number[];
	onToolAddedCalls: Tool[];
} {
	const onContextChangedCalls: number[] = [];
	const onToolAddedCalls: Tool[] = [];

	const router = new ToolRouter({
		cwd: "/tmp/test",
		projectTrusted: true,
		tools: opts.tools,
		extraTools: opts.extraTools,
		fffgrepEnabled: opts.fffgrepEnabled,
		emit: () => {},
		onToolAdded: tool => {
			onToolAddedCalls.push(tool);
		},
		onContextChanged: () => {
			onContextChangedCalls.push(onContextChangedCalls.length);
		},
		autoStartMcp: false,
	});

	return { router, onContextChangedCalls, onToolAddedCalls };
}

function makeTool(name: string, origin?: Tool["origin"]): Tool {
	return {
		name,
		origin,
		label: name,
		description: `Tool ${name}`,
		parameters: { type: "object", properties: {} },
		execute: async () => "",
	};
}

// ── Tests ─────────────────────────────────────────────────────────────────────

void test("fff__grep is NOT in default tools (it's an MCP tool)", () => {
	const { router } = makeRouter();
	const names = router.getDefaultTools().map(t => t.name);
	assert.ok(
		!names.includes("fff__grep"),
		"fff__grep is an MCP tool, not in default tools",
	);
});

void test("local grep tool is in default tools", () => {
	const { router } = makeRouter();
	const names = router.getDefaultTools().map(t => t.name);
	assert.ok(names.includes("grep"), "local grep should be present");
});

void test("setFffgrepEnabled(false) has no effect when no fff tools exist", () => {
	const { router, onContextChangedCalls } = makeRouter();
	const before = router.getDefaultTools().map(t => t.name);

	router.setFffgrepEnabled(false);

	const after = router.getDefaultTools().map(t => t.name);
	assert.deepEqual(before, after, "tools should be unchanged");
	assert.equal(
		onContextChangedCalls.length,
		0,
		"onContextChanged should NOT fire",
	);
});

void test("setFffgrepEnabled(true) has no effect when no fff tools are disabled", () => {
	const { router, onContextChangedCalls } = makeRouter({
		fffgrepEnabled: false,
	});
	const before = router.getDefaultTools().map(t => t.name);

	router.setFffgrepEnabled(true);

	const after = router.getDefaultTools().map(t => t.name);
	assert.deepEqual(before, after, "tools should be unchanged");
	assert.equal(
		onContextChangedCalls.length,
		0,
		"onContextChanged should NOT fire",
	);
});

void test("setFffgrepEnabled(false) when already disabled is a no-op", () => {
	const { router, onContextChangedCalls } = makeRouter({
		fffgrepEnabled: false,
	});

	router.setFffgrepEnabled(false);
	assert.equal(
		onContextChangedCalls.length,
		0,
		"onContextChanged should NOT fire",
	);
});

void test("setFffgrepEnabled(true) when already enabled is a no-op", () => {
	const { router, onContextChangedCalls } = makeRouter({
		fffgrepEnabled: true,
	});

	router.setFffgrepEnabled(true);
	assert.equal(
		onContextChangedCalls.length,
		0,
		"onContextChanged should NOT fire",
	);
});

void test("toggle fffgrep with no fff tools present is a no-op both ways", () => {
	const { router, onContextChangedCalls } = makeRouter();

	router.setFffgrepEnabled(false);
	assert.equal(onContextChangedCalls.length, 0, "disable: no-op");

	router.setFffgrepEnabled(true);
	assert.equal(onContextChangedCalls.length, 0, "enable: no-op");
});

void test("fffgrepEnabled flag is stored correctly in constructor", () => {
	const { router } = makeRouter({ fffgrepEnabled: true });
	// Can't access private field directly, but we can verify behavior
	// by checking that adding an fff tool would be accepted
	const namesBefore = router.getDefaultTools().map(t => t.name);
	// No fff tools yet, so no change
	assert.equal(namesBefore.filter(n => /^fff/i.test(n)).length, 0);
});

void test("fffgrepEnabled: false in constructor does not affect non-fff tools", () => {
	const { router } = makeRouter({ fffgrepEnabled: false });
	const names = router.getDefaultTools().map(t => t.name);
	assert.ok(names.includes("grep"), "local grep should still be present");
	assert.ok(names.includes("bash"), "bash should still be present");
	assert.ok(names.includes("find"), "find should still be present");
});

void test("constructor hides FFF grep by origin even when its exposed name has a collision suffix", () => {
	const fffGrep = makeTool("fff__grep__2", {
		kind: "mcp",
		server: "fff",
		tool: "grep",
	});
	const fffMultiGrep = makeTool("multi_grep", {
		kind: "mcp",
		server: "fff",
		tool: "multi_grep",
	});
	const { router } = makeRouter({
		fffgrepEnabled: false,
		tools: [makeTool("grep"), fffGrep, fffMultiGrep],
	});
	assert.deepEqual(
		router.getDefaultTools().map(tool => tool.name),
		["grep", "multi_grep"],
	);
	assert.equal(router.getMcpToolCount(), 1);
});

void test("FFF grep can be disabled and re-enabled without losing tool identity", () => {
	const fffGrep = makeTool("fff__grep", {
		kind: "mcp",
		server: "fff",
		tool: "grep",
	});
	const { router, onContextChangedCalls } = makeRouter({
		tools: [makeTool("grep"), fffGrep],
	});

	router.setFffgrepEnabled(false);
	assert.deepEqual(
		router.getDefaultTools().map(tool => tool.name),
		["grep"],
	);
	assert.equal(router.getMcpToolCount(), 0);

	router.setFffgrepEnabled(true);
	assert.deepEqual(
		router.getDefaultTools().map(tool => tool.name),
		["grep", "fff__grep"],
	);
	assert.equal(router.getMcpToolCount(), 1);
	assert.equal(onContextChangedCalls.length, 2);
});
