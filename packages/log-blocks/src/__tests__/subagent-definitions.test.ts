import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type {
	AgentConfig,
	AgentEvent,
	GenerateOptions,
	LLMBackend,
	LLMResponse,
	Tool,
} from "@logician/log-core";
import { PermissionPolicy } from "@logician/log-core/permissions";
import {
	BUILTIN_AGENTS,
	createSpawnAgentsTool,
	createSpawnAgentTool,
	createSubagentConcurrencyLimiter,
	loadAgentDefinitions,
} from "../delegation/definitions.ts";

function mkAgentDir(): string {
	return mkdtempSync(path.join(tmpdir(), "logician-agents-"));
}

// ── loadAgentDefinitions ─────────────────────────────────────────────────

void test("loadAgentDefinitions returns just the builtins when no dirs are given", async () => {
	const defs = await loadAgentDefinitions([]);
	assert.deepEqual(
		defs.map(d => d.name).sort(),
		BUILTIN_AGENTS.map(d => d.name).sort(),
	);
});

void test("loadAgentDefinitions silently skips a directory that does not exist", async () => {
	const defs = await loadAgentDefinitions([
		"/definitely/does/not/exist/agents",
	]);
	assert.equal(defs.length, BUILTIN_AGENTS.length);
});

void test("loadAgentDefinitions parses a markdown file with YAML frontmatter into an agent", async () => {
	const dir = mkAgentDir();
	writeFileSync(
		path.join(dir, "reviewer.md"),
		[
			"---",
			"name: reviewer",
			"description: Reviews code for correctness.",
			"tools: [read_file, grep]",
			"model: claude-haiku-4-5-20251001",
			"max-turns: 8",
			"max-execution-seconds: 30",
			"max-tool-calls: 20",
			"---",
			"",
			"You are a meticulous code reviewer.",
		].join("\n"),
		"utf8",
	);
	try {
		const defs = await loadAgentDefinitions([dir]);
		const reviewer = defs.find(d => d.name === "reviewer");
		assert.ok(reviewer);
		assert.equal(reviewer.description, "Reviews code for correctness.");
		assert.deepEqual(reviewer.tools, ["read_file", "grep"]);
		assert.equal(reviewer.model, "claude-haiku-4-5-20251001");
		assert.equal(reviewer.maxIterations, 8);
		assert.equal(reviewer.maxExecutionTimeMs, 30_000);
		assert.equal(reviewer.maxToolCalls, 20);
		assert.equal(reviewer.prompt, "You are a meticulous code reviewer.");
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

void test("loadAgentDefinitions accepts a comma-separated tools string", async () => {
	const dir = mkAgentDir();
	writeFileSync(
		path.join(dir, "fixer.md"),
		[
			"---",
			"name: fixer",
			"description: Fixes bugs.",
			"tools: read_file, edit_file , grep",
			"---",
			"Fix the bug.",
		].join("\n"),
		"utf8",
	);
	try {
		const defs = await loadAgentDefinitions([dir]);
		const fixer = defs.find(d => d.name === "fixer");
		assert.deepEqual(fixer?.tools, ["read_file", "edit_file", "grep"]);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

void test("loadAgentDefinitions falls back to the filename as name and generic prompt when omitted", async () => {
	const dir = mkAgentDir();
	writeFileSync(
		path.join(dir, "no-name.md"),
		["---", "description: A nameless agent.", "---", ""].join("\n"),
		"utf8",
	);
	try {
		const defs = await loadAgentDefinitions([dir]);
		const found = defs.find(d => d.name === "no-name");
		assert.ok(found);
		assert.match(found.prompt, /subagent completing one delegated task/);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

void test("loadAgentDefinitions skips files missing both name and description", async () => {
	const dir = mkAgentDir();
	writeFileSync(
		path.join(dir, "invalid.md"),
		"---\n---\nNo frontmatter fields.",
		"utf8",
	);
	try {
		const defs = await loadAgentDefinitions([dir]);
		assert.equal(
			defs.find(d => d.description === ""),
			undefined,
		);
		assert.equal(defs.length, BUILTIN_AGENTS.length);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

void test("loadAgentDefinitions ignores non-.md files", async () => {
	const dir = mkAgentDir();
	writeFileSync(path.join(dir, "readme.txt"), "not an agent", "utf8");
	try {
		const defs = await loadAgentDefinitions([dir]);
		assert.equal(defs.length, BUILTIN_AGENTS.length);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

void test("loadAgentDefinitions lets later directories override earlier ones and builtins by name", async () => {
	const dirA = mkAgentDir();
	const dirB = mkAgentDir();
	writeFileSync(
		path.join(dirA, "general.md"),
		["---", "name: general", "description: First version.", "---", "V1"].join(
			"\n",
		),
		"utf8",
	);
	writeFileSync(
		path.join(dirB, "general.md"),
		["---", "name: general", "description: Second version.", "---", "V2"].join(
			"\n",
		),
		"utf8",
	);
	try {
		const defs = await loadAgentDefinitions([dirA, dirB]);
		const generals = defs.filter(d => d.name === "general");
		assert.equal(generals.length, 1);
		assert.equal(generals[0].description, "Second version.");
		assert.equal(generals[0].prompt, "V2");
	} finally {
		rmSync(dirA, { recursive: true, force: true });
		rmSync(dirB, { recursive: true, force: true });
	}
});

void test("loadAgentDefinitions skips a file with no frontmatter block (missing name/description)", async () => {
	const dir = mkAgentDir();
	writeFileSync(
		path.join(dir, "broken.md"),
		"not frontmatter at all, just text",
		"utf8",
	);
	try {
		const defs = await loadAgentDefinitions([dir]);
		assert.equal(defs.length, BUILTIN_AGENTS.length);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

// ── resolveChildTools (exercised indirectly via createSpawnAgentTool) ────

const baseConfig: AgentConfig = {
	baseUrl: "http://test",
	model: "fake",
	systemPrompt: "parent",
};

function toolNamesBackend(capture: { names?: string[] }): LLMBackend {
	return {
		model: "fake",
		withModel() {
			return this;
		},
		async generate(
			_messages: Record<string, unknown>[],
			options?: GenerateOptions,
		): Promise<LLMResponse> {
			capture.names = (options?.tools ?? []).map(
				t => (t as { function?: { name?: string } }).function?.name ?? "",
			);
			return { content: "done", toolCalls: [], stopReason: "stop" };
		},
	};
}

function fakeTool(name: string, readOnly = false): Tool {
	return {
		name,
		description: name,
		parameters: { type: "object", properties: {} },
		readOnly,
		execute: async () => "ok",
	};
}

void test("spawn_agent never forwards spawn_agent/spawn_agents to the child, even with no tool allowlist", async () => {
	const capture: { names?: string[] } = {};
	const parentTools = [
		fakeTool("read_file"),
		fakeTool("spawn_agent"),
		fakeTool("spawn_agents"),
	];
	const tool = createSpawnAgentTool({
		config: () => ({ ...baseConfig, tools: parentTools }),
		backend: toolNamesBackend(capture),
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
	});
	await tool.execute({ task: "do it", agent: "general" }, {});
	assert.deepEqual(capture.names, ["read_file"]);
});

void test("spawn_agent with the explorer definition restricts the child to read-only tools", async () => {
	const capture: { names?: string[] } = {};
	const parentTools = [
		fakeTool("read_file", true),
		fakeTool("grep", true),
		fakeTool("edit_file", false),
	];
	const tool = createSpawnAgentTool({
		config: () => ({ ...baseConfig, tools: parentTools }),
		backend: toolNamesBackend(capture),
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
	});
	await tool.execute({ task: "look around", agent: "explorer" }, {});
	assert.deepEqual(capture.names?.sort(), ["grep", "read_file"]);
});

void test("spawn_agent with a custom allowlist restricts the child to exactly those tools", async () => {
	const capture: { names?: string[] } = {};
	const parentTools = [
		fakeTool("read_file"),
		fakeTool("edit_file"),
		fakeTool("bash"),
	];
	const tool = createSpawnAgentTool({
		config: () => ({ ...baseConfig, tools: parentTools }),
		backend: toolNamesBackend(capture),
		agents: () => [
			{
				name: "editor-only",
				description: "Can only edit.",
				prompt: "Edit files.",
				tools: ["edit_file"],
			},
		],
		emit: () => {},
	});
	await tool.execute({ task: "edit it", agent: "editor-only" }, {});
	assert.deepEqual(capture.names, ["edit_file"]);
});

// ── Error paths not covered by delegation-runtime.test.ts ────────────────

void test("spawn_agent rejects a blank task before touching the backend", async () => {
	let called = false;
	const tool = createSpawnAgentTool({
		config: () => ({ ...baseConfig, tools: [] }),
		backend: {
			model: "fake",
			withModel() {
				return this;
			},
			async generate() {
				called = true;
				return { content: "done", toolCalls: [], stopReason: "stop" };
			},
		},
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
	});
	const result = await tool.execute({ task: "   " }, {});
	assert.equal(result, "Error: spawn_agent requires a task.");
	assert.equal(called, false);
});

void test("spawn_agent rejects an unknown agent name and lists the available ones", async () => {
	const tool = createSpawnAgentTool({
		config: () => ({ ...baseConfig, tools: [] }),
		backend: {
			model: "fake",
			withModel() {
				return this;
			},
			async generate() {
				return { content: "done", toolCalls: [], stopReason: "stop" };
			},
		},
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
	});
	const result = await tool.execute(
		{ task: "do it", agent: "nonexistent" },
		{},
	);
	assert.equal(typeof result, "string");
	assert.match(result as string, /Unknown agent "nonexistent"/);
	assert.match(result as string, /general/);
	assert.match(result as string, /explorer/);
});

void test("spawn_agents rejects a task entry using an unknown agent before spawning any", async () => {
	let called = false;
	const tool = createSpawnAgentsTool({
		config: () => ({ ...baseConfig, tools: [] }),
		backend: {
			model: "fake",
			withModel() {
				return this;
			},
			async generate() {
				called = true;
				return { content: "done", toolCalls: [], stopReason: "stop" };
			},
		},
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
	});
	const result = await tool.execute(
		{ tasks: [{ task: "ok" }, { task: "bad", agent: "nonexistent" }] },
		{},
	);
	assert.equal(typeof result, "string");
	assert.match(result as string, /tasks\[1\] uses unknown agent "nonexistent"/);
	assert.equal(called, false);
});

void test("spawn_agents rejects a task entry with a blank task string", async () => {
	const tool = createSpawnAgentsTool({
		config: () => ({ ...baseConfig, tools: [] }),
		backend: {
			model: "fake",
			withModel() {
				return this;
			},
			async generate() {
				return { content: "done", toolCalls: [], stopReason: "stop" };
			},
		},
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
	});
	const result = await tool.execute(
		{ tasks: [{ task: "ok" }, { task: "  " }] },
		{},
	);
	assert.equal(typeof result, "string");
	assert.match(result as string, /tasks\[1\] is invalid/);
});

void test("spawn_agent surfaces backend errors as an isError result instead of throwing", async () => {
	const events: Array<{ type: string }> = [];
	const tool = createSpawnAgentTool({
		config: () => ({ ...baseConfig, tools: [] }),
		backend: {
			model: "fake",
			withModel() {
				return this;
			},
			async generate() {
				throw new Error("backend exploded");
			},
		},
		agents: () => BUILTIN_AGENTS,
		emit: event => events.push(event),
	});
	const result = await tool.execute({ task: "do it" }, {});
	assert.equal(typeof result, "object");
	if (typeof result === "string") return;
	assert.equal(result.isError, true);
	assert.ok(events.some(e => e.type === "subagent_end"));
});

void test("subagents inherit the parent permission boundary", async () => {
	let executed = false;
	let calls = 0;
	const events: AgentEvent[] = [];
	const mutate: Tool = {
		name: "mutate",
		description: "mutate state",
		parameters: { type: "object", properties: {} },
		execute: async () => {
			executed = true;
			return "changed";
		},
	};
	const tool = createSpawnAgentTool({
		config: () => ({
			...baseConfig,
			tools: [mutate],
			permissions: new PermissionPolicy({ mode: "plan" }),
		}),
		backend: {
			model: "fake",
			withModel() {
				return this;
			},
			async generate() {
				calls++;
				return calls === 1
					? {
							content: null,
							toolCalls: [
								{ id: "child-call", name: "mutate", arguments: "{}" },
							],
							stopReason: "stop" as const,
						}
					: { content: "done", toolCalls: [], stopReason: "stop" as const };
			},
		},
		agents: () => BUILTIN_AGENTS,
		emit: event => events.push(event),
	});
	await tool.execute({ task: "change state", agent: "general" }, {});
	assert.equal(executed, false);
	assert.ok(
		events.some(
			event =>
				event.type === "subagent_event" &&
				event.event.type === "tool_permission_decision",
		),
	);
});

// ── spawn_agents: concurrency-limiter abort emits a synthetic subagent_end ──
// so the TUI's per-task status doesn't hang on "running" when a queued task
// never reaches _runSpawn's own subagent_end.

void test("spawn_agents emits subagent_end for a task aborted while queued on the concurrency limiter", async () => {
	const events: Array<Record<string, unknown>> = [];
	const controller = new AbortController();
	// Cap of 1: task 0 acquires the slot and blocks forever; task 1 queues.
	const limiter = createSubagentConcurrencyLimiter(1);
	let releaseFirst: (() => void) | undefined;
	const blocker = new Promise<void>(resolve => {
		releaseFirst = resolve;
	});

	const tool = createSpawnAgentsTool({
		config: () => ({ ...baseConfig, tools: [] }),
		backend: {
			model: "fake",
			withModel() {
				return this;
			},
			async generate() {
				await blocker;
				return { content: "done", toolCalls: [], stopReason: "stop" };
			},
		},
		agents: () => BUILTIN_AGENTS,
		emit: event => events.push(event as unknown as Record<string, unknown>),
		concurrencyLimiter: limiter,
	});

	const run = tool.execute(
		{ tasks: [{ task: "first" }, { task: "second" }] },
		{ signal: controller.signal },
	);

	// Let task 0 acquire the limiter slot and start blocking in generate(),
	// then abort before task 1 ever gets a turn.
	await new Promise(resolve => setTimeout(resolve, 10));
	controller.abort();
	releaseFirst?.();

	const result = await run;
	assert.equal(typeof result, "object");
	if (typeof result === "string") return;
	const details = result.details as {
		results: Array<{ index: number; isError: boolean }>;
	};
	const task1 = details.results.find(r => r.index === 1);
	assert.ok(task1);
	assert.equal(task1?.isError, true);

	const task1End = events.find(
		e => e.type === "subagent_end" && e.taskIndex === 1,
	);
	assert.ok(
		task1End,
		"expected a subagent_end event for the aborted queued task",
	);
	assert.equal(task1End?.isError, true);
});
