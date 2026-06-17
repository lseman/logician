import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { AgentLoop } from "../core/loop.ts";
import { task_status } from "../tools/skills/task-status.ts";
import type { AgentConfig, Tool } from "../core/types.ts";
import { FakeBackend, textResponse } from "./fake-backend.ts";

const noop: Tool = {
	name: "noop",
	description: "does nothing",
	parameters: { type: "object", properties: {} },
	execute: async () => "ok",
};

function makeConfig(backendOverrides: Partial<AgentConfig> = {}): AgentConfig {
	return {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		tools: [noop, task_status],
		...backendOverrides,
	};
}

void test("mid-stream interrupt keeps the partial text and continues the run", async () => {
	const backend = new FakeBackend([
		// First call: stream a partial delta, then hang until aborted.
		(_messages, options) => {
			options.callbacks?.onDelta?.("partial answer");
			return new Promise((_, reject) => {
				options.signal?.addEventListener("abort", () =>
					reject(new Error("aborted by client")),
				);
			});
		},
		// Second call (after the interrupt + steering drain): finish normally.
		() => textResponse("final answer"),
	]);

	let followUpDrained = 0;
	const loop = new AgentLoop({
		config: makeConfig({
			continuationEnabled: false,
			hooks: {
				// Steering message waiting at the follow-up drain keeps the loop
				// going after the interrupted (tool-less) turn.
				getFollowUpMessages: () => {
					if (followUpDrained === 0) {
						followUpDrained++;
						return [{ role: "user", content: "steered: be brief" }];
					}
					return undefined;
				},
			},
		}),
		backend,
	});

	// Interrupt once the first call is in flight.
	setTimeout(() => loop.interruptTurn(), 25);
	const messages = await loop.run("question");

	const texts = messages.map((m) => `${m.role}:${m.content ?? ""}`);
	assert.ok(
		texts.includes("assistant:partial answer"),
		`partial kept: ${JSON.stringify(texts)}`,
	);
	assert.ok(texts.includes("user:steered: be brief"));
	assert.ok(texts.includes("assistant:final answer"));
	assert.equal(backend.calls, 2);
});

void test("task_status terminates the run without a follow-up nudge", async () => {
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [
				{
					id: "c1",
					name: "task_status",
					arguments: JSON.stringify({ status: "done", summary: "all good" }),
				},
			],
			stopReason: "stop",
		}),
		// Would only be reached if the loop failed to terminate.
		() => textResponse("should never happen"),
	]);
	const loop = new AgentLoop({ config: makeConfig(), backend });
	const messages = await loop.run("do the thing");
	assert.equal(backend.calls, 1);
	const toolResult = messages.find((m) => m.role === "tool");
	assert.match(toolResult?.content ?? "", /done/);
});

void test("Stop plugin hook can block stop and continue with hook context", async () => {
	const root = mkdtempSync(join(tmpdir(), "logician-stop-hook-"));
	const pluginDir = join(root, "plugin");
	mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
	mkdirSync(join(pluginDir, "hooks"), { recursive: true });
	const hookScript = join(pluginDir, "stop-hook.mjs");
	const callsFile = join(root, "calls.txt");
	writeFileSync(
		join(pluginDir, ".claude-plugin", "plugin.json"),
		JSON.stringify({
			name: "stopper",
			version: "1.0.0",
			hooks: "hooks/hooks.json",
		}),
	);
	writeFileSync(
		join(pluginDir, "hooks", "hooks.json"),
		JSON.stringify({
			hooks: {
				Stop: [
					{
						hooks: [
							{
								type: "command",
								command: `node ${JSON.stringify(hookScript)}`,
							},
						],
					},
				],
			},
		}),
	);
	writeFileSync(
		hookScript,
		`
import { appendFileSync } from "node:fs";
const input = await new Promise((resolve) => {
  let data = "";
  process.stdin.on("data", (chunk) => data += chunk);
  process.stdin.on("end", () => resolve(data));
});
const payload = JSON.parse(input || "{}");
appendFileSync(${JSON.stringify(callsFile)}, String(payload.stop_hook_active) + "\\n");
if (!payload.stop_hook_active) {
  console.log(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "Stop",
      decision: "block",
      additionalContext: "Stop hook says one more step is required."
    }
  }));
}
`,
	);
	writeFileSync(
		join(root, "installed_plugins.json"),
		JSON.stringify({
			version: 2,
			plugins: {
				"stopper@test": [
					{
						scope: "user",
						installPath: pluginDir,
						version: "1.0.0",
						installedAt: new Date().toISOString(),
						lastUpdated: new Date().toISOString(),
						enabled: true,
					},
				],
			},
		}),
	);

	const previousCache = process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR;
	process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR = root;
	try {
		const backend = new FakeBackend([
			() => textResponse("I think this is done."),
			(messages) => {
				assert.ok(
					messages.some((m) =>
						String(m.content ?? "").includes(
							"Stop hook says one more step is required.",
						),
					),
				);
				return textResponse("final answer after hook");
			},
			() => textResponse("should never happen"),
		]);
		const loop = new AgentLoop({
			config: makeConfig({
				runtimeHooksEnabled: true,
				maxContinuations: 3,
			}),
			backend,
		});
		const messages = await loop.run("finish the task");

		assert.equal(backend.calls, 2);
		assert.ok(messages.some((m) => m.content === "final answer after hook"));
		assert.deepEqual(readFileSync(callsFile, "utf8").trim().split("\n"), [
			"false",
			"true",
		]);
	} finally {
		if (previousCache === undefined)
			delete process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR;
		else process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR = previousCache;
	}
});

void test("permission deny records an error tool result instead of executing", async () => {
	const { PermissionManager } = await import("../tools/shared/permissions.ts");
	let executed = false;
	const spy: Tool = {
		...noop,
		name: "writer",
		execute: async () => {
			executed = true;
			return "wrote";
		},
	};
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [{ id: "c1", name: "writer", arguments: "{}" }],
			stopReason: "stop",
		}),
		() => textResponse("understood"),
	]);
	const loop = new AgentLoop({
		config: makeConfig({
			tools: [spy],
			permissions: new PermissionManager({
				mode: "acceptAll",
				rules: { deny: ["writer"] },
			}),
		}),
		backend,
	});
	const messages = await loop.run("write something");
	assert.equal(executed, false);
	const toolResult = messages.find((m) => m.role === "tool");
	assert.match(toolResult?.content ?? "", /Permission denied/);
});

void test("budget_exhausted stops the run between turns", async () => {
	const events: string[] = [];
	const backend = new FakeBackend([
		// Tool call so the loop wants to continue, with reported usage.
		() => ({
			content: "",
			toolCalls: [{ id: "c1", name: "noop", arguments: "{}" }],
			stopReason: "stop",
			usage: { totalTokens: 5000 },
		}),
		() => textResponse("should never happen"),
	]);
	const loop = new AgentLoop({
		config: makeConfig({
			maxTotalTokens: 1000,
			onEvent: (e) => events.push(e.type),
		}),
		backend,
	});
	await loop.run("go");
	assert.equal(backend.calls, 1);
	assert.ok(events.includes("budget_exhausted"));
});

void test("events carry a monotonic seq and timestamp", async () => {
	const seqs: number[] = [];
	const backend = new FakeBackend([() => textResponse("hi")]);
	const loop = new AgentLoop({
		config: makeConfig({
			onEvent: (e) => {
				if (e.seq !== undefined) seqs.push(e.seq);
				assert.ok(e.ts === undefined || e.ts > 0);
			},
		}),
		backend,
	});
	await loop.run("q");
	assert.ok(seqs.length > 3);
	for (let i = 1; i < seqs.length; i++) assert.ok(seqs[i] > seqs[i - 1]);
});

void test("auto_retry_end reports failure when retries keep failing", async () => {
	const { BackendError } = await import("../core/backend.ts");
	const events: Array<{ type: string; success?: boolean }> = [];
	const fail = () => {
		throw new BackendError({ category: "transient", message: "fetch failed" });
	};
	const backend = new FakeBackend([fail, fail, fail]);
	const loop = new AgentLoop({
		config: makeConfig({
			maxRetries: 2,
			retryBaseDelayMs: 1,
			onEvent: (e) =>
				events.push({
					type: e.type,
					success: e.type === "auto_retry_end" ? e.success : undefined,
				}),
		}),
		backend,
	});
	const messages = await loop.run("q");

	const starts = events.filter((e) => e.type === "auto_retry_start");
	const ends = events.filter((e) => e.type === "auto_retry_end");
	assert.equal(starts.length, 2);
	assert.equal(ends.length, 2);
	assert.ok(
		ends.every((e) => e.success === false),
		`no false-positive successes: ${JSON.stringify(ends)}`,
	);
	assert.ok(events.some((e) => e.type === "error"));
	const last = messages.at(-1);
	assert.match(last?.content ?? "", /Model request failed/);
});

void test("auto_retry_end reports success only after the retried request works", async () => {
	const { BackendError } = await import("../core/backend.ts");
	const events: Array<{ type: string; success?: boolean }> = [];
	const backend = new FakeBackend([
		() => {
			throw new BackendError({ category: "transient", message: "blip" });
		},
		() => textResponse("recovered"),
	]);
	const loop = new AgentLoop({
		config: makeConfig({
			maxRetries: 2,
			retryBaseDelayMs: 1,
			onEvent: (e) =>
				events.push({
					type: e.type,
					success: e.type === "auto_retry_end" ? e.success : undefined,
				}),
		}),
		backend,
	});
	const messages = await loop.run("q");

	const ends = events.filter((e) => e.type === "auto_retry_end");
	assert.deepEqual(ends, [{ type: "auto_retry_end", success: true }]);
	assert.ok(messages.some((m) => m.content === "recovered"));
});
