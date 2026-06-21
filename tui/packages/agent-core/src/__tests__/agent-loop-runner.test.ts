import assert from "node:assert/strict";
import { test } from "node:test";
import { runAgentLoop } from "../core/agent-loop-runner.ts";
import type { AgentConfig, AgentEvent, Message, Tool } from "../core/types.ts";
import { FakeBackend, textResponse } from "./fake-backend.ts";

const noop: Tool = {
	name: "noop",
	description: "does nothing",
	parameters: { type: "object", properties: {} },
	execute: async () => "ok",
};

function makeConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
	return {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		tools: [noop],
		...overrides,
	};
}

function user(content: string): Message {
	return { role: "user", content };
}

void test("runAgentLoop injects steering before the next assistant call", async () => {
	const backend = new FakeBackend([
		(messages) => {
			assert.ok(messages.some((m) => m.role === "user" && m.content === "steer"));
			return textResponse("done");
		},
	]);
	let drained = false;
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{
			...makeConfig(),
			backend,
			getSteeringMessages: () => {
				if (drained) return [];
				drained = true;
				return [user("steer")];
			},
		},
		() => {},
	);
	assert.deepEqual(newMessages.map((m) => `${m.role}:${m.content ?? ""}`), [
		"user:prompt",
		"user:steer",
		"assistant:done",
	]);
});

void test("runAgentLoop processes follow-up messages after a stop", async () => {
	const backend = new FakeBackend([
		() => textResponse("first"),
		(messages) => {
			assert.ok(messages.some((m) => m.role === "user" && m.content === "follow"));
			return textResponse("second");
		},
	]);
	let followUps = 0;
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{
			...makeConfig(),
			backend,
			getFollowUpMessages: () => {
				if (followUps++ === 0) return [user("follow")];
				return [];
			},
		},
		() => {},
	);
	assert.deepEqual(newMessages.map((m) => `${m.role}:${m.content ?? ""}`), [
		"user:prompt",
		"assistant:first",
		"user:follow",
		"assistant:second",
	]);
});

void test("runAgentLoop executes a tool batch and returns ordered tool results", async () => {
	const tool: Tool = {
		name: "echo",
		description: "echoes",
		parameters: { type: "object", properties: { text: { type: "string" } } },
		execute: async (args) => `echo:${String(args.text)}`,
	};
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [
				{ id: "a", name: "echo", arguments: JSON.stringify({ text: "one" }) },
				{ id: "b", name: "echo", arguments: JSON.stringify({ text: "two" }) },
			],
			stopReason: "stop",
		}),
		() => textResponse("final"),
	]);
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [tool] },
		[user("prompt")],
		{ ...makeConfig({ tools: [tool] }), backend },
		() => {},
	);
	assert.deepEqual(
		newMessages.filter((m) => m.role === "tool").map((m) => `${m.tool_call_id}:${m.content ?? ""}`),
		["a:echo:one", "b:echo:two"],
	);
});

void test("runAgentLoop reports provider errors as terminal new messages", async () => {
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		() => ({ content: null, toolCalls: [], stopReason: "error", errorMessage: "boom" }),
	]);
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{ ...makeConfig(), backend },
		(event) => {
			events.push(event);
		},
	);
	assert.equal(newMessages.at(-1)?.role, "assistant");
	assert.ok(events.some((event) => event.type === "error" && event.message === "boom"));
	assert.ok(events.some((event) => event.type === "agent_end"));
});

void test("runAgentLoop stops before provider call when aborted", async () => {
	const controller = new AbortController();
	controller.abort();
	const backend = new FakeBackend([() => textResponse("should not run")]);
	const events: AgentEvent[] = [];
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{ ...makeConfig(), backend, signal: controller.signal },
		(event) => {
			events.push(event);
		},
	);
	assert.equal(backend.calls, 0);
	assert.deepEqual(newMessages.map((m) => `${m.role}:${m.content ?? ""}`), ["user:prompt"]);
	assert.ok(events.some((event) => event.type === "error" && event.message.includes("aborted")));
});
