// Golden event-stream pin for the agent loop (P3.7 safety net).
//
// The loop refactor (turn-phase decomposition) must not change observable
// behavior: each scenario below scripts a deterministic provider session and
// pins the FULL normalized event stream plus the returned transcript.
// Wall-clock fields (event `ts`, message `timestamp`, budget `elapsedMs`) are
// stripped before comparison.
//
// To re-pin after an intentional behavior change:
//   LOGICIAN_GOLDEN_UPDATE=1 bun test packages/log-core/src/__tests__/agent/agent-loop-golden.test.ts
// then review and paste the printed streams into GOLDEN.

import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	type RunAgentLoopConfig,
	runAgentLoop,
} from "../../runtime/harness/agent-harness.ts";
import type { AgentConfig } from "../../system/types/types-config.ts";
import type {
	AgentEvent,
	Message,
	Tool,
} from "../../system/types/types-messages.ts";
import { FakeBackend, textResponse } from "../fake-backend.ts";

function user(content: string): Message {
	return { role: "user", content };
}

const noop: Tool = {
	name: "noop",
	description: "does nothing",
	parameters: { type: "object", properties: {} },
	execute: async () => "ok",
};

function makeConfig(overrides: Partial<AgentConfig> = {}): RunAgentLoopConfig {
	return {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		executionProfile: "autonomous",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		tools: [noop],
		...overrides,
	} as unknown as RunAgentLoopConfig;
}

/** Strip wall-clock fields so streams compare equal across runs. */
function normalize(events: readonly AgentEvent[]): unknown[] {
	const clean = (value: unknown): unknown => {
		if (Array.isArray(value)) return value.map(clean);
		if (value !== null && typeof value === "object") {
			const out: Record<string, unknown> = {};
			for (const [key, val] of Object.entries(
				value as Record<string, unknown>,
			)) {
				if (key === "ts" || key === "timestamp" || key === "elapsedMs")
					continue;
				if (val === undefined) continue; // JSON cannot represent it
				out[key] = clean(val);
			}
			return out;
		}
		return value;
	};
	return events.map(clean);
}

interface ScenarioResult {
	events: unknown[];
	returned: string[];
}

async function runScenario(
	scenario: () => Promise<{ events: AgentEvent[]; returned: Message[] }>,
): Promise<ScenarioResult> {
	const { events, returned } = await scenario();
	return {
		events: normalize(events),
		returned: returned.map(m => `${m.role}:${m.content ?? ""}`),
	};
}

function assertGolden(name: string, actual: ScenarioResult): void {
	if (process.env.LOGICIAN_GOLDEN_UPDATE === "1") {
		console.log(`// ── ${name} ──`);
		console.log(JSON.stringify(actual, null, 1));
		return;
	}
	const golden = GOLDEN[name];
	assert.ok(
		golden !== undefined,
		`missing golden for "${name}" — run with LOGICIAN_GOLDEN_UPDATE=1`,
	);
	assert.deepEqual(actual, golden);
}

// ── Scenarios ────────────────────────────────────────────────────────────────

/**
 * Happy path: steering drained on the first turn, a tool round-trip, then a
 * text turn that ends the run. Exercises hook ordering (beforeAgentStart
 * seed, request-scoped transformContext), the inner tool loop, context
 * updates, and the normal termination path.
 */
async function scenarioToolTurnWithSteering() {
	let steerDrained = false;
	const events: AgentEvent[] = [];
	const returned = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{
			...makeConfig({ maxIterations: 6 }),
			backend: new FakeBackend([
				() => ({
					content: "calling a tool",
					toolCalls: [{ id: "t1", name: "noop", arguments: "{}" }],
					stopReason: "stop" as const,
					usage: {
						totalTokens: 120,
						promptTokens: 100,
						completionTokens: 20,
					},
				}),
				() => textResponse("all done"),
			]),
			hooks: {
				beforeAgentStart: () => ({ messages: [user("seed from hook")] }),
				getSteeringMessages: () => {
					if (steerDrained) return [];
					steerDrained = true;
					return [user("steer mid-run")];
				},
				transformContext: ({ messages }) => ({
					messages: [...messages, { role: "system", content: "[transformed]" }],
				}),
			},
		},
		event => {
			events.push(event);
		},
	);
	return { events, returned };
}

/**
 * Stagnation recovery: the first exact-repetition hit injects one nudge and
 * re-enters the loop WITHOUT a turn_end; the second hit blocks the run.
 */
async function scenarioStagnationNudgeThenBlock() {
	const unit = "checking progress on the current task step ";
	const stagnant = unit.repeat(5);
	const events: AgentEvent[] = [];
	const returned = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("do the thing")],
		{
			...makeConfig({ maxIterations: 5 }),
			backend: new FakeBackend([
				() => textResponse(stagnant),
				() => textResponse(stagnant),
			]),
		},
		event => {
			events.push(event);
		},
	);
	return { events, returned };
}

/**
 * Iteration ceiling: a tool call keeps the inner loop alive until
 * maxIterations is reached, then the run ends failed with a max_iterations
 * event.
 */
async function scenarioMaxIterations() {
	const events: AgentEvent[] = [];
	const returned = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("loop")],
		{
			...makeConfig({ maxIterations: 1 }),
			backend: new FakeBackend([
				() => ({
					content: "",
					toolCalls: [{ id: "t1", name: "noop", arguments: "{}" }],
					stopReason: "stop" as const,
				}),
			]),
		},
		event => {
			events.push(event);
		},
	);
	return { events, returned };
}

/**
 * Run budget: one provider call allowed, so the second turn's budget check
 * intervenes and blocks the run.
 */
async function scenarioProviderBudgetExhaustion() {
	const events: AgentEvent[] = [];
	const returned = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("spend it")],
		{
			...makeConfig({
				maxIterations: 5,
				runBudget: { maxProviderCalls: 1 },
			}),
			backend: new FakeBackend([
				() => ({
					content: "first call",
					toolCalls: [{ id: "t1", name: "noop", arguments: "{}" }],
					stopReason: "stop" as const,
				}),
			]),
		},
		event => {
			events.push(event);
		},
	);
	return { events, returned };
}

// ── Goldens ──────────────────────────────────────────────────────────────────

const GOLDEN: Record<string, ScenarioResult> = {
	"tool-turn-with-steering": {
		events: [
			{
				type: "agent_start",
				seq: 1,
			},
			{
				type: "message_start",
				turnId: "turn_0",
				role: "user",
				seq: 2,
			},
			{
				type: "message_end",
				turnId: "turn_0",
				message: {
					role: "user",
					content: "prompt",
				},
				seq: 3,
			},
			{
				type: "turn_start",
				turnId: "turn_1",
				seq: 4,
			},
			{
				type: "message_start",
				turnId: "turn_1",
				role: "user",
				seq: 5,
			},
			{
				type: "message_end",
				turnId: "turn_1",
				message: {
					role: "user",
					content: "steer mid-run",
				},
				seq: 6,
			},
			{
				type: "message_start",
				turnId: "turn_1",
				role: "assistant",
				seq: 7,
			},
			{
				type: "message_update",
				turnId: "turn_1",
				message: {
					role: "assistant",
					content: "calling a tool",
					tool_calls: [
						{
							id: "t1",
							name: "noop",
							arguments: "{}",
						},
					],
				},
				seq: 8,
			},
			{
				type: "message_end",
				turnId: "turn_1",
				message: {
					role: "assistant",
					content: "calling a tool",
					tool_calls: [
						{
							id: "t1",
							name: "noop",
							arguments: "{}",
						},
					],
				},
				seq: 9,
			},
			{
				type: "tool_execution_start",
				toolCallId: "t1",
				toolName: "noop",
				args: {},
				seq: 10,
			},
			{
				type: "tool_call_end",
				toolName: "noop",
				toolCallId: "t1",
				result: "ok",
				isError: false,
				seq: 11,
			},
			{
				type: "tool_execution_end",
				toolCallId: "t1",
				toolName: "noop",
				result: "ok",
				isError: false,
				seq: 12,
			},
			{
				type: "message_start",
				turnId: "turn_1",
				role: "tool",
				seq: 13,
			},
			{
				type: "message_end",
				turnId: "turn_1",
				message: {
					role: "tool",
					content: "ok",
					tool_call_id: "t1",
					name: "noop",
				},
				seq: 14,
			},
			{
				type: "context_update",
				tokens: 120,
				cachedTokens: null,
				promptTokens: 100,
				completionTokens: 20,
				prefixStable: true,
				prefixDivergedAt: 0,
				prefixRewritten: false,
				seq: 15,
			},
			{
				type: "turn_end",
				turnId: "turn_1",
				stopReason: "tool_calls",
				message: {
					role: "assistant",
					content: "calling a tool",
					tool_calls: [
						{
							id: "t1",
							name: "noop",
							arguments: "{}",
						},
					],
				},
				toolResults: [
					{
						role: "tool",
						content: "ok",
						tool_call_id: "t1",
						name: "noop",
					},
				],
				seq: 16,
			},
			{
				type: "turn_start",
				turnId: "turn_2",
				seq: 17,
			},
			{
				type: "message_start",
				turnId: "turn_2",
				role: "assistant",
				seq: 18,
			},
			{
				type: "message_update",
				turnId: "turn_2",
				message: {
					role: "assistant",
					content: "all done",
					tool_calls: [],
				},
				seq: 19,
			},
			{
				type: "message_end",
				turnId: "turn_2",
				message: {
					role: "assistant",
					content: "all done",
					tool_calls: [],
				},
				seq: 20,
			},
			{
				type: "context_update",
				tokens: 135,
				cachedTokens: null,
				promptTokens: null,
				completionTokens: null,
				prefixStable: false,
				prefixDivergedAt: 4,
				prefixRewritten: false,
				seq: 21,
			},
			{
				type: "turn_end",
				turnId: "turn_2",
				stopReason: "stop",
				message: {
					role: "assistant",
					content: "all done",
					tool_calls: [],
				},
				toolResults: [],
				seq: 22,
			},
			{
				type: "agent_end",
				messages: [
					{
						role: "user",
						content: "prompt",
					},
					{
						role: "user",
						content: "seed from hook",
					},
					{
						role: "user",
						content: "steer mid-run",
					},
					{
						role: "assistant",
						content: "calling a tool",
						tool_calls: [
							{
								id: "t1",
								name: "noop",
								arguments: "{}",
							},
						],
					},
					{
						role: "tool",
						content: "ok",
						tool_call_id: "t1",
						name: "noop",
					},
					{
						role: "assistant",
						content: "all done",
						tool_calls: [],
					},
				],
				status: "completed",
				summary: "all done",
				stepCount: 2,
				seq: 23,
			},
		],
		returned: [
			"user:prompt",
			"user:seed from hook",
			"user:steer mid-run",
			"assistant:calling a tool",
			"tool:ok",
			"assistant:all done",
		],
	},
	"stagnation-nudge-then-block": {
		events: [
			{
				type: "agent_start",
				seq: 1,
			},
			{
				type: "message_start",
				turnId: "turn_0",
				role: "user",
				seq: 2,
			},
			{
				type: "message_end",
				turnId: "turn_0",
				message: {
					role: "user",
					content: "do the thing",
				},
				seq: 3,
			},
			{
				type: "turn_start",
				turnId: "turn_1",
				seq: 4,
			},
			{
				type: "message_start",
				turnId: "turn_1",
				role: "assistant",
				seq: 5,
			},
			{
				type: "message_update",
				turnId: "turn_1",
				message: {
					role: "assistant",
					content:
						"checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step ",
					tool_calls: [],
				},
				seq: 6,
			},
			{
				type: "message_end",
				turnId: "turn_1",
				message: {
					role: "assistant",
					content:
						"checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step ",
					tool_calls: [],
				},
				seq: 7,
			},
			{
				type: "harness_intervention",
				id: "intervention-1",
				kind: "loop",
				cause: "text_stagnation",
				action: "change_strategy",
				severity: "warning",
				detector: "text_loop_detector",
				attempt: 1,
				evidence: {
					summary:
						'Text stagnation detected: Exact repetition detected: "checking progress on the current task st..." repeated 5×; strategy-change nudge injected.',
				},
				iteration: 1,
				seq: 8,
			},
			{
				type: "turn_start",
				turnId: "turn_2",
				seq: 9,
			},
			{
				type: "message_start",
				turnId: "turn_2",
				role: "user",
				seq: 10,
			},
			{
				type: "message_end",
				turnId: "turn_2",
				message: {
					role: "user",
					content:
						'[stagnation-nudge] The harness detected that your last response repeats prior content without new progress (Exact repetition detected: "checking progress on the current task st..." repeated 5×). Do not restate the same analysis. Change strategy and take a concrete next step: run a tool, make an edit, or state a decision and proceed. If the task is already complete, say so in one explicit sentence.',
				},
				seq: 11,
			},
			{
				type: "message_start",
				turnId: "turn_2",
				role: "assistant",
				seq: 12,
			},
			{
				type: "message_update",
				turnId: "turn_2",
				message: {
					role: "assistant",
					content:
						"checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step ",
					tool_calls: [],
				},
				seq: 13,
			},
			{
				type: "message_end",
				turnId: "turn_2",
				message: {
					role: "assistant",
					content:
						"checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step ",
					tool_calls: [],
				},
				seq: 14,
			},
			{
				type: "harness_intervention",
				id: "intervention-1",
				kind: "loop",
				cause: "text_stagnation",
				action: "change_strategy",
				severity: "warning",
				detector: "text_loop_detector",
				attempt: 2,
				evidence: {
					summary:
						'Text stagnation detected: Exact repetition detected: "checking progress on the current task st..." repeated 5×',
				},
				iteration: 2,
				seq: 15,
			},
			{
				type: "agent_end",
				messages: [
					{
						role: "user",
						content: "do the thing",
					},
					{
						role: "assistant",
						content:
							"checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step ",
						tool_calls: [],
					},
					{
						role: "user",
						content:
							'[stagnation-nudge] The harness detected that your last response repeats prior content without new progress (Exact repetition detected: "checking progress on the current task st..." repeated 5×). Do not restate the same analysis. Change strategy and take a concrete next step: run a tool, make an edit, or state a decision and proceed. If the task is already complete, say so in one explicit sentence.',
					},
					{
						role: "assistant",
						content:
							"checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step ",
						tool_calls: [],
					},
				],
				status: "blocked",
				summary:
					"Agent entered a text stagnation loop — repeating content without progress.",
				stepCount: 2,
				seq: 16,
			},
		],
		returned: [
			"user:do the thing",
			"assistant:checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step ",
			'user:[stagnation-nudge] The harness detected that your last response repeats prior content without new progress (Exact repetition detected: "checking progress on the current task st..." repeated 5×). Do not restate the same analysis. Change strategy and take a concrete next step: run a tool, make an edit, or state a decision and proceed. If the task is already complete, say so in one explicit sentence.',
			"assistant:checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step checking progress on the current task step ",
		],
	},
	"max-iterations": {
		events: [
			{
				type: "agent_start",
				seq: 1,
			},
			{
				type: "message_start",
				turnId: "turn_0",
				role: "user",
				seq: 2,
			},
			{
				type: "message_end",
				turnId: "turn_0",
				message: {
					role: "user",
					content: "loop",
				},
				seq: 3,
			},
			{
				type: "turn_start",
				turnId: "turn_1",
				seq: 4,
			},
			{
				type: "message_start",
				turnId: "turn_1",
				role: "assistant",
				seq: 5,
			},
			{
				type: "message_update",
				turnId: "turn_1",
				message: {
					role: "assistant",
					content: null,
					tool_calls: [
						{
							id: "t1",
							name: "noop",
							arguments: "{}",
						},
					],
				},
				seq: 6,
			},
			{
				type: "message_end",
				turnId: "turn_1",
				message: {
					role: "assistant",
					content: null,
					tool_calls: [
						{
							id: "t1",
							name: "noop",
							arguments: "{}",
						},
					],
				},
				seq: 7,
			},
			{
				type: "tool_execution_start",
				toolCallId: "t1",
				toolName: "noop",
				args: {},
				seq: 8,
			},
			{
				type: "tool_call_end",
				toolName: "noop",
				toolCallId: "t1",
				result: "ok",
				isError: false,
				seq: 9,
			},
			{
				type: "tool_execution_end",
				toolCallId: "t1",
				toolName: "noop",
				result: "ok",
				isError: false,
				seq: 10,
			},
			{
				type: "message_start",
				turnId: "turn_1",
				role: "tool",
				seq: 11,
			},
			{
				type: "message_end",
				turnId: "turn_1",
				message: {
					role: "tool",
					content: "ok",
					tool_call_id: "t1",
					name: "noop",
				},
				seq: 12,
			},
			{
				type: "context_update",
				tokens: 99,
				cachedTokens: null,
				promptTokens: null,
				completionTokens: null,
				prefixStable: true,
				prefixDivergedAt: 0,
				prefixRewritten: false,
				seq: 13,
			},
			{
				type: "turn_end",
				turnId: "turn_1",
				stopReason: "tool_calls",
				message: {
					role: "assistant",
					content: null,
					tool_calls: [
						{
							id: "t1",
							name: "noop",
							arguments: "{}",
						},
					],
				},
				toolResults: [
					{
						role: "tool",
						content: "ok",
						tool_call_id: "t1",
						name: "noop",
					},
				],
				seq: 14,
			},
			{
				type: "max_iterations",
				iterations: 1,
				limit: 1,
				seq: 15,
			},
			{
				type: "agent_end",
				messages: [
					{
						role: "user",
						content: "loop",
					},
					{
						role: "assistant",
						content: null,
						tool_calls: [
							{
								id: "t1",
								name: "noop",
								arguments: "{}",
							},
						],
					},
					{
						role: "tool",
						content: "ok",
						tool_call_id: "t1",
						name: "noop",
					},
				],
				status: "failed",
				stepCount: 1,
				seq: 16,
			},
		],
		returned: ["user:loop", "assistant:", "tool:ok"],
	},
	"provider-budget-exhaustion": {
		events: [
			{
				type: "agent_start",
				seq: 1,
			},
			{
				type: "message_start",
				turnId: "turn_0",
				role: "user",
				seq: 2,
			},
			{
				type: "message_end",
				turnId: "turn_0",
				message: {
					role: "user",
					content: "spend it",
				},
				seq: 3,
			},
			{
				type: "turn_start",
				turnId: "turn_1",
				seq: 4,
			},
			{
				type: "message_start",
				turnId: "turn_1",
				role: "assistant",
				seq: 5,
			},
			{
				type: "message_update",
				turnId: "turn_1",
				message: {
					role: "assistant",
					content: "first call",
					tool_calls: [
						{
							id: "t1",
							name: "noop",
							arguments: "{}",
						},
					],
				},
				seq: 6,
			},
			{
				type: "message_end",
				turnId: "turn_1",
				message: {
					role: "assistant",
					content: "first call",
					tool_calls: [
						{
							id: "t1",
							name: "noop",
							arguments: "{}",
						},
					],
				},
				seq: 7,
			},
			{
				type: "tool_execution_start",
				toolCallId: "t1",
				toolName: "noop",
				args: {},
				seq: 8,
			},
			{
				type: "tool_call_end",
				toolName: "noop",
				toolCallId: "t1",
				result: "ok",
				isError: false,
				seq: 9,
			},
			{
				type: "tool_execution_end",
				toolCallId: "t1",
				toolName: "noop",
				result: "ok",
				isError: false,
				seq: 10,
			},
			{
				type: "message_start",
				turnId: "turn_1",
				role: "tool",
				seq: 11,
			},
			{
				type: "message_end",
				turnId: "turn_1",
				message: {
					role: "tool",
					content: "ok",
					tool_call_id: "t1",
					name: "noop",
				},
				seq: 12,
			},
			{
				type: "context_update",
				tokens: 106,
				cachedTokens: null,
				promptTokens: null,
				completionTokens: null,
				prefixStable: true,
				prefixDivergedAt: 0,
				prefixRewritten: false,
				seq: 13,
			},
			{
				type: "turn_end",
				turnId: "turn_1",
				stopReason: "tool_calls",
				message: {
					role: "assistant",
					content: "first call",
					tool_calls: [
						{
							id: "t1",
							name: "noop",
							arguments: "{}",
						},
					],
				},
				toolResults: [
					{
						role: "tool",
						content: "ok",
						tool_call_id: "t1",
						name: "noop",
					},
				],
				seq: 14,
			},
			{
				type: "harness_intervention",
				id: "intervention-1",
				kind: "budget",
				cause: "run_budget",
				action: "recover",
				severity: "warning",
				detector: "run_budget",
				attempt: 1,
				evidence: {
					summary: "provider-call budget exhausted",
					counters: {
						providerCalls: 1,
						toolCalls: 1,
					},
				},
				iteration: 1,
				seq: 15,
			},
			{
				type: "agent_end",
				messages: [
					{
						role: "user",
						content: "spend it",
					},
					{
						role: "assistant",
						content: "first call",
						tool_calls: [
							{
								id: "t1",
								name: "noop",
								arguments: "{}",
							},
						],
					},
					{
						role: "tool",
						content: "ok",
						tool_call_id: "t1",
						name: "noop",
					},
				],
				status: "blocked",
				summary: "provider-call budget exhausted",
				stepCount: 1,
				seq: 16,
			},
		],
		returned: ["user:spend it", "assistant:first call", "tool:ok"],
	},
};

// ── Tests ────────────────────────────────────────────────────────────────────

void test("golden: tool turn with steering", async () => {
	assertGolden(
		"tool-turn-with-steering",
		await runScenario(scenarioToolTurnWithSteering),
	);
});

void test("golden: stagnation nudge then block", async () => {
	assertGolden(
		"stagnation-nudge-then-block",
		await runScenario(scenarioStagnationNudgeThenBlock),
	);
});

void test("golden: max iterations", async () => {
	assertGolden("max-iterations", await runScenario(scenarioMaxIterations));
});

void test("golden: provider budget exhaustion", async () => {
	assertGolden(
		"provider-budget-exhaustion",
		await runScenario(scenarioProviderBudgetExhaustion),
	);
});
