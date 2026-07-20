import assert from "node:assert/strict";
import { test } from "node:test";
import type {
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "@logician/agent-core/core/backend.ts";
import type { AgentConfig, Tool } from "@logician/agent-core";
import { runDelegatedAgent } from "../subagents/delegation-runtime.ts";

class FakeBackend implements LLMBackend {
	readonly model = "fake";
	private readonly responses: LLMResponse[];
	constructor(responses: LLMResponse[]) {
		this.responses = [...responses];
	}
	withModel(): LLMBackend {
		return this;
	}
	async generate(
		_messages: Record<string, unknown>[],
		_options?: GenerateOptions,
	): Promise<LLMResponse> {
		return this.responses.shift() ?? {
			content: "done",
			toolCalls: [],
			stopReason: "stop",
		};
	}
}

const baseConfig: AgentConfig = {
	baseUrl: "http://test",
	model: "fake",
	systemPrompt: "You are the specialist agent.",
	tools: [],
};

function report(status: "satisfied" | "failed", answer: string): LLMResponse {
	return {
		content: `${answer}\n\n\`\`\`acceptance-report\n{"criteriaSatisfied":[{"id":"criterion-1","status":"${status}","evidence":"checked"}]}\n\`\`\``,
		toolCalls: [],
		stopReason: "stop",
	};
}

void test("delegated contracts retry failed output and preserve a clean final result", async () => {
	const result = await runDelegatedAgent({
		task: "Produce the result",
		config: baseConfig,
		backend: new FakeBackend([
			report("failed", "incomplete"),
			report("satisfied", "corrected result"),
		]),
		tools: [],
		maxIterations: 4,
		contract: { expectedOutput: "a corrected result", maxValidationRetries: 1 },
		onEvent: () => {},
	});

	assert.equal(result.status, "completed");
	assert.equal(result.content, "corrected result");
	assert.equal(result.validationAttempts, 2);
	assert.equal(result.turns, 2);
});

void test("delegated tool-call budgets are shared across the whole run", async () => {
	let executions = 0;
	const tool: Tool = {
		name: "probe",
		description: "probe",
		parameters: { type: "object", properties: {} },
		execute: async () => {
			executions++;
			return "ok";
		},
	};
	const toolCall = (id: string): LLMResponse => ({
		content: "",
		toolCalls: [{ id, name: "probe", arguments: "{}" }],
		stopReason: "stop",
	});
	const result = await runDelegatedAgent({
		task: "Probe twice",
		config: { ...baseConfig, tools: [tool] },
		backend: new FakeBackend([toolCall("one"), toolCall("two"), {
			content: "finished",
			toolCalls: [],
			stopReason: "stop",
		}]),
		tools: [tool],
		maxIterations: 3,
		budget: { maxToolCalls: 1 },
		onEvent: () => {},
	});

	assert.equal(executions, 1);
	assert.equal(result.toolCalls, 1);
	assert.equal(result.toolCallsByName.probe, 1);
	assert.equal(result.status, "failed");
});

void test("whole-task deadlines cancel a delegated run", async () => {
	const backend: LLMBackend = {
		model: "slow",
		withModel() {
			return this;
		},
		generate: async (_messages, options = {}) =>
			new Promise<LLMResponse>((_resolve, reject) => {
				const rejectAbort = () => reject(new DOMException("timed out", "AbortError"));
				if (options.signal?.aborted) rejectAbort();
				else options.signal?.addEventListener("abort", rejectAbort, { once: true });
			}),
	};
	const result = await runDelegatedAgent({
		task: "Never finishes",
		config: baseConfig,
		backend,
		tools: [],
		maxIterations: 3,
		budget: { timeoutMs: 10 },
		onEvent: () => {},
	});

	assert.equal(result.status, "cancelled");
	assert.ok(result.durationMs < 1_000);
});
