import { test } from "bun:test";
import assert from "node:assert/strict";
import type {
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "../agent/backend.ts";
import { BackendError } from "../agent/backend.ts";
import { evaluateTrajectory } from "../agent/trajectory.ts";

type InjectedFault =
	| "rate_limit"
	| "timeout"
	| "context_full"
	| "malformed_response";

class FaultInjectingBackend implements LLMBackend {
	private cursor = 0;
	readonly model: string;
	constructor(
		private readonly backend: LLMBackend,
		private readonly faults: InjectedFault[],
	) {
		this.model = backend.model;
	}
	async generate(
		messages: Record<string, unknown>[],
		options?: GenerateOptions,
	): Promise<LLMResponse> {
		const fault = this.faults[this.cursor++];
		if (fault === "rate_limit")
			throw new BackendError({
				category: "rate_limit",
				message: "injected rate limit",
				status: 429,
			});
		if (fault === "timeout")
			throw new BackendError({
				category: "transient",
				message: "injected timeout",
			});
		if (fault === "context_full")
			throw new BackendError({
				category: "context_full",
				message: "injected context overflow",
			});
		if (fault === "malformed_response")
			return {
				content: null,
				toolCalls: [],
				stopReason: "error",
				errorMessage: "injected malformed response",
			};
		return this.backend.generate(messages, options);
	}
	withModel(model: string): LLMBackend {
		return new FaultInjectingBackend(
			this.backend.withModel(model),
			this.faults.slice(this.cursor),
		);
	}
	withEndpoint(model: string, baseUrl: string): LLMBackend {
		const backend =
			this.backend.withEndpoint?.(model, baseUrl) ??
			this.backend.withModel(model);
		return new FaultInjectingBackend(backend, this.faults.slice(this.cursor));
	}
}

void test("trajectory evaluation flags unsupported completed outcomes", () => {
	const base = {
		version: 1 as const,
		sessionId: "s",
		runId: "r",
		operationId: "o",
	};
	const report = evaluateTrajectory([
		{
			...base,
			sequence: 1,
			timestamp: 10,
			kind: "agent_event",
			payload: { type: "run_outcome", status: "completed" },
		},
		{
			...base,
			sequence: 2,
			timestamp: 20,
			kind: "run_finish",
			payload: { status: "completed" },
		},
	]);
	assert.equal(report.acceptancePassed, true);
	assert.equal(report.durationMs, 10);
});

void test("fault injecting backend deterministically exercises recovery categories", async () => {
	const backend: LLMBackend = {
		model: "base",
		generate: async () => ({
			content: "ok",
			toolCalls: [],
			stopReason: "stop",
		}),
		withModel: () => backend,
	};
	const injected = new FaultInjectingBackend(backend, [
		"rate_limit",
		"context_full",
	]);
	await assert.rejects(
		injected.generate([]),
		error => error instanceof BackendError && error.category === "rate_limit",
	);
	await assert.rejects(
		injected.generate([]),
		error => error instanceof BackendError && error.category === "context_full",
	);
	assert.equal((await injected.generate([])).content, "ok");
});
