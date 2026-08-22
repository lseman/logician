import { describe, it } from "bun:test";
import assert from "node:assert/strict";
import { AgentHarness } from "../../runtime/harness/agent-harness.ts";
import { clampThinkingLevel } from "../../runtime/harness/live/model.ts";
import type { LLMBackend } from "../../capabilities/provider/backend.ts";
import type { AgentEvent } from "../../system/types/types-messages.ts";

class FakeBackend implements LLMBackend {
	readonly model = "fake-model";
	generate = async () => ({
		content: "",
		toolCalls: [],
		stopReason: "stop" as const,
	});
	withModel(_model: string): LLMBackend {
		return new FakeBackend();
	}
}

const fakeBackend = new FakeBackend();

it("falls back to thinking off for an invalid level", () => {
	assert.equal(clampThinkingLevel("invalid"), "off");
});

function makeHarness(overrides?: Record<string, unknown>) {
	const config = {
		baseUrl: "http://localhost:11434/v1",
		model: "gpt-4",
		models: [
			{ name: "Claude Sonnet", model: "claude-sonnet" },
			{ name: "Gemma", model: "gemma" },
			{ name: "LLaMa", model: "llama" },
		],
		tools: [],
		...overrides,
	} as Record<string, unknown>;
	return new AgentHarness({
		config: config as any,
		backend: fakeBackend,
		maxIterations: 5,
	});
}

// ── Model cycling ────────────────────────────────────────────────────────────

describe("Model cycling", () => {
	it("cycles forward through models", () => {
		const h = makeHarness();
		assert.strictEqual(h.models.model, "gpt-4");

		assert.strictEqual(h.models.cycle("forward"), "claude-sonnet");
		assert.strictEqual(h.models.cycle("forward"), "gemma");
		assert.strictEqual(h.models.cycle("forward"), "llama");
		assert.strictEqual(h.models.cycle("forward"), "claude-sonnet");
	});

	it("cycles backward through models", () => {
		const h = makeHarness();
		assert.strictEqual(h.models.model, "gpt-4");
		assert.strictEqual(h.models.cycle("backward"), "llama");
		assert.strictEqual(h.models.cycle("backward"), "gemma");
	});

	it("returns same model when only one available", () => {
		const h = makeHarness({ models: undefined });
		assert.strictEqual(h.models.cycle("forward"), "gpt-4");
	});

	it("returns same model when only one model in list", () => {
		const h = makeHarness({ models: [{ name: "gpt-4", model: "gpt-4" }] });
		assert.strictEqual(h.models.cycle("forward"), "gpt-4");
	});

	it("includes current model in getModels", () => {
		const h = makeHarness();
		const models = h.models.models();
		assert.strictEqual(models[0], "gpt-4");
		assert.strictEqual(models.length, 4);
		assert.strictEqual(models[1], "claude-sonnet");
		assert.strictEqual(models[2], "gemma");
		assert.strictEqual(models[3], "llama");
	});

	it("cycleModel with no models returns current model", () => {
		const h = makeHarness({ models: [] });
		assert.strictEqual(h.models.cycle("forward"), "gpt-4");
	});

	it("switches baseUrl when target model has url override", () => {
		const h = makeHarness({
			models: [
				{ name: "Local", model: "llama-local", url: "http://localhost:8080" },
				{ name: "Remote", model: "qwen", url: "http://192.168.1.225:8080" },
			],
		});
		assert.strictEqual(h.models.baseUrl, "http://localhost:11434/v1");

		h.models.cycle("forward");
		assert.strictEqual(h.models.model, "llama-local");
		assert.strictEqual(h.models.baseUrl, "http://localhost:8080");

		h.models.cycle("forward");
		assert.strictEqual(h.models.model, "qwen");
		assert.strictEqual(h.models.baseUrl, "http://192.168.1.225:8080");
	});
});

// ── Thinking level ───────────────────────────────────────────────────────────

describe("Thinking level", () => {
	it("defaults to off", () => {
		const h = makeHarness();
		assert.strictEqual(h.models.thinkingLevel, "off");
	});

	it("setThinkingLevel accepts all valid levels", () => {
		const h = makeHarness();
		const levels: Array<
			"off" | "minimal" | "low" | "medium" | "high" | "xhigh"
		> = ["off", "minimal", "low", "medium", "high", "xhigh"];
		for (const level of levels) {
			h.models.setThinkingLevel(level);
			assert.strictEqual(h.models.thinkingLevel, level);
		}
	});
});

// ── Model cycle events ──────────────────────────────────────────────────────

describe("Model cycle events", () => {
	it("emits model_cycle event with thinking level", () => {
		const events: AgentEvent[] = [];
		const h = makeHarness();
		const unsub = h.observe({ event: e => events.push(e) });
		h.models.setThinkingLevel("high");

		h.models.cycle("forward");
		unsub();
		const cycleEvent = events.find(
			e => e.type === "model_cycle" && e.fromModel !== e.model,
		);
		assert.ok(cycleEvent !== undefined);
		if (cycleEvent && cycleEvent.type === "model_cycle") {
			assert.strictEqual(cycleEvent.model, "claude-sonnet");
			assert.strictEqual(cycleEvent.fromModel, "gpt-4");
			assert.strictEqual(cycleEvent.thinkingLevel, "high");
		}
	});

	it("setThinkingLevel emits model_cycle when level changes", () => {
		const events: AgentEvent[] = [];
		const h = makeHarness();
		const unsub = h.observe({ event: e => events.push(e) });

		h.models.setThinkingLevel("xhigh");
		unsub();
		const cycleEvent = events.find(e => e.type === "model_cycle");
		assert.ok(cycleEvent !== undefined);
		if (cycleEvent && cycleEvent.type === "model_cycle") {
			assert.strictEqual(cycleEvent.thinkingLevel, "xhigh");
		}
	});
});
