import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { AgentHarness } from "../core/harness.ts";
import type { LLMBackend } from "../core/backend.ts";
import type { AgentEvent } from "../core/types.ts";

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
		assert.strictEqual(h.getModel(), "gpt-4");

		assert.strictEqual(h.cycleModel("forward"), "claude-sonnet");
		assert.strictEqual(h.cycleModel("forward"), "gemma");
		assert.strictEqual(h.cycleModel("forward"), "llama");
		assert.strictEqual(h.cycleModel("forward"), "claude-sonnet");
	});

	it("cycles backward through models", () => {
		const h = makeHarness();
		assert.strictEqual(h.getModel(), "gpt-4");
		assert.strictEqual(h.cycleModel("backward"), "llama");
		assert.strictEqual(h.cycleModel("backward"), "gemma");
	});

	it("returns same model when only one available", () => {
		const h = makeHarness({ models: undefined });
		assert.strictEqual(h.cycleModel("forward"), "gpt-4");
	});

	it("returns same model when only one model in list", () => {
		const h = makeHarness({ models: [{ name: "gpt-4", model: "gpt-4" }] });
		assert.strictEqual(h.cycleModel("forward"), "gpt-4");
	});

	it("includes current model in getModels", () => {
		const h = makeHarness();
		const models = h.getModels();
		assert.strictEqual(models[0], "gpt-4");
		assert.strictEqual(models.length, 4);
		assert.strictEqual(models[1], "Claude Sonnet");
		assert.strictEqual(models[2], "Gemma");
		assert.strictEqual(models[3], "LLaMa");
	});

	it("cycleModel with no models returns current model", () => {
		const h = makeHarness({ models: [] });
		assert.strictEqual(h.cycleModel("forward"), "gpt-4");
	});

	it("switches baseUrl when target model has url override", () => {
		const h = makeHarness({
			models: [
				{ name: "Local", model: "llama-local", url: "http://localhost:8080" },
				{ name: "Remote", model: "qwen", url: "http://192.168.1.225:8080" },
			],
		});
		assert.strictEqual(h.getBaseUrl(), "http://localhost:11434/v1");

		h.cycleModel("forward");
		assert.strictEqual(h.getModel(), "llama-local");
		assert.strictEqual(h.getBaseUrl(), "http://localhost:8080");

		h.cycleModel("forward");
		assert.strictEqual(h.getModel(), "qwen");
		assert.strictEqual(h.getBaseUrl(), "http://192.168.1.225:8080");
	});
});

// ── Thinking level ───────────────────────────────────────────────────────────

describe("Thinking level", () => {
	it("defaults to off", () => {
		const h = makeHarness();
		assert.strictEqual(h.getThinkingLevel(), "off");
	});

	it("setThinkingLevel accepts all valid levels", () => {
		const h = makeHarness();
		const levels: Array<
			"off" | "minimal" | "low" | "medium" | "high" | "xhigh"
		> = ["off", "minimal", "low", "medium", "high", "xhigh"];
		for (const level of levels) {
			h.setThinkingLevel(level);
			assert.strictEqual(h.getThinkingLevel(), level);
		}
	});
});

// ── Model cycle events ──────────────────────────────────────────────────────

describe("Model cycle events", () => {
	it("emits model_cycle event with thinking level", () => {
		const events: AgentEvent[] = [];
		const h = makeHarness();
		const unsub = h.subscribe((e) => events.push(e));
		h.setThinkingLevel("high");

		h.cycleModel("forward");
		unsub();
		const cycleEvent = events.find(
			(e) => e.type === "model_cycle" && e.fromModel !== e.model,
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
		const unsub = h.subscribe((e) => events.push(e));

		h.setThinkingLevel("xhigh");
		unsub();
		const cycleEvent = events.find((e) => e.type === "model_cycle");
		assert.ok(cycleEvent !== undefined);
		if (cycleEvent && cycleEvent.type === "model_cycle") {
			assert.strictEqual(cycleEvent.thinkingLevel, "xhigh");
		}
	});
});
