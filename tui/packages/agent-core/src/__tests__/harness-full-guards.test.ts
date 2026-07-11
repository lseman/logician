import assert from "node:assert/strict";
import { test } from "node:test";
import { AgentHarness } from "../core/harness.ts";
import type { AgentConfig, Tool } from "../core/types.ts";
import { OpenAIBackend } from "../core/backend.ts";

let config: AgentConfig;
let backend: OpenAIBackend;

test("setup", () => {
	config = {
		baseUrl: "https://api.openai.com/v1",
		model: "gpt-4o",
		contextWindowTokens: 128000,
		temperature: 1,

		// Enable all guards
		guardsEnabled: true,
		duplicateToolThreshold: 3,
		toolFailureLoopThreshold: 3,
		budgetStopEnabled: true,
		proactiveCompactionEnabled: true,
		proactiveCompactionFraction: 0.8,
		continuationEnabled: true,

		// Loop detection
		loopDetectionEnabled: true,
		loopDetectionWindow: 10,
		degenerateLoopThreshold: 4,
		stagnationThreshold: 5,

		// Error recovery
		autoRetryEnabled: true,
		maxRetries: 3,
		retryBaseDelayMs: 1000,
		turnTimeoutMs: 60000,

		tools: [],
	};

	backend = new OpenAIBackend({
		baseUrl: config.baseUrl,
		model: config.model,
	});
});

function describe(name: string, fn: () => void) {
	fn();
}
function it(name: string, fn: () => void | Promise<void>) {
	test(name, fn);
}
function expect<T>(actual: T) {
	return {
		toBe(expected: unknown) {
			assert.equal(actual, expected);
		},
		toBeDefined() {
			assert.notEqual(actual, undefined);
		},
		toThrow() {
			let threw = false;
			try {
				(actual as () => void)();
			} catch {
				threw = true;
			}
			assert.ok(threw, "expected function to throw");
		},
	};
}

void describe("harness with full guards", () => {
	void it("constructs with all config fields validated", () => {
		const harness = new AgentHarness({
			config,
			backend,
		});

		expect(harness.phase).toBe("idle");
		expect(harness.getLoopDetector()).toBeDefined();
	});

	void it("rejects invalid config at construction", () => {
		const badConfig: AgentConfig = {
			...config,
			temperature: 3, // out of range
		};

		expect(() => {
			new AgentHarness({ config: badConfig, backend });
		}).toThrow();
	});

	void it("accepts valid queue modes", () => {
		const queueConfig: AgentConfig = {
			...config,
			steeringQueueMode: "all",
			followUpQueueMode: "one-at-a-time",
		};

		const harness = new AgentHarness({ config: queueConfig, backend });
		expect(harness.phase).toBe("idle");
	});

	void it("accepts valid thinkingLevel", () => {
		const thinkingConfig: AgentConfig = {
			...config,
			thinkingLevel: "high",
		};

		const harness = new AgentHarness({ config: thinkingConfig, backend });
		expect(harness.phase).toBe("idle");
	});

	void it("exposes loopDetector via getLoopDetector", () => {
		const harness = new AgentHarness({ config, backend });
		const detector = harness.getLoopDetector();

		expect(detector).toBeDefined();
		expect(typeof detector.checkToolCall).toBe("function");
		expect(typeof detector.recordFailure).toBe("function");
		expect(typeof detector.consumeTurn).toBe("function");
	});

	void it("enforces positive numeric constraints", () => {
		const configs = [
			{ ...config, maxTokens: -1 },
			{ ...config, maxRetries: -1 },
			{ ...config, contextWindowTokens: 0 },
			{ ...config, maxIterations: -1 },
		];

		for (const badCfg of configs) {
			expect(() => {
				new AgentHarness({ config: badCfg, backend });
			}).toThrow();
		}
	});

	void it("enforces fraction constraints (0-1)", () => {
		const badConfigs = [
			{ ...config, proactiveCompactionFraction: 1.5 },
			{ ...config, proactiveCompactionFraction: -0.1 },
			{ ...config, proactiveCompactionFraction: 0 },
		];

		for (const badCfg of badConfigs) {
			expect(() => {
				new AgentHarness({ config: badCfg, backend });
			}).toThrow();
		}
	});
});
