import { describe, it, expect, beforeEach } from "bun:test";
import { AgentHarness } from "../core/harness.ts";
import type { AgentConfig, Tool } from "../core/types.ts";
import { OpenAIBackend } from "../core/backend.ts";

describe("harness with full guards", () => {
	let config: AgentConfig;
	let backend: OpenAIBackend;

	beforeEach(() => {
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
			apiKey: "test-key",
			model: config.model,
		});
	});

	it("constructs with all config fields validated", () => {
		const harness = new AgentHarness({
			config,
			backend,
		});

		expect(harness.phase).toBe("idle");
		expect(harness.getLoopDetector()).toBeDefined();
	});

	it("rejects invalid config at construction", () => {
		const badConfig: AgentConfig = {
			...config,
			temperature: 3, // out of range
		};

		expect(() => {
			new AgentHarness({ config: badConfig, backend });
		}).toThrow();
	});

	it("accepts valid queue modes", () => {
		const queueConfig: AgentConfig = {
			...config,
			steeringQueueMode: "all",
			followUpQueueMode: "one-at-a-time",
		};

		const harness = new AgentHarness({ config: queueConfig, backend });
		expect(harness.phase).toBe("idle");
	});

	it("accepts valid thinkingLevel", () => {
		const thinkingConfig: AgentConfig = {
			...config,
			thinkingLevel: "high",
		};

		const harness = new AgentHarness({ config: thinkingConfig, backend });
		expect(harness.phase).toBe("idle");
	});

	it("exposes loopDetector via getLoopDetector", () => {
		const harness = new AgentHarness({ config, backend });
		const detector = harness.getLoopDetector();

		expect(detector).toBeDefined();
		expect(typeof detector.checkToolCall).toBe("function");
		expect(typeof detector.recordFailure).toBe("function");
		expect(typeof detector.consumeTurn).toBe("function");
	});

	it("enforces positive numeric constraints", () => {
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

	it("enforces fraction constraints (0-1)", () => {
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

	it("respects deprecation: reasonerId ignored", () => {
		const legacyConfig: AgentConfig = {
			...config,
			reasonerId: "some-reasoner", // deprecated, should not error
		};

		const harness = new AgentHarness({ config: legacyConfig, backend });
		expect(harness.phase).toBe("idle");
	});
});
