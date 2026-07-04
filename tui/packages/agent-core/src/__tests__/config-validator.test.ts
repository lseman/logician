import { describe, it, expect } from "bun:test";
import { validateConfig, throwOnValidationErrors } from "../core/config-validator.ts";
import type { AgentConfig } from "../core/types.ts";

describe("config validator", () => {
	const validConfig: AgentConfig = {
		baseUrl: "https://api.example.com",
		model: "claude-opus",
		temperature: 1,
		contextWindowTokens: 200000,
	};

	it("passes valid config", () => {
		const errors = validateConfig(validConfig);
		expect(errors).toHaveLength(0);
	});

	it("rejects missing baseUrl", () => {
		const config = { ...validConfig, baseUrl: "" };
		const errors = validateConfig(config);
		expect(errors.some((e) => e.field === "baseUrl")).toBe(true);
	});

	it("rejects invalid temperature", () => {
		const config = { ...validConfig, temperature: 3 };
		const errors = validateConfig(config);
		expect(errors.some((e) => e.field === "temperature")).toBe(true);
	});

	it("rejects invalid thinkingLevel", () => {
		const config = { ...validConfig, thinkingLevel: "ultra" as any };
		const errors = validateConfig(config);
		expect(errors.some((e) => e.field === "thinkingLevel")).toBe(true);
	});

	it("rejects invalid queue mode", () => {
		const config = { ...validConfig, steeringQueueMode: "invalid" as any };
		const errors = validateConfig(config);
		expect(errors.some((e) => e.field === "steeringQueueMode")).toBe(true);
	});

	it("throws on validation errors", () => {
		const config = { ...validConfig, baseUrl: "" };
		expect(() => {
			const errors = validateConfig(config);
			throwOnValidationErrors(errors);
		}).toThrow();
	});

	it("allows valid optional fields", () => {
		const config: AgentConfig = {
			...validConfig,
			maxRetries: 3,
			retryBaseDelayMs: 1000,
			maxIterations: 50,
		};
		const errors = validateConfig(config);
		expect(errors).toHaveLength(0);
	});
});
