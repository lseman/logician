import { test } from "bun:test";
import assert from "node:assert/strict";
import type { AgentConfig } from "@logician/agent-core";
import {
	throwOnAgentConfigErrors as throwOnValidationErrors,
	validateAgentConfig as validateConfig,
} from "../../core/configuration/config-validator.ts";

function describe(_name: string, fn: () => void) {
	fn();
}
function it(name: string, fn: () => void | Promise<void>) {
	test(name, fn);
}
function expect<T>(actual: T) {
	return {
		toHaveLength(len: number) {
			assert.equal(Array.isArray(actual) ? actual.length : 0, len);
		},
		toBe(expected: unknown) {
			assert.equal(actual, expected);
		},
		toBeTrue() {
			assert.equal(actual, true);
		},
		toBeFalse() {
			assert.equal(actual, false);
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

void describe("config validator", () => {
	const validConfig: AgentConfig = {
		baseUrl: "https://api.example.com",
		model: "claude-opus",
		temperature: 1,
		contextWindowTokens: 200000,
	};

	void it("passes valid config", () => {
		const errors = validateConfig(validConfig);
		expect(errors).toHaveLength(0);
	});

	void it("rejects missing baseUrl", () => {
		const config = { ...validConfig, baseUrl: "" };
		const errors = validateConfig(config);
		expect(errors.some(e => e.field === "baseUrl")).toBeTrue();
	});

	void it("rejects invalid temperature", () => {
		const config = { ...validConfig, temperature: 3 };
		const errors = validateConfig(config);
		expect(errors.some(e => e.field === "temperature")).toBeTrue();
	});

	void it("rejects invalid thinkingLevel", () => {
		const config = { ...validConfig, thinkingLevel: "ultra" as any };
		const errors = validateConfig(config);
		expect(errors.some(e => e.field === "thinkingLevel")).toBeTrue();
	});

	void it("rejects invalid inferenceMode", () => {
		const config = { ...validConfig, inferenceMode: "bonkers" as any };
		const errors = validateConfig(config);
		expect(errors.some(e => e.field === "inferenceMode")).toBeTrue();
	});

	void it("rejects invalid queue mode", () => {
		const config = { ...validConfig, steeringQueueMode: "invalid" as any };
		const errors = validateConfig(config);
		expect(errors.some(e => e.field === "steeringQueueMode")).toBeTrue();
	});

	void it("throws on validation errors", () => {
		const config = { ...validConfig, baseUrl: "" };
		expect(() => {
			const errors = validateConfig(config);
			throwOnValidationErrors(errors);
		}).toThrow();
	});

	void it("allows valid optional fields", () => {
		const config: AgentConfig = {
			...validConfig,
			maxRetries: 3,
			retryBaseDelayMs: 1000,
			maxIterations: 50,
		};
		const errors = validateConfig(config);
		expect(errors).toHaveLength(0);
	});

	void it("rejects non-positive cache settings", () => {
		const errors = validateConfig({
			...validConfig,
			cacheSize: 0,
			cacheTtlMs: -1,
		});
		expect(errors.some(e => e.field === "cacheSize")).toBeTrue();
		expect(errors.some(e => e.field === "cacheTtlMs")).toBeTrue();
	});
});
