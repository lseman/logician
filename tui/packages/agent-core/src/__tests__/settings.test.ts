// ── Settings command tests ───────────────────────────────────────────────────

import test, { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import {
	parseSettingsCommand,
	buildSettingsSnapshot,
} from "../core/settings.ts";

describe("buildSettingsSnapshot", () => {
	it("includes all sections", () => {
		const snap = buildSettingsSnapshot({
			currentModel: "claude-sonnet-4",
			models: ["gpt-4", "gemini-pro"],
			temperature: 0.7,
			maxTokens: 8192,
			maxIterations: 30,
			contextWindowTokens: 128000,
			thinkingLevel: "medium",
			proactiveCompactionEnabled: true,
			proactiveCompactionFraction: 0.8,
			loopDetectionEnabled: true,
			guardsEnabled: true,
			continuationEnabled: false,
			budgetStopEnabled: false,
			toolExecution: "sequential",
			steeringQueueMode: "all",
			followUpQueueMode: "one-at-a-time",
			autoRetryEnabled: true,
			maxRetries: 3,
			retryBaseDelayMs: 500,
			turnTimeoutMs: 60000,
			acceptAllPermissions: false,
		});

		assert.ok(snap.includes("── Runtime Settings ──"));
		assert.ok(snap.includes("── Reasoning ──"));
		assert.ok(snap.includes("── Guardrails ──"));
		assert.ok(snap.includes("── Compaction ──"));
		assert.ok(snap.includes("── Execution ──"));
		assert.ok(snap.includes("── Permissions ──"));
		assert.ok(snap.includes("── Quick Changes ──"));
	});

	it("shows model info", () => {
		const snap = buildSettingsSnapshot({
			currentModel: "claude-sonnet-4",
			models: ["gpt-4", "gemini-pro"],
			temperature: 0.7,
			maxTokens: undefined,
			maxIterations: 30,
			contextWindowTokens: undefined,
			thinkingLevel: "off",
			proactiveCompactionEnabled: undefined,
			proactiveCompactionFraction: undefined,
			loopDetectionEnabled: undefined,
			guardsEnabled: undefined,
			continuationEnabled: undefined,
			budgetStopEnabled: undefined,
			toolExecution: undefined,
			steeringQueueMode: undefined,
			followUpQueueMode: undefined,
			autoRetryEnabled: undefined,
			maxRetries: undefined,
			retryBaseDelayMs: undefined,
			turnTimeoutMs: undefined,
			acceptAllPermissions: true,
		});

		assert.ok(snap.includes("claude-sonnet-4"));
		assert.ok(snap.includes("gpt-4, gemini-pro"));
	});

	it("shows unset values", () => {
		const snap = buildSettingsSnapshot({
			currentModel: "test-model",
			models: [],
			temperature: 1.0,
			maxTokens: undefined,
			maxIterations: 10,
			contextWindowTokens: undefined,
			thinkingLevel: "high",
			proactiveCompactionEnabled: undefined,
			proactiveCompactionFraction: undefined,
			loopDetectionEnabled: undefined,
			guardsEnabled: undefined,
			continuationEnabled: undefined,
			budgetStopEnabled: undefined,
			toolExecution: undefined,
			steeringQueueMode: undefined,
			followUpQueueMode: undefined,
			autoRetryEnabled: undefined,
			maxRetries: undefined,
			retryBaseDelayMs: undefined,
			turnTimeoutMs: undefined,
			acceptAllPermissions: true,
		});

		assert.ok(snap.includes("unset"));
	});

	it("shows quick change hints", () => {
		const snap = buildSettingsSnapshot({
			currentModel: "test",
			models: [],
			temperature: 0.0,
			maxTokens: undefined,
			maxIterations: 1,
			contextWindowTokens: undefined,
			thinkingLevel: "off",
			proactiveCompactionEnabled: false,
			proactiveCompactionFraction: 0.5,
			loopDetectionEnabled: false,
			guardsEnabled: false,
			continuationEnabled: false,
			budgetStopEnabled: false,
			toolExecution: "sequential",
			steeringQueueMode: "all",
			followUpQueueMode: "all",
			autoRetryEnabled: false,
			maxRetries: 1,
			retryBaseDelayMs: 100,
			turnTimeoutMs: 30000,
			acceptAllPermissions: true,
		});

		assert.ok(snap.includes("quick changes") || snap.includes("Quick Changes"));
		assert.ok(snap.includes("/settings thinking"));
		assert.ok(snap.includes("/settings model"));
		assert.ok(snap.includes("/settings compaction"));
	});
});

describe("parseSettingsCommand — no args", () => {
	it("returns view action with empty snapshot", () => {
		const result = parseSettingsCommand("");
		assert.strictEqual(result.type, "view");
		assert.strictEqual((result as { snapshot: string }).snapshot, "");
	});

	it("returns view for whitespace-only", () => {
		const result = parseSettingsCommand("   ");
		assert.strictEqual(result.type, "view");
	});
});

describe("parseSettingsCommand — thinking level", () => {
	it("accepts valid levels", () => {
		for (const level of ["off", "minimal", "low", "medium", "high", "xhigh"]) {
			const result = parseSettingsCommand(`thinking ${level}`);
			assert.strictEqual(result.type, "change");
			assert.strictEqual(result.key, "thinking_level");
			assert.strictEqual(result.value, level);
		}
	});

	it("rejects invalid levels", () => {
		const result = parseSettingsCommand("thinking invalid");
		assert.strictEqual(result.type, "error");
		assert.ok((result as { message: string }).message.includes("Invalid"));
	});

	it("requires level value", () => {
		const result = parseSettingsCommand("thinking");
		assert.strictEqual(result.type, "error");
		assert.ok((result as { message: string }).message.includes("Usage"));
	});

	it("case-insensitive", () => {
		const result = parseSettingsCommand("thinking HIGH");
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.value, "high");
	});
});

describe("parseSettingsCommand — model", () => {
	it("accepts model name", () => {
		const result = parseSettingsCommand("model claude-sonnet-4");
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.key, "model");
		assert.strictEqual(result.value, "claude-sonnet-4");
	});

	it("rejects missing model name", () => {
		const result = parseSettingsCommand("model");
		assert.strictEqual(result.type, "error");
	});
});

describe("parseSettingsCommand — model-cycle", () => {
	it("returns cycle action", () => {
		const result = parseSettingsCommand("model-cycle");
		assert.strictEqual(result.type, "cycle");
	});

	it("accepts underscore variant", () => {
		const result = parseSettingsCommand("model_cycle");
		assert.strictEqual(result.type, "cycle");
	});
});

describe("parseSettingsCommand — temperature", () => {
	it("accepts valid temperature", () => {
		const result = parseSettingsCommand("temp 0.7");
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.key, "temperature");
		assert.strictEqual(result.value, "0.7");
	});

	it("accepts integer temperature", () => {
		const result = parseSettingsCommand("temp 1");
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.value, "1");
	});

	it("rejects out-of-range temperature", () => {
		const result = parseSettingsCommand("temp 3.0");
		assert.strictEqual(result.type, "error");
	});

	it("rejects negative temperature", () => {
		const result = parseSettingsCommand("temp -0.5");
		assert.strictEqual(result.type, "error");
	});

	it("rejects non-numeric", () => {
		const result = parseSettingsCommand("temp abc");
		assert.strictEqual(result.type, "error");
	});
});

describe("parseSettingsCommand — max-tokens", () => {
	it("accepts valid integer", () => {
		const result = parseSettingsCommand("max-tokens 8192");
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.key, "max_tokens");
		assert.strictEqual(result.value, "8192");
	});

	it("accepts underscore variant", () => {
		const result = parseSettingsCommand("max_tokens 4096");
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.value, "4096");
	});

	it("rejects non-numeric", () => {
		const result = parseSettingsCommand("max-tokens abc");
		assert.strictEqual(result.type, "error");
	});

	it("rejects zero", () => {
		const result = parseSettingsCommand("max-tokens 0");
		assert.strictEqual(result.type, "error");
	});

	it("rejects negative", () => {
		const result = parseSettingsCommand("max-tokens -1");
		assert.strictEqual(result.type, "error");
	});
});

describe("parseSettingsCommand — max-iterations", () => {
	it("accepts valid integer", () => {
		const result = parseSettingsCommand("max-iterations 20");
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.key, "max_iterations");
		assert.strictEqual(result.value, "20");
	});

	it("rejects non-numeric", () => {
		const result = parseSettingsCommand("max-iterations abc");
		assert.strictEqual(result.type, "error");
	});
});

describe("parseSettingsCommand — loop-detection", () => {
	it("accepts on", () => {
		const result = parseSettingsCommand("loop-detection on");
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.key, "loop_detection");
		assert.strictEqual(result.value, "on");
	});

	it("accepts off", () => {
		const result = parseSettingsCommand("loop-detection off");
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.value, "off");
	});

	it("rejects missing state", () => {
		const result = parseSettingsCommand("loop-detection");
		assert.strictEqual(result.type, "error");
	});

	it("rejects invalid state", () => {
		const result = parseSettingsCommand("loop-detection maybe");
		assert.strictEqual(result.type, "error");
	});
});

type ChangeResult = { type: "change"; key: string; value: string };

describe("parseSettingsCommand — guards", () => {
	it("accepts on/off", () => {
		const resultOn = parseSettingsCommand("guards on") as ChangeResult;
		assert.strictEqual(resultOn.type, "change");
		assert.strictEqual(resultOn.value, "on");

		const resultOff = parseSettingsCommand("guards off") as ChangeResult;
		assert.strictEqual(resultOff.value, "off");
	});
});

describe("parseSettingsCommand — compaction", () => {
	it("accepts on/off", () => {
		const resultOn = parseSettingsCommand("compaction on") as ChangeResult;
		assert.strictEqual(resultOn.type, "change");
		assert.strictEqual(resultOn.key, "compaction");

		const resultOff = parseSettingsCommand("compaction off") as ChangeResult;
		assert.strictEqual(resultOff.value, "off");
	});
});

describe("parseSettingsCommand — permissions", () => {
	it("accepts valid modes", () => {
		for (const mode of ["acceptAll", "acceptEdits", "ask", "plan"]) {
			const result = parseSettingsCommand(`permissions ${mode}`) as ChangeResult;
			assert.strictEqual(result.type, "change");
			assert.strictEqual(result.key, "permissions");
		}
	});

	it("is case-insensitive", () => {
		const result = parseSettingsCommand("permissions ASK") as ChangeResult;
		assert.strictEqual(result.type, "change");
		assert.strictEqual(result.value, "ask");
	});

	it("rejects invalid mode", () => {
		const result = parseSettingsCommand("permissions invalid");
		assert.strictEqual(result.type, "error");
	});
});

describe("parseSettingsCommand — unknown subcommand", () => {
	it("returns error", () => {
		const result = parseSettingsCommand("foo bar");
		assert.strictEqual(result.type, "error");
		assert.ok((result as { message: string }).message.includes("Unknown"));
	});
});
