// ── Settings defs generator tests ────────────────────────────────────────────

import { describe, it } from "bun:test";
import { strict as assert } from "node:assert";
import {
	buildSettingsDefs,
	type SettingsData,
} from "../app/overlay-controllers/settings-defs.ts";

const view: SettingsData = {
	model: "test-model",
	temperature: 0.7,
	maxTokens: 4096,
	maxIterations: 30,
	thinkingLevel: "low",
	inferenceMode: "auto",
	permissionMode: "ask",
	executionProfile: "minimal",
	guardsEnabled: true,
	proactiveCompactionEnabled: true,
	postEditDiagnostics: true,
	rtkProxyEnabled: false,
	graphicianEnabled: true,
	fffgrepEnabled: false,
	legroomEnabled: true,
	memoriamEnabled: false,
	duplicateGuardEnabled: true,
	failureGuardEnabled: false,
	continuationEnabled: true,
	autoRetryEnabled: true,
	progressStopEnabled: false,
	guardMode: "on",
};

describe("buildSettingsDefs", () => {
	it("emits the registry UI entries in display order with stable tabs", () => {
		const defs = buildSettingsDefs(view, "act");
		assert.deepEqual(
			defs.map(def => def.name),
			[
				"Model",
				"Temperature",
				"Max tokens",
				"Max iterations",
				"Thinking level",
				"Workflow mode",
				"Guards",
				"Compaction",
				"Inference mode",
				"Post-edit diagnostics",
				"RTK CLI proxy",
				"Legroom SDK",
				"Memoriam SDK",
				"Graphician",
				"fffgrep",
				"Execution policy",
				"Duplicate-call guard",
				"Failure-loop guard",
				"Continuation",
				"Auto-compact on full context",
				"Budget early-stop",
			],
		);
		assert.deepEqual(
			[...new Set(defs.map(def => def.tab))],
			["Model", "Behavior", "Tools", "Guards"],
		);
	});

	it("projects nested enum integrations as toggles and enums as selects", () => {
		const defs = buildSettingsDefs(view, "plan");
		const legroom = defs.find(def => def.name === "Legroom SDK");
		assert.equal(legroom?.currentValue, "on");
		assert.equal(legroom?.displayType, "toggle");
		const memoriam = defs.find(def => def.name === "Memoriam SDK");
		assert.equal(memoriam?.currentValue, "off");
		const guards = defs.find(def => def.name === "Guards");
		assert.deepEqual(
			guards?.options?.map(option => option.value),
			["auto", "on", "off"],
		);
		assert.equal(guards?.currentValue, "on");
		assert.equal(
			guards?.options?.find(option => option.value === "on")?.current,
			true,
		);
		const workflow = defs.find(def => def.name === "Workflow mode");
		assert.equal(workflow?.currentValue, "plan");
	});

	it("marks the matching number preset current", () => {
		const defs = buildSettingsDefs(view, "act");
		const temperature = defs.find(def => def.name === "Temperature");
		assert.equal(temperature?.displayType, "number");
		assert.equal(
			temperature?.options?.find(option => option.value === "0.7")?.current,
			true,
		);
		const tokens = defs.find(def => def.name === "Max tokens");
		assert.equal(
			tokens?.options?.find(option => option.value === "4096")?.current,
			true,
		);
	});
});
