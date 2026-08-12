import { test } from "bun:test";
import assert from "node:assert/strict";
import { LogicianTUI } from "../app/tui.ts";

void test("thinking-level transition synchronizes runtime, local state, and status", () => {
	let bridgeLevel = "";
	let statusLevel = "";
	const instance = Object.create(LogicianTUI.prototype) as {
		thinkingLevel: string;
		bridge: { setThinkingLevel(level: string): void };
		statusPanel: { update(info: { thinkingLevel: string }): void };
		applyThinkingLevel(level: string): void;
	};
	instance.thinkingLevel = "off";
	instance.bridge = {
		setThinkingLevel: level => {
			bridgeLevel = level;
		},
	};
	instance.statusPanel = {
		update: ({ thinkingLevel }) => {
			statusLevel = thinkingLevel;
		},
	};

	instance.applyThinkingLevel("high");

	assert.equal(instance.thinkingLevel, "high");
	assert.equal(bridgeLevel, "high");
	assert.equal(statusLevel, "high");
});
