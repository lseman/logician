import assert from "node:assert/strict";
import { test } from "node:test";
import {
	cycleInferenceMode,
	DEFAULT_MODE,
	getInferenceMode,
	INFERENCE_MODES,
	isValidInferenceMode,
} from "../core/configuration/inference-modes.ts";

void test("INFERENCE_MODES has exactly 4 entries", () => {
	assert.equal(INFERENCE_MODES.size, 4);
});

void test("getInferenceMode returns correct params", () => {
	const mode = getInferenceMode("thinking-general");
	assert.ok(mode);
	assert.equal(mode.label, "Think Gen");
	assert.equal(mode.thinking, true);
	assert.equal(mode.params.temperature, 1.0);
	assert.equal(mode.params.top_p, 0.95);
	assert.equal(mode.params.top_k, 20);
	assert.equal(mode.params.min_p, 0.0);
	assert.equal(mode.params.presence_penalty, 1.5);
	assert.equal(mode.params.repetition_penalty, 1.0);
});

void test("getInferenceMode returns undefined for unknown mode", () => {
	const mode = getInferenceMode("nonexistent" as "thinking-general");
	assert.equal(mode, undefined);
});

void test("isValidInferenceMode returns correct values", () => {
	assert.equal(isValidInferenceMode("thinking-general"), true);
	assert.equal(isValidInferenceMode("thinking-coding"), true);
	assert.equal(isValidInferenceMode("instruct-general"), true);
	assert.equal(isValidInferenceMode("instruct-reasoning"), true);
	assert.equal(isValidInferenceMode("invalid"), false);
});

void test("cycleInferenceMode cycles through all modes in order", () => {
	const modes: Array<"thinking-general" | "thinking-coding" | "instruct-general" | "instruct-reasoning"> =
		["thinking-general", "thinking-coding", "instruct-general", "instruct-reasoning"];
	for (let i = 0; i < modes.length; i++) {
		assert.equal(cycleInferenceMode(modes[i]), modes[(i + 1) % modes.length]);
	}
});

void test("cycleInferenceMode wraps around from last to first", () => {
	const mode = cycleInferenceMode("instruct-reasoning");
	assert.equal(mode, "thinking-general");
});

void test("instruct-reasoning has high temp and presence penalty", () => {
	const mode = getInferenceMode("instruct-reasoning");
	assert.ok(mode);
	assert.equal(mode.params.temperature, 1.0);
	assert.equal(mode.params.presence_penalty, 1.5);
});

void test("thinking-coding has lower temperature and zero presence penalty", () => {
	const mode = getInferenceMode("thinking-coding");
	assert.ok(mode);
	assert.equal(mode.params.temperature, 0.6);
	assert.equal(mode.params.presence_penalty, 0.0);
});

void test("instruct-general has balanced sampling", () => {
	const mode = getInferenceMode("instruct-general");
	assert.ok(mode);
	assert.equal(mode.params.temperature, 0.7);
	assert.equal(mode.params.top_p, 0.8);
});
