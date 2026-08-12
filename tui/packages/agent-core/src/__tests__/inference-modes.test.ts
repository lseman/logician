import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	cycleInferenceMode,
	getInferenceMode,
	INFERENCE_MODES,
	isValidInferenceMode,
} from "../agent/configuration/inference-modes.ts";

void test("INFERENCE_MODES has exactly 10 entries", () => {
	assert.equal(INFERENCE_MODES.size, 10);
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
	assert.equal(isValidInferenceMode("auto"), true);
	assert.equal(isValidInferenceMode("none"), true);
	assert.equal(isValidInferenceMode("thinking-general"), true);
	assert.equal(isValidInferenceMode("thinking-coding"), true);
	assert.equal(isValidInferenceMode("instruct-general"), true);
	assert.equal(isValidInferenceMode("instruct-reasoning"), true);
	assert.equal(isValidInferenceMode("instruct-coding"), true);
	assert.equal(isValidInferenceMode("deterministic"), true);
	assert.equal(isValidInferenceMode("creative"), true);
	assert.equal(isValidInferenceMode("analytical"), true);
	assert.equal(isValidInferenceMode("invalid"), false);
});

void test("cycleInferenceMode cycles through all modes in order", () => {
	const modes = [
		"auto",
		"none",
		"thinking-general",
		"thinking-coding",
		"instruct-general",
		"instruct-reasoning",
		"instruct-coding",
		"deterministic",
		"creative",
		"analytical",
	] as const;
	for (let i = 0; i < modes.length; i++) {
		assert.equal(cycleInferenceMode(modes[i]), modes[(i + 1) % modes.length]);
	}
});

void test("cycleInferenceMode wraps around from last to first", () => {
	const mode = cycleInferenceMode("analytical");
	assert.equal(mode, "auto");
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

void test("instruct-coding has low temperature and zero presence penalty", () => {
	const mode = getInferenceMode("instruct-coding");
	assert.ok(mode);
	assert.equal(mode.label, "Code");
	assert.equal(mode.thinking, false);
	assert.equal(mode.params.temperature, 0.3);
	assert.equal(mode.params.presence_penalty, 0.0);
});

void test("deterministic has zero temperature and top_p", () => {
	const mode = getInferenceMode("deterministic");
	assert.ok(mode);
	assert.equal(mode.label, "Exact");
	assert.equal(mode.thinking, false);
	assert.equal(mode.params.temperature, 0.0);
	assert.equal(mode.params.top_p, 0.0);
	assert.equal(mode.params.top_k, 1);
});

void test("creative has ultra-high temperature", () => {
	const mode = getInferenceMode("creative");
	assert.ok(mode);
	assert.equal(mode.label, "Creative");
	assert.equal(mode.thinking, false);
	assert.equal(mode.params.temperature, 1.3);
	assert.equal(mode.params.top_p, 0.99);
	assert.equal(mode.params.top_k, 40);
	assert.equal(mode.params.presence_penalty, 2.0);
});

void test("analytical has low temperature and tight top_p", () => {
	const mode = getInferenceMode("analytical");
	assert.ok(mode);
	assert.equal(mode.label, "Analyze");
	assert.equal(mode.thinking, false);
	assert.equal(mode.params.temperature, 0.2);
	assert.equal(mode.params.top_p, 0.7);
	assert.equal(mode.params.presence_penalty, 0.5);
	assert.equal(mode.params.repetition_penalty, 1.1);
});

void test("none mode omits sampling params to provider", () => {
	const mode = getInferenceMode("none");
	assert.ok(mode);
	assert.equal(mode.label, "Provider");
	assert.equal(mode.thinking, false);
	assert.equal(mode.useProviderDefaults, true);
	assert.equal(isValidInferenceMode("none"), true);
});
