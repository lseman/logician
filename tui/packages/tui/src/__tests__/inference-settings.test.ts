import assert from "node:assert/strict";
import { test } from "node:test";
import {
	cycleInferenceMode,
	INFERENCE_MODE_ORDER,
	setInferenceMode,
	type InferenceMode,
} from "../app/inference-settings.ts";

function createContext() {
	const applied: string[] = [];
	const savedNotifications: string[] = [];
	const ctx = {
		bridge: { setInferenceMode: (mode: string) => applied.push(mode) },
		statusPanel: { update: () => {} },
		tui: { requestRender: () => {} },
		inferenceMode: "instruct-general" as InferenceMode,
		thinkingLevel: "off",
		notify: (message: string) => savedNotifications.push(message),
	};
	return { ctx, applied, savedNotifications };
}

void test("all inference presets can be selected", () => {
	const { ctx, applied } = createContext();
	for (const mode of INFERENCE_MODE_ORDER) setInferenceMode(ctx as never, mode);
	assert.deepEqual(applied, INFERENCE_MODE_ORDER);
});

void test("inference mode cycling reaches every preset and wraps", () => {
	const { ctx } = createContext();
	ctx.inferenceMode = "thinking-general";
	const visited: string[] = [];
	for (let i = 0; i < INFERENCE_MODE_ORDER.length; i++) {
		cycleInferenceMode(ctx as never);
		visited.push(ctx.inferenceMode);
	}
	const start = INFERENCE_MODE_ORDER.indexOf("thinking-general");
	assert.deepEqual(visited, [
		...INFERENCE_MODE_ORDER.slice(start + 1),
		...INFERENCE_MODE_ORDER.slice(0, start + 1),
	]);
});
