import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	buildDefaultSystemPrompt,
	buildDynamicSystemContext,
	buildStaticSystemPromptPrefix,
	buildSystemPrompt,
} from "../../runtime/context/system-prompt.ts";
import type { Tool } from "@logician/log-core";

void test("buildStaticSystemPromptPrefix generates a deterministic cacheable prefix across different times", () => {
	const toolA: Tool = {
		name: "toolA",
		description: "Tool Alpha description",
		parameters: { type: "object", properties: {} },
		execute: async () => "",
	};
	const toolB: Tool = {
		name: "toolB",
		description: "Tool Beta description",
		parameters: { type: "object", properties: {} },
		execute: async () => "",
	};

	const prefix1 = buildStaticSystemPromptPrefix({
		selectedTools: [toolB, toolA], // intentionally unsorted
		cwd: "/home/user/project",
	});

	const prefix2 = buildStaticSystemPromptPrefix({
		selectedTools: [toolA, toolB], // reversed order
		cwd: "/home/user/project",
	});

	// Both prefixes should be identical for 100% prompt cache hit
	assert.equal(prefix1, prefix2);
	assert.ok(!prefix1.includes("Current date:"));
	assert.ok(!prefix1.includes("Current working directory:"));
});

void test("buildDynamicSystemContext formats dynamic runtime metadata cleanly", () => {
	const dynamicContext = buildDynamicSystemContext({
		cwd: "/workspace/my-app",
		date: "2026-08-21",
	});

	assert.match(dynamicContext, /Current date: 2026-08-21/);
	assert.match(dynamicContext, /Current working directory: \/workspace\/my-app/);
});

void test("buildSystemPrompt integrates static prefix and dynamic context seamlessly", () => {
	const prompt = buildSystemPrompt({
		cwd: "/workspace/my-app",
	});

	assert.match(prompt, /You are Logician/);
	assert.match(prompt, /Current date:/);
	assert.match(prompt, /Current working directory: \/workspace\/my-app/);
});
