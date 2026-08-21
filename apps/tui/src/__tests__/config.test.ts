import { test } from "bun:test";
import assert from "node:assert/strict";
import { validateConfig } from "@logician/log-runtime/configuration";

void test("validateConfig rejects non-object input", () => {
	const w: string[] = [];
	const cfg = validateConfig(null, w);
	assert.deepEqual(cfg, {});
	assert.ok(w.length > 0);
});

void test("validateConfig warns on unknown keys", () => {
	const w: string[] = [];
	const cfg = validateConfig(
		{ baseUrl: "http://localhost:8080", bogusField: true, anotherBogus: 42 },
		w,
	);
	assert.equal(cfg.baseUrl, "http://localhost:8080");
	assert.equal(w.length, 2);
	assert.ok(w[0].includes("Unknown config key"));
});

void test("validateConfig validates URL format for baseUrl", () => {
	const w: string[] = [];
	const cfg = validateConfig({ baseUrl: "not-a-url" }, w);
	assert.equal(cfg.baseUrl, undefined);
	assert.ok(w.length > 0);
});

void test("validateConfig accepts valid http and https URLs", () => {
	const w: string[] = [];
	const cfg = validateConfig(
		{ baseUrl: "http://localhost:8080", llmUrl: "https://api.example.com/v1" },
		w,
	);
	assert.equal(cfg.baseUrl, "http://localhost:8080");
	assert.equal(cfg.llmUrl, "https://api.example.com/v1");
	assert.equal(w.length, 0);
});

void test("validateConfig clamps temperature to [0, 2]", () => {
	const w: string[] = [];
	assert.equal(validateConfig({ temperature: -1 }, w).temperature, 0);
	assert.equal(validateConfig({ temperature: 3 }, w).temperature, 2);
	assert.equal(validateConfig({ temperature: 0.7 }, w).temperature, 0.7);
});

void test("validateConfig rejects maxTokens <= 0", () => {
	const w: string[] = [];
	const cfg = validateConfig({ maxTokens: -5 }, w);
	assert.equal(cfg.maxTokens, undefined);
	assert.ok(w.length > 0);
});

void test("validateConfig accepts maxTokens > 0", () => {
	const w: string[] = [];
	const cfg = validateConfig({ maxTokens: 4096 }, w);
	assert.equal(cfg.maxTokens, 4096);
	assert.equal(w.length, 0);
});

void test("validateConfig rejects maxIterations <= 0", () => {
	const w: string[] = [];
	const cfg = validateConfig({ maxIterations: 0 }, w);
	assert.equal(cfg.maxIterations, undefined);
	assert.ok(w.length > 0);
});

void test("validateConfig rejects invalid toolExecution values", () => {
	const w: string[] = [];
	const cfg = validateConfig({ toolExecution: "random" }, w);
	assert.equal(cfg.toolExecution, undefined);
	assert.ok(w.length > 0);
});

void test("validateConfig accepts valid toolExecution", () => {
	const w: string[] = [];
	assert.equal(
		validateConfig({ toolExecution: "sequential" }, w).toolExecution,
		"sequential",
	);
	assert.equal(
		validateConfig({ toolExecution: "parallel" }, w).toolExecution,
		"parallel",
	);
});

void test("validateConfig rejects invalid permissionMode", () => {
	const w: string[] = [];
	const cfg = validateConfig({ permissionMode: "evil" }, w);
	assert.equal(cfg.permissionMode, undefined);
	assert.ok(w.length > 0);
});

void test("validateConfig accepts valid permissionMode", () => {
	const w: string[] = [];
	assert.equal(
		validateConfig({ permissionMode: "acceptAll" }, w).permissionMode,
		"acceptAll",
	);
	assert.equal(
		validateConfig({ permissionMode: "ask" }, w).permissionMode,
		"ask",
	);
});

void test("validateConfig validates webSearch sub-object", () => {
	const w: string[] = [];
	const cfg = validateConfig(
		{ webSearch: { baseUrl: "http://search.local", maxResults: 5 } },
		w,
	);
	assert.equal(cfg.webSearch?.baseUrl, "http://search.local");
	assert.equal(cfg.webSearch?.maxResults, 5);
	assert.equal(w.length, 0);
});

void test("validateConfig warns on webSearch maxResults out of range", () => {
	const w: string[] = [];
	const cfg = validateConfig({ webSearch: { maxResults: 200 } }, w);
	assert.equal(cfg.webSearch?.maxResults, undefined);
	assert.ok(w.length > 0);
});

void test("validateConfig warns on unknown webSearch keys", () => {
	const w: string[] = [];
	validateConfig({ webSearch: { fakeField: true } }, w);
	assert.equal(w.length, 1);
	assert.ok(w[0].includes("Unknown webSearch key"));
});

void test("validateConfig rejects webSearch.baseUrl that is not a URL", () => {
	const w: string[] = [];
	const cfg = validateConfig({ webSearch: { baseUrl: "ftp://invalid" } }, w);
	assert.equal(cfg.webSearch?.baseUrl, undefined);
	assert.ok(w.length > 0);
});

void test("validateConfig handles permissions sub-object", () => {
	const w: string[] = [];
	const cfg = validateConfig(
		{ permissions: { allow: ["ls", "cat"], deny: ["rm -rf /"] } },
		w,
	);
	assert.deepEqual(cfg.permissions?.allow, ["ls", "cat"]);
	assert.deepEqual(cfg.permissions?.deny, ["rm -rf /"]);
	assert.equal(w.length, 0);
});

void test("validateConfig accepts the act/plan workflow mode", () => {
	const warnings: string[] = [];
	assert.deepEqual(validateConfig({ workflowMode: "plan" }, warnings), {
		workflowMode: "plan",
		duplicateGuardEnabled: true,
		continuationEnabled: true,
		postEditDiagnostics: true,
		autoRetryEnabled: true,
		ariadneEnabled: true,
		fffgrepEnabled: true,
	});
	assert.deepEqual(warnings, []);
});

void test("validateConfig filters empty strings from permissions arrays", () => {
	const w: string[] = [];
	const cfg = validateConfig(
		{ permissions: { allow: ["ls", "", "  "], deny: [] } },
		w,
	);
	assert.deepEqual(cfg.permissions?.allow, ["ls"]);
	assert.deepEqual(cfg.permissions?.deny, []);
});

void test("validateConfig warns on unknown permissions keys", () => {
	const w: string[] = [];
	validateConfig({ permissions: { fake: true } }, w);
	assert.equal(w.length, 1);
});

void test("validateConfig parses boolean fields from strings", () => {
	const w: string[] = [];
	const cfg = validateConfig({ hooks: "true" }, w);
	assert.equal(cfg.hooks, true);
});

void test("validateConfig parses numeric strings", () => {
	const w: string[] = [];
	const cfg = validateConfig({ temperature: "1.5" }, w);
	assert.equal(cfg.temperature, 1.5);
});

void test("validateConfig trims string values", () => {
	const w: string[] = [];
	const cfg = validateConfig({ model: "  llama-3  " }, w);
	assert.equal(cfg.model, "llama-3");
});

void test("validateConfig handles MCP and mcpServers passthrough", () => {
	const w: string[] = [];
	const cfg = validateConfig(
		{
			mcp: { server1: { url: "http://localhost:3000" } },
			mcpServers: { server2: { args: ["--port", "3001"] } },
		},
		w,
	);
	assert.ok(cfg.mcp);
	assert.ok(cfg.mcpServers);
});

void test("validateConfig rejects non-object webSearch", () => {
	const w: string[] = [];
	const cfg = validateConfig({ webSearch: "invalid" }, w);
	assert.equal(cfg.webSearch, undefined);
	assert.ok(w.length > 0);
});

void test("validateConfig rejects non-object permissions", () => {
	const w: string[] = [];
	const cfg = validateConfig({ permissions: "invalid" }, w);
	assert.equal(cfg.permissions, undefined);
	assert.ok(w.length > 0);
});

void test("validateConfig empty config applies defaults with no warnings", () => {
	const w: string[] = [];
	const cfg = validateConfig({}, w);
	assert.deepEqual(cfg, {
		ariadneEnabled: true,
		autoRetryEnabled: true,
		continuationEnabled: true,
		duplicateGuardEnabled: true,
		fffgrepEnabled: true,
		postEditDiagnostics: true,
	});
	assert.equal(w.length, 0);
});

void test("validateConfig clamps maxTotalTokens > 0", () => {
	const w: string[] = [];
	assert.equal(
		validateConfig({ maxTotalTokens: -10 }, w).maxTotalTokens,
		undefined,
	);
	assert.equal(
		validateConfig({ maxTotalTokens: 10000 }, w).maxTotalTokens,
		10000,
	);
});

void test("validateConfig clamps contextWindow and contextWindowTokens > 0", () => {
	const w: string[] = [];
	assert.equal(
		validateConfig({ contextWindow: -5 }, w).contextWindow,
		undefined,
	);
	assert.equal(
		validateConfig({ contextWindowTokens: 4096 }, w).contextWindowTokens,
		4096,
	);
});
