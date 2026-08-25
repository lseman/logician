import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { resolveRuntimeConfig } from "../../runtime/configuration/runtime-config.ts";

function configuredWorkspace(): string {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-runtime-config-"));
	writeFileSync(
		path.join(cwd, ".logician.json"),
		JSON.stringify({
			baseUrl: "http://config.test:8000",
			model: "config-model",
			executionProfile: "minimal",
			thinkingLevel: "high",
			inferenceMode: "none",
			compaction: { enabled: true },
			maxParallelAgents: 4,
			lsp: { enabled: false, timeoutMs: 3210 },
			permissionMode: "ask",
			toolExecution: "sequential",
			hooks: true,
			rtkProxyEnabled: true,
			graphicianEnabled: false,
			fffgrepEnabled: false,
			autoRetryEnabled: false,
			maxRetries: 2,
			retryBaseDelayMs: 25,
			turnTimeoutMs: 5000,
			cacheSize: 64,
			cacheTtlMs: 2000,
			memory: true,
			memoryExtractor: {
				baseUrl: "http://memory.test:8081",
				model: "small-extractor",
			},
			memoryViewer: false,
			memoryViewerPort: 4321,
			memoryEmbeddings: true,
			memoryEmbeddingModel: "local/test-embedder",
			reasoner: "reflexion",
			reasonerConfig: { maxTrials: 2 },
			legroom: {
				mode: "sdk",
				python: "/opt/legroom/bin/python",
				failOpen: false,
				timeoutMs: 12000,
				config: { protect_recent: 2 },
			},
		}),
		"utf8",
	);
	return cwd;
}

void test("runtime resolver applies shared environment precedence", () => {
	const resolved = resolveRuntimeConfig(configuredWorkspace(), {
		HOME: mkdtempSync(path.join(tmpdir(), "logician-runtime-empty-home-")),
		LOGICIAN_LLM_URL: "http://env.test:9000",
		LOGICIAN_MODEL: "env-model",
		LOGICIAN_HOOKS: "0",
	});

	assert.equal(resolved.bridge.baseUrl, "http://env.test:9000");
	assert.equal(resolved.bridge.model, "env-model");
	assert.equal(resolved.bridge.executionProfile, "minimal");
	assert.equal(resolved.bridge.thinkingLevel, "high");
	assert.equal(resolved.bridge.inferenceMode, "none");
	assert.equal(resolved.bridge.proactiveCompactionEnabled, true);
	assert.deepEqual(resolved.bridge.compaction, { enabled: true });
	assert.equal(resolved.bridge.maxParallelAgents, 4);
	assert.deepEqual(resolved.bridge.lsp, { enabled: false, timeoutMs: 3210 });
	assert.equal(resolved.bridge.configPath, resolved.configPath);
	assert.equal(resolved.bridge.runtimeHooksEnabled, false);
	assert.equal(resolved.bridge.toolExecution, "sequential");
	assert.equal(resolved.bridge.permissions?.mode, "ask");
	assert.equal(resolved.bridge.rtkProxyEnabled, true);
	assert.equal(resolved.bridge.graphicianEnabled, false);
	assert.equal(resolved.bridge.fffgrepEnabled, false);
	assert.equal(resolved.bridge.autoRetryEnabled, false);
	assert.equal(resolved.bridge.maxRetries, 2);
	assert.equal(resolved.bridge.retryBaseDelayMs, 25);
	assert.equal(resolved.bridge.turnTimeoutMs, 5000);
	assert.equal(resolved.bridge.cacheSize, 64);
	assert.equal(resolved.bridge.cacheTtlMs, 2000);
	assert.equal(
		resolved.bridge.memory?.extractorBaseUrl,
		"http://memory.test:8081",
	);
	assert.equal(resolved.bridge.memory?.extractorModel, "small-extractor");
	assert.equal(resolved.bridge.memory?.viewerEnabled, false);
	assert.equal(resolved.bridge.memory?.viewerPort, 4321);
	assert.equal(resolved.bridge.memory?.embeddingsEnabled, true);
	assert.equal(resolved.bridge.memory?.embeddingModel, "local/test-embedder");
	assert.equal(resolved.bridge.reasoner, "reflexion");
	assert.deepEqual(resolved.bridge.reasonerConfig, { maxTrials: 2 });
	assert.deepEqual(resolved.bridge.legroom, {
		mode: "sdk",
		python: "/opt/legroom/bin/python",
		failOpen: false,
		timeoutMs: 12000,
		config: { protect_recent: 2 },
	});
});

void test("reasoners are disabled by default", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-runtime-defaults-"));
	const resolved = resolveRuntimeConfig(cwd, {});
	assert.equal(resolved.bridge.reasoner, "none");
	assert.equal(resolved.bridge.memory?.embeddingsEnabled, false);
});

void test("untrusted runtime resolution ignores project configuration", () => {
	const home = mkdtempSync(path.join(tmpdir(), "logician-runtime-home-"));
	const settingsDir = path.join(home, ".logician");
	mkdirSync(settingsDir, { recursive: true });
	writeFileSync(
		path.join(settingsDir, "settings.json"),
		JSON.stringify({
			baseUrl: "http://global.test:7000",
			model: "global-model",
			permissionMode: "acceptEdits",
			compaction: { reserveTokens: 8_000, keepRecentTokens: 12_000 },
			lsp: { enabled: true, timeoutMs: 5_000 },
		}),
		"utf8",
	);
	const resolved = resolveRuntimeConfig(
		configuredWorkspace(),
		{ HOME: home },
		{ loadProjectConfig: false },
	);
	assert.equal(resolved.source.model, "global-model");
	assert.equal(resolved.bridge.model, "global-model");
	assert.equal(resolved.bridge.baseUrl, "http://global.test:7000");
	assert.equal(resolved.bridge.permissions?.mode, "acceptEdits");
	assert.equal(resolved.bridge.projectTrusted, false);
});

void test("trusted runtime resolution overlays project config on global settings", () => {
	const home = mkdtempSync(path.join(tmpdir(), "logician-runtime-home-"));
	const settingsDir = path.join(home, ".logician");
	const workspace = path.join(home, "workspace");
	mkdirSync(settingsDir, { recursive: true });
	mkdirSync(workspace, { recursive: true });
	writeFileSync(
		path.join(settingsDir, "settings.json"),
		JSON.stringify({
			baseUrl: "http://global.test:7000",
			model: "global-model",
			permissionMode: "acceptEdits",
			compaction: { reserveTokens: 8_000, keepRecentTokens: 12_000 },
			lsp: { enabled: true, timeoutMs: 5_000 },
		}),
		"utf8",
	);
	writeFileSync(
		path.join(workspace, ".logician.json"),
		JSON.stringify({
			compaction: { enabled: true },
			lsp: { timeoutMs: 1_000 },
			mcpServers: {
				project: { command: "project-mcp" },
			},
		}),
		"utf8",
	);

	const resolved = resolveRuntimeConfig(
		workspace,
		{ HOME: home },
		{ loadProjectConfig: true },
	);

	assert.equal(resolved.configPath, path.join(workspace, ".logician.json"));
	assert.equal(resolved.bridge.model, "global-model");
	assert.equal(resolved.bridge.baseUrl, "http://global.test:7000");
	assert.equal(resolved.bridge.permissions?.mode, "acceptEdits");
	assert.deepEqual(resolved.source.mcpServers, {
		project: { command: "project-mcp" },
	});
	assert.equal(resolved.bridge.projectTrusted, true);
	assert.deepEqual(resolved.bridge.compaction, {
		reserveTokens: 8_000,
		keepRecentTokens: 12_000,
		enabled: true,
	});
	assert.deepEqual(resolved.bridge.lsp, {
		enabled: true,
		timeoutMs: 1_000,
	});
});
