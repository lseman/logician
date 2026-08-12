import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { resolveRuntimeConfig } from "../runtime/runtime-config.ts";

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
			permissionMode: "ask",
			toolExecution: "sequential",
			mcpEager: true,
			hooks: true,
			rtkProxyEnabled: true,
			ariadneEnabled: false,
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
		}),
		"utf8",
	);
	return cwd;
}

void test("runtime resolver applies shared environment precedence", () => {
	const resolved = resolveRuntimeConfig(configuredWorkspace(), {
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
	assert.equal(resolved.bridge.runtimeHooksEnabled, false);
	assert.equal(resolved.bridge.toolExecution, "sequential");
	assert.equal(resolved.bridge.permissionMode, "ask");
	assert.equal(resolved.bridge.mcpEager, true);
	assert.equal(resolved.bridge.rtkProxyEnabled, true);
	assert.equal(resolved.bridge.ariadneEnabled, false);
	assert.equal(resolved.bridge.fffgrepEnabled, false);
	assert.equal(resolved.bridge.autoRetryEnabled, false);
	assert.equal(resolved.bridge.maxRetries, 2);
	assert.equal(resolved.bridge.retryBaseDelayMs, 25);
	assert.equal(resolved.bridge.turnTimeoutMs, 5000);
	assert.equal(resolved.bridge.cacheSize, 64);
	assert.equal(resolved.bridge.cacheTtlMs, 2000);
	assert.equal(
		resolved.bridge.memoryExtractorBaseUrl,
		"http://memory.test:8081",
	);
	assert.equal(resolved.bridge.memoryExtractorModel, "small-extractor");
	assert.equal(resolved.bridge.memoryViewerEnabled, false);
	assert.equal(resolved.bridge.memoryViewerPort, 4321);
	assert.equal(resolved.bridge.memoryEmbeddingsEnabled, true);
	assert.equal(resolved.bridge.memoryEmbeddingModel, "local/test-embedder");
	assert.equal(resolved.bridge.reasoner, "reflexion");
	assert.deepEqual(resolved.bridge.reasonerConfig, { maxTrials: 2 });
});

void test("reasoners are disabled by default", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-runtime-defaults-"));
	const resolved = resolveRuntimeConfig(cwd, {});
	assert.equal(resolved.bridge.reasoner, "none");
	assert.equal(resolved.bridge.memoryEmbeddingsEnabled, false);
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
	assert.equal(resolved.bridge.permissionMode, "acceptEdits");
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
		}),
		"utf8",
	);
	writeFileSync(
		path.join(workspace, ".logician.json"),
		JSON.stringify({
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
	assert.equal(resolved.bridge.permissionMode, "acceptEdits");
	assert.deepEqual(resolved.source.mcpServers, {
		project: { command: "project-mcp" },
	});
	assert.equal(resolved.bridge.projectTrusted, true);
});
