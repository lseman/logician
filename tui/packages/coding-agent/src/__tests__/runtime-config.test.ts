import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { resolveRuntimeConfig } from "../runtime/runtime-config.ts";

function configuredWorkspace(): string {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-runtime-config-"));
	writeFileSync(
		path.join(cwd, ".logician.json"),
		JSON.stringify({
			baseUrl: "http://config.test:8000",
			model: "config-model",
			permissionMode: "ask",
			toolExecution: "sequential",
			mcpEager: true,
			hooks: true,
		}),
		"utf8",
	);
	return cwd;
}

void test("runtime resolver applies shared environment precedence", () => {
	const resolved = resolveRuntimeConfig(
		configuredWorkspace(),
		{
			LOGICIAN_LLM_URL: "http://env.test:9000",
			LOGICIAN_MODEL: "env-model",
			LOGICIAN_HOOKS: "0",
		},
	);

	assert.equal(resolved.bridge.baseUrl, "http://env.test:9000");
	assert.equal(resolved.bridge.model, "env-model");
	assert.equal(resolved.bridge.runtimeHooksEnabled, false);
	assert.equal(resolved.bridge.toolExecution, "sequential");
	assert.equal(resolved.bridge.permissionMode, "ask");
	assert.equal(resolved.bridge.mcpEager, true);
});
