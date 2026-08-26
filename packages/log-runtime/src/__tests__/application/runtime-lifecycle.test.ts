import { describe, expect, test } from "bun:test";
import {
	RuntimeLifecycle,
	type RuntimeLifecycleDependencies,
} from "../../runtime/bridge/application/runtime-lifecycle.ts";

function createLifecycle() {
	const calls: string[] = [];
	const record = (name: string) => () => {
		calls.push(name);
	};
	const recordAsync = (name: string) => async () => {
		calls.push(name);
	};
	const dependencies: RuntimeLifecycleDependencies = {
		cancel: recordAsync("cancel"),
		resetTurns: record("reset-turns"),
		dropSession: record("drop-session"),
		clearSession: record("clear-session"),
		resetIdentity: record("reset-identity"),
		endPluginSession: async reason => {
			calls.push(`end:${reason}`);
		},
		resetPlugin: options => calls.push(`reset-plugin:${options?.clearResult}`),
		refreshPluginContext: record("refresh-context"),
		resetInjectedContext: record("reset-injected"),
		resetDiscoveredResources: record("reset-resources"),
		injectSkills: recordAsync("skills"),
		injectPrompts: recordAsync("prompts"),
		reloadExtensions: recordAsync("extensions-reload"),
		reportExtensionError: record("extension-error"),
		extensionsReady: recordAsync("extensions-ready"),
		ensurePluginsStarted: recordAsync("plugins-started"),
		ensureSession: record("ensure-session"),
		loadMcp: recordAsync("mcp"),
		reportMcpError: record("mcp-error"),
		waitForMemory: recordAsync("memory-settled"),
		closeResources: recordAsync("resources-closed"),
		resetActivity: record("reset-activity"),
		publishUsage: record("publish-usage"),
		emitTurnEnd: id => calls.push(`turn-end:${id}`),
	};
	return { lifecycle: new RuntimeLifecycle(dependencies), calls };
}

describe("RuntimeLifecycle", () => {
	test("reload preserves the required teardown and discovery order", async () => {
		const state = createLifecycle();
		await state.lifecycle.reload();
		expect(state.calls).toEqual([
			"cancel",
			"reset-turns",
			"drop-session",
			"reset-identity",
			"reset-resources",
			"reset-plugin:true",
			"skills",
			"prompts",
			"extensions-reload",
			"mcp",
			"turn-end:reload",
		]);
	});

	test("shutdown settles memory and hooks before closing resources", async () => {
		const state = createLifecycle();
		await state.lifecycle.stop();
		expect(state.calls).toEqual([
			"cancel",
			"memory-settled",
			"end:shutdown",
			"resources-closed",
			"reset-turns",
		]);
	});
});
