import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { TsPluginManager } from "../compatibility/claude-code/plugin-manager.ts";

function setupEnv(): { home: string; restore: () => void } {
	const home = mkdtempSync(join(tmpdir(), "logician-plug-"));
	const pluginsDir = join(home, ".claude", "plugins");
	const installPath = join(pluginsDir, "cache", "m", "demo", "1.0.0");
	mkdirSync(installPath, { recursive: true });
	writeFileSync(
		join(pluginsDir, "installed_plugins.json"),
		JSON.stringify({
			version: 2,
			plugins: {
				"demo@m": [
					{
						scope: "user",
						installPath,
						version: "1.0.0",
						installedAt: "2026-01-01T00:00:00Z",
						lastUpdated: "2026-01-01T00:00:00Z",
					},
				],
			},
		}),
	);

	const prevHome = process.env.HOME;
	const prevCache = process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR;
	process.env.HOME = home;
	process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR = pluginsDir;
	return {
		home,
		restore: () => {
			process.env.HOME = prevHome;
			if (prevCache === undefined)
				delete process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR;
			else process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR = prevCache;
		},
	};
}

void test("plugin enablement falls back to Claude Code enabledPlugins", async () => {
	const { home, restore } = setupEnv();
	try {
		// Claude Code disables the plugin; logician has no opinion.
		mkdirSync(join(home, ".claude"), { recursive: true });
		writeFileSync(
			join(home, ".claude", "settings.json"),
			JSON.stringify({ enabledPlugins: { "demo@m": false } }),
		);

		const manager = new TsPluginManager();
		const result = await manager.listPlugins();
		const row = (result.plugins ?? []).find((p) => p.plugin_id === "demo@m");
		assert.ok(row);
		assert.equal(row.enabled, false);
	} finally {
		restore();
	}
});

void test("logician settings override Claude Code enablement", async () => {
	const { home, restore } = setupEnv();
	try {
		mkdirSync(join(home, ".claude"), { recursive: true });
		writeFileSync(
			join(home, ".claude", "settings.json"),
			JSON.stringify({ enabledPlugins: { "demo@m": false } }),
		);
		mkdirSync(join(home, ".logician"), { recursive: true });
		writeFileSync(
			join(home, ".logician", "settings.json"),
			JSON.stringify({ plugins: { "demo@m": { enabled: true } } }),
		);

		const manager = new TsPluginManager();
		const result = await manager.listPlugins();
		const row = (result.plugins ?? []).find((p) => p.plugin_id === "demo@m");
		assert.ok(row);
		assert.equal(row.enabled, true);
	} finally {
		restore();
	}
});

void test("plugins default to enabled when neither settings file has an entry", async () => {
	const { restore } = setupEnv();
	try {
		const manager = new TsPluginManager();
		const result = await manager.listPlugins();
		const row = (result.plugins ?? []).find((p) => p.plugin_id === "demo@m");
		assert.ok(row);
		assert.equal(row.enabled, true);
	} finally {
		restore();
	}
});
