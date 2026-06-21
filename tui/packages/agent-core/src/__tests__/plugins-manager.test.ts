import assert from "node:assert/strict";
import {
	mkdirSync,
	mkdtempSync,
	readFileSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { TsPluginManager } from "../tools/shared/plugins-manager.ts";

function writePluginSource(root: string, version: string): string {
	const pluginDir = join(root, "demo-plugin");
	mkdirSync(join(pluginDir, ".claude-plugin"), { recursive: true });
	writeFileSync(
		join(pluginDir, ".claude-plugin", "plugin.json"),
		JSON.stringify({ name: "demo-plugin", version }),
		"utf8",
	);
	return pluginDir;
}

function readSettings(home: string): Record<string, unknown> {
	return JSON.parse(
		readFileSync(join(home, ".logician", "settings.json"), "utf8"),
	) as Record<string, unknown>;
}

void test("plugin enabled state is persisted in user settings and survives reinstall", async () => {
	const root = mkdtempSync(join(tmpdir(), "logician-plugins-"));
	const oldHome = process.env.HOME;
	const oldPluginCache = process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR;
	process.env.HOME = join(root, "home");
	process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR = join(root, "plugins");

	try {
		const pluginDir = writePluginSource(root, "1.0.0");
		const manager = new TsPluginManager();

		await manager.install(pluginDir);
		assert.deepEqual(readSettings(process.env.HOME).plugins, {
			"demo-plugin@local": { enabled: true },
		});

		await manager.setEnabled("demo-plugin", false);
		assert.deepEqual(readSettings(process.env.HOME).plugins, {
			"demo-plugin@local": { enabled: false },
		});

		writePluginSource(root, "1.0.1");
		await manager.install(pluginDir);

		const listing = await manager.listPlugins();
		assert.equal(listing.plugins?.[0]?.enabled, false);
		assert.deepEqual(readSettings(process.env.HOME).plugins, {
			"demo-plugin@local": { enabled: false },
		});
	} finally {
		if (oldHome === undefined) delete process.env.HOME;
		else process.env.HOME = oldHome;
		if (oldPluginCache === undefined) delete process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR;
		else process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR = oldPluginCache;
	}
});
