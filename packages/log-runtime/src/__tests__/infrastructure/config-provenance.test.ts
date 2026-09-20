import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	buildConfigProvenance,
	formatConfigProvenance,
} from "../../runtime/configuration/config-provenance.ts";

void test("provenance gives each key to its highest-precedence layer", () => {
	const table = buildConfigProvenance({
		global: { model: "global-model", theme: "dark" },
		project: { model: "project-model", compaction: { enabled: true } },
		env: { model: "env-model" },
	});
	const byKey = new Map(table.map(e => [e.key, e]));
	assert.deepEqual(byKey.get("model"), {
		key: "model",
		value: "env-model",
		layer: "env",
	});
	assert.deepEqual(byKey.get("theme"), {
		key: "theme",
		value: "dark",
		layer: "global",
	});
	assert.deepEqual(byKey.get("compaction.enabled"), {
		key: "compaction.enabled",
		value: true,
		layer: "project",
	});
	// Shadowed values never surface.
	assert.equal(table.filter(e => e.value === "global-model").length, 0);
	assert.equal(table.filter(e => e.value === "project-model").length, 0);
});

void test("provenance flattens nested sections and keeps passthroughs whole", () => {
	const table = buildConfigProvenance({
		project: {
			lsp: { enabled: true, serverOverrides: { ts: { command: "tsserver" } } },
			mcpServers: { srv: { url: "http://mcp.test" } },
			models: [{ name: "fast", model: "m" }],
		},
	});
	const byKey = new Map(table.map(e => [e.key, e]));
	assert.equal(byKey.get("lsp.enabled")?.value, true);
	assert.equal(byKey.get("lsp.serverOverrides.ts.command")?.value, "tsserver");
	// Unconstrained passthrough objects stay at their top-level key.
	assert.deepEqual(byKey.get("mcpServers")?.value, {
		srv: { url: "http://mcp.test" },
	});
	assert.deepEqual(byKey.get("models")?.value, [{ name: "fast", model: "m" }]);
});

void test("provenance omits layers that were not present", () => {
	const table = buildConfigProvenance({
		project: { hooks: true },
	});
	assert.deepEqual(table, [{ key: "hooks", value: true, layer: "project" }]);
});

void test("provenance rows are sorted by key", () => {
	const table = buildConfigProvenance({
		global: { zeta: 1, alpha: 2 },
	});
	assert.deepEqual(
		table.map(e => e.key),
		["alpha", "zeta"],
	);
});
void test("formatConfigProvenance renders key, value, and layer", () => {
	const table = buildConfigProvenance({
		project: { model: "m", compaction: { enabled: true } },
		env: { theme: "light" },
	});
	const text = formatConfigProvenance(table);
	assert.match(text, /model\s+"m" {2}\[project\]/);
	assert.match(text, /compaction\.enabled\s+true {2}\[project\]/);
	assert.match(text, /theme\s+"light" {2}\[env\]/);
});

void test("formatConfigProvenance marks an empty table", () => {
	assert.equal(formatConfigProvenance([]), "  (no keys set — defaults only)");
});
