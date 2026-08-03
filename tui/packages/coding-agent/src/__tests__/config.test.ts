import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import {
	validateConfig,
	loadLogicianConfig,
	findLogicianConfig,
	configString,
	configNumber,
	configBool,
	saveConfigField,
} from "../configuration/config.ts";

// ── validateConfig ───────────────────────────────────────────────────────

void test("validateConfig rejects a non-object root", () => {
	const warnings: string[] = [];
	const cfg = validateConfig("not an object", warnings);
	assert.deepEqual(cfg, {});
	assert.equal(warnings.length, 1);
	assert.match(warnings[0], /not an object/);
});

void test("validateConfig rejects null", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(null, warnings);
	assert.deepEqual(cfg, {});
	assert.equal(warnings.length, 1);
});

void test("validateConfig warns on unknown top-level keys but keeps known ones", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({ model: "gpt", bogusKey: 1 }, warnings);
	assert.equal(cfg.model, "gpt");
	assert.ok(warnings.some((w) => w.includes("Unknown config key: \"bogusKey\"")));
});

void test("validateConfig accepts valid baseUrl and llmUrl", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{ baseUrl: "http://localhost:8000", llmUrl: "https://api.test/v1" },
		warnings,
	);
	assert.equal(cfg.baseUrl, "http://localhost:8000");
	assert.equal(cfg.llmUrl, "https://api.test/v1");
	assert.equal(warnings.length, 0);
});

void test("validateConfig rejects an invalid baseUrl", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({ baseUrl: "not-a-url" }, warnings);
	assert.equal(cfg.baseUrl, undefined);
	assert.ok(warnings.some((w) => w.includes("\"baseUrl\" must be a valid http/https URL")));
});

void test("validateConfig parses legacy string entries and object entries in models", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{
			models: [
				"legacy-model",
				{ name: "Named", model: "named-model", url: " http://x " },
				{ name: "", model: "" },
				42,
			],
		},
		warnings,
	);
	assert.deepEqual(cfg.models, [
		{ name: "legacy-model", model: "legacy-model" },
		{ name: "Named", model: "named-model", url: "http://x" },
	]);
	assert.ok(warnings.some((w) => w.includes("\"models\" entry invalid")));
});

void test("validateConfig clamps out-of-range temperature and warns", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({ temperature: 5 }, warnings);
	assert.equal(cfg.temperature, 2);
	assert.ok(warnings.some((w) => w.includes("\"temperature\" out of range")));
});

void test("validateConfig accepts in-range temperature without warning", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({ temperature: 0.7 }, warnings);
	assert.equal(cfg.temperature, 0.7);
	assert.equal(warnings.length, 0);
});

void test("validateConfig ignores non-positive maxTokens/maxIterations/maxTotalTokens", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{ maxTokens: 0, maxIterations: -1, maxTotalTokens: 0 },
		warnings,
	);
	assert.equal(cfg.maxTokens, undefined);
	assert.equal(cfg.maxIterations, undefined);
	assert.equal(cfg.maxTotalTokens, undefined);
	assert.equal(warnings.length, 3);
});

void test("validateConfig rejects invalid toolExecution enum", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({ toolExecution: "bogus" }, warnings);
	assert.equal(cfg.toolExecution, undefined);
	assert.ok(warnings.some((w) => w.includes("\"toolExecution\" must be")));
});

void test("validateConfig accepts valid toolExecution and permissionMode", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{ toolExecution: "parallel", permissionMode: "plan" },
		warnings,
	);
	assert.equal(cfg.toolExecution, "parallel");
	assert.equal(cfg.permissionMode, "plan");
	assert.equal(warnings.length, 0);
});

void test("validateConfig rejects invalid inferenceMode", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({ inferenceMode: "bogus-mode" }, warnings);
	assert.equal(cfg.inferenceMode, undefined);
	assert.ok(warnings.some((w) => w.includes("\"inferenceMode\" must be one of")));
});

void test("validateConfig accepts every inference preset", () => {
	for (const inferenceMode of [
		"auto",
		"thinking-general",
		"thinking-coding",
		"instruct-general",
		"instruct-reasoning",
		"instruct-coding",
		"deterministic",
		"creative",
		"analytical",
	] as const) {
		const warnings: string[] = [];
		const cfg = validateConfig({ inferenceMode }, warnings);
		assert.equal(cfg.inferenceMode, inferenceMode);
		assert.deepEqual(warnings, []);
	}
});

void test("validateConfig accepts execution profiles and rejects unknown profiles", () => {
	const validWarnings: string[] = [];
	const valid = validateConfig({ executionProfile: "minimal" }, validWarnings);
	assert.equal(valid.executionProfile, "minimal");
	assert.deepEqual(validWarnings, []);

	const invalidWarnings: string[] = [];
	const invalid = validateConfig(
		{ executionProfile: "maximal" },
		invalidWarnings,
	);
	assert.equal(invalid.executionProfile, undefined);
	assert.ok(
		invalidWarnings.some((warning) =>
			warning.includes("\"executionProfile\" must be one of")),
	);
});

void test("validateConfig applies default booleans (continuationEnabled, postEditDiagnostics, autoRetryEnabled ON by default)", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({}, warnings);
	assert.equal(cfg.continuationEnabled, true);
	assert.equal(cfg.postEditDiagnostics, true);
	assert.equal(cfg.autoRetryEnabled, true);
	assert.equal(cfg.guardsEnabled, undefined);
});

void test("validateConfig enforces bounds for maxRetries/retryBaseDelayMs/turnTimeoutMs/cacheSize/cacheTtlMs", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{
			maxRetries: -1,
			retryBaseDelayMs: -1,
			turnTimeoutMs: 0,
			cacheSize: 0,
			cacheTtlMs: 0,
		},
		warnings,
	);
	assert.equal(cfg.maxRetries, undefined);
	assert.equal(cfg.retryBaseDelayMs, undefined);
	assert.equal(cfg.turnTimeoutMs, undefined);
	assert.equal(cfg.cacheSize, undefined);
	assert.equal(cfg.cacheTtlMs, undefined);
	assert.equal(warnings.length, 5);
});

void test("validateConfig accepts maxRetries=0 and retryBaseDelayMs=0 as inclusive bounds", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({ maxRetries: 0, retryBaseDelayMs: 0 }, warnings);
	assert.equal(cfg.maxRetries, 0);
	assert.equal(cfg.retryBaseDelayMs, 0);
	assert.equal(warnings.length, 0);
});

void test("validateConfig rejects relative allowedPaths but keeps absolute ones", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{ allowedPaths: ["/abs/path", "relative/path"] },
		warnings,
	);
	assert.deepEqual(cfg.allowedPaths, ["/abs/path"]);
	assert.ok(warnings.some((w) => w.includes("must be an absolute path")));
});

void test("validateConfig rejects a cwd that does not exist", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({ cwd: "/definitely/does/not/exist/xyz" }, warnings);
	assert.equal(cfg.cwd, undefined);
	assert.ok(warnings.some((w) => w.includes("\"cwd\" path does not exist")));
});

void test("validateConfig accepts an existing cwd, resolved to an absolute path", () => {
	const warnings: string[] = [];
	const cfg = validateConfig({ cwd: tmpdir() }, warnings);
	assert.equal(cfg.cwd, path.resolve(tmpdir()));
	assert.equal(warnings.length, 0);
});

void test("validateConfig parses lsp sub-object and warns on unknown lsp keys", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{
			lsp: {
				enabled: true,
				timeoutMs: 5000,
				bogus: 1,
				serverOverrides: {
					ts: { command: "tsserver", args: ["--stdio"], languageId: "typescript" },
					bad: { command: "" },
				},
			},
		},
		warnings,
	);
	assert.equal(cfg.lsp?.enabled, true);
	assert.equal(cfg.lsp?.timeoutMs, 5000);
	assert.deepEqual(cfg.lsp?.serverOverrides?.ts, {
		command: "tsserver",
		args: ["--stdio"],
		languageId: "typescript",
	});
	assert.equal(cfg.lsp?.serverOverrides?.bad, undefined);
	assert.ok(warnings.some((w) => w.includes("Unknown lsp key: \"bogus\"")));
	assert.ok(warnings.some((w) => w.includes("command")));
});

void test("validateConfig parses compaction sub-object with positive-only numeric fields", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{ compaction: { enabled: true, reserveTokens: 0, keepRecentTokens: 100 } },
		warnings,
	);
	assert.equal(cfg.compaction?.enabled, true);
	assert.equal(cfg.compaction?.reserveTokens, undefined);
	assert.equal(cfg.compaction?.keepRecentTokens, 100);
});

void test("validateConfig parses truncation sub-object including nested microCompactMaxChars", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{
			truncation: {
				toolResultMaxChars: 500,
				maxLines: -1,
				microCompactMaxChars: { tool: 100, assistant: 0, bogus: 5 },
			},
		},
		warnings,
	);
	assert.equal(cfg.truncation?.toolResultMaxChars, 500);
	assert.equal(cfg.truncation?.maxLines, undefined);
	assert.equal(cfg.truncation?.microCompactMaxChars?.tool, 100);
	assert.equal(cfg.truncation?.microCompactMaxChars?.assistant, undefined);
	assert.ok(warnings.some((w) => w.includes("\"truncation.maxLines\" must be > 0")));
	assert.ok(warnings.some((w) => w.includes("Unknown truncation.microCompactMaxChars key")));
});

void test("validateConfig passes through mcp/mcpServers objects and rejects non-object plugins", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{
			mcp: { foo: "bar" },
			mcpServers: { srv: {} },
			plugins: ["not", "an", "object"],
		},
		warnings,
	);
	assert.deepEqual(cfg.mcp, { foo: "bar" });
	assert.deepEqual(cfg.mcpServers, { srv: {} });
	assert.equal(cfg.plugins, undefined);
	assert.ok(warnings.some((w) => w.includes("\"plugins\" must be an object")));
});

void test("validateConfig validates webSearch.baseUrl and clamps maxResults range", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{ webSearch: { baseUrl: "not-a-url", maxResults: 500 } },
		warnings,
	);
	assert.equal(cfg.webSearch?.baseUrl, undefined);
	assert.equal(cfg.webSearch?.maxResults, undefined);
	assert.ok(warnings.some((w) => w.includes("\"webSearch.baseUrl\" must be a valid")));
	assert.ok(warnings.some((w) => w.includes("\"webSearch.maxResults\" must be 1")));
});

void test("validateConfig filters non-string entries from permissions.allow/deny", () => {
	const warnings: string[] = [];
	const cfg = validateConfig(
		{ permissions: { allow: ["bash", "", 42, "  "], deny: ["rm"] } },
		warnings,
	);
	assert.deepEqual(cfg.permissions?.allow, ["bash"]);
	assert.deepEqual(cfg.permissions?.deny, ["rm"]);
});

// ── configString / configNumber / configBool ────────────────────────────

void test("configString trims and falls back on non-string or blank input", () => {
	assert.equal(configString("  hi  "), "hi");
	assert.equal(configString(""), undefined);
	assert.equal(configString(42), undefined);
	assert.equal(configString(undefined, "fallback"), "fallback");
});

void test("configNumber coerces numeric strings and rejects non-finite values", () => {
	assert.equal(configNumber(42), 42);
	assert.equal(configNumber("42"), 42);
	assert.equal(configNumber("not-a-number"), undefined);
	assert.equal(configNumber(NaN), undefined);
	assert.equal(configNumber(undefined, 7), 7);
});

void test("configBool parses common truthy/falsy string variants", () => {
	assert.equal(configBool(true), true);
	assert.equal(configBool("yes"), true);
	assert.equal(configBool("OFF"), false);
	assert.equal(configBool("maybe"), undefined);
	assert.equal(configBool(undefined, false), false);
});

// ── findLogicianConfig / loadLogicianConfig ──────────────────────────────

function mkWorkspace(): string {
	return mkdtempSync(path.join(tmpdir(), "logician-config-"));
}

void test("findLogicianConfig finds .logician.json by walking up from cwd", () => {
	const root = mkWorkspace();
	const nested = path.join(root, "a", "b");
	mkdirSync(nested, { recursive: true });
	writeFileSync(path.join(root, ".logician.json"), "{}", "utf8");
	try {
		const found = findLogicianConfig(nested);
		assert.equal(found, path.join(root, ".logician.json"));
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
});

void test("findLogicianConfig returns null when no config exists anywhere up to root", () => {
	const root = mkWorkspace();
	const isolatedHome = mkWorkspace();
	const prevEnv = process.env.LOGICIAN_CONFIG;
	const prevHome = process.env.HOME;
	delete process.env.LOGICIAN_CONFIG;
	process.env.HOME = isolatedHome; // avoid picking up the real ~/.logician/settings.json fallback
	try {
		const found = findLogicianConfig(root);
		assert.equal(found, null);
	} finally {
		if (prevEnv === undefined) delete process.env.LOGICIAN_CONFIG;
		else process.env.LOGICIAN_CONFIG = prevEnv;
		if (prevHome === undefined) delete process.env.HOME;
		else process.env.HOME = prevHome;
		rmSync(root, { recursive: true, force: true });
		rmSync(isolatedHome, { recursive: true, force: true });
	}
});

void test("findLogicianConfig honors LOGICIAN_CONFIG env override when the path exists", () => {
	const root = mkWorkspace();
	const explicit = path.join(root, "custom.json");
	writeFileSync(explicit, "{}", "utf8");
	const prev = process.env.LOGICIAN_CONFIG;
	process.env.LOGICIAN_CONFIG = explicit;
	try {
		assert.equal(findLogicianConfig(root), explicit);
	} finally {
		if (prev === undefined) delete process.env.LOGICIAN_CONFIG;
		else process.env.LOGICIAN_CONFIG = prev;
		rmSync(root, { recursive: true, force: true });
	}
});

void test("findLogicianConfig returns null when LOGICIAN_CONFIG points to a missing file", () => {
	const prev = process.env.LOGICIAN_CONFIG;
	process.env.LOGICIAN_CONFIG = "/definitely/does/not/exist/config.json";
	try {
		assert.equal(findLogicianConfig(), null);
	} finally {
		if (prev === undefined) delete process.env.LOGICIAN_CONFIG;
		else process.env.LOGICIAN_CONFIG = prev;
	}
});

void test("loadLogicianConfig returns empty config with no warnings when no config file is found", () => {
	const root = mkWorkspace();
	const isolatedHome = mkWorkspace();
	const prevEnv = process.env.LOGICIAN_CONFIG;
	const prevHome = process.env.HOME;
	delete process.env.LOGICIAN_CONFIG;
	process.env.HOME = isolatedHome; // avoid picking up the real ~/.logician/settings.json fallback
	try {
		const result = loadLogicianConfig(root);
		assert.deepEqual(result.config, {});
		assert.deepEqual(result.warnings, []);
		assert.equal(result.path, undefined);
	} finally {
		if (prevEnv === undefined) delete process.env.LOGICIAN_CONFIG;
		else process.env.LOGICIAN_CONFIG = prevEnv;
		if (prevHome === undefined) delete process.env.HOME;
		else process.env.HOME = prevHome;
		rmSync(root, { recursive: true, force: true });
		rmSync(isolatedHome, { recursive: true, force: true });
	}
});

void test("loadLogicianConfig loads and validates a project .logician.json", () => {
	const root = mkWorkspace();
	writeFileSync(
		path.join(root, ".logician.json"),
		JSON.stringify({ model: "test-model", temperature: 0.5 }),
		"utf8",
	);
	try {
		const result = loadLogicianConfig(root);
		assert.equal(result.config.model, "test-model");
		assert.equal(result.config.temperature, 0.5);
		assert.equal(result.path, path.join(root, ".logician.json"));
		assert.deepEqual(result.warnings, []);
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
});

void test("loadLogicianConfig throws with a descriptive message on malformed JSON", () => {
	const root = mkWorkspace();
	writeFileSync(path.join(root, ".logician.json"), "{ not valid json", "utf8");
	try {
		assert.throws(
			() => loadLogicianConfig(root),
			/Failed to read .*\.logician\.json/,
		);
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
});

// ── saveConfigField ───────────────────────────────────────────────────────

void test("saveConfigField writes a field to ~/.logician/settings.json, creating dirs as needed", () => {
	const fakeHome = mkWorkspace();
	const prevHome = process.env.HOME;
	process.env.HOME = fakeHome;
	try {
		const ok = saveConfigField("theme", "dark");
		assert.equal(ok, true);
		const written = JSON.parse(
			readFileSync(
				path.join(fakeHome, ".logician", "settings.json"),
				"utf8",
			),
		);
		assert.equal(written.theme, "dark");
	} finally {
		if (prevHome === undefined) delete process.env.HOME;
		else process.env.HOME = prevHome;
		rmSync(fakeHome, { recursive: true, force: true });
	}
});

void test("saveConfigField merges into an existing settings file instead of overwriting it", () => {
	const fakeHome = mkWorkspace();
	const settingsDir = path.join(fakeHome, ".logician");
	mkdirSync(settingsDir, { recursive: true });
	writeFileSync(
		path.join(settingsDir, "settings.json"),
		JSON.stringify({ existing: "value" }),
		"utf8",
	);
	const prevHome = process.env.HOME;
	process.env.HOME = fakeHome;
	try {
		saveConfigField("theme", "light");
		const written = JSON.parse(
			readFileSync(
				path.join(settingsDir, "settings.json"),
				"utf8",
			),
		);
		assert.equal(written.existing, "value");
		assert.equal(written.theme, "light");
	} finally {
		if (prevHome === undefined) delete process.env.HOME;
		else process.env.HOME = prevHome;
		rmSync(fakeHome, { recursive: true, force: true });
	}
});

void test("saveConfigField returns false when HOME is unset", () => {
	const prevHome = process.env.HOME;
	delete process.env.HOME;
	try {
		const ok = saveConfigField("theme", "dark");
		assert.equal(ok, false);
	} finally {
		if (prevHome === undefined) delete process.env.HOME;
		else process.env.HOME = prevHome;
	}
});
