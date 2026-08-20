import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { LspManager } from "../../infrastructure/developer-tools/lsp-manager.ts";

void test("LspManager lazily collects publishDiagnostics", async () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-lsp-"));
	const target = path.join(cwd, "sample.fake");
	writeFileSync(target, "hello\nworld\n", "utf8");
	const fixture = fileURLToPath(
		new URL("../fixtures/fake-lsp.mjs", import.meta.url),
	);
	const manager = new LspManager(cwd, {
		timeoutMs: 1_000,
		servers: {
			".fake": {
				command: process.execPath,
				args: [fixture],
				languageId: "fake",
			},
		},
	});
	try {
		const diagnostics = await manager.diagnosticsFor(target);
		assert.equal(diagnostics?.[0]?.line, 2);
		assert.equal(diagnostics?.[0]?.column, 3);
		assert.equal(diagnostics?.[0]?.source, "fake-lsp");
		assert.equal(diagnostics?.[0]?.code, "fake-1");
	} finally {
		manager.close();
	}
});

void test("LspManager treats a missing server as an unavailable fallback", async () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-lsp-missing-"));
	const target = path.join(cwd, "sample.missing");
	writeFileSync(target, "content", "utf8");
	const manager = new LspManager(cwd, {
		timeoutMs: 100,
		servers: {
			".missing": {
				command: "logician-language-server-that-does-not-exist",
				args: [],
				languageId: "missing",
			},
		},
	});
	try {
		assert.equal(await manager.diagnosticsFor(target), null);
	} finally {
		manager.close();
	}
});
