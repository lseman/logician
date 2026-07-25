import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import {
	buildDoctorReport,
	formatDoctorReport,
} from "../runtime/doctor.ts";

void test("doctor reports configuration without modifying it", async () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-doctor-"));
	const configPath = path.join(cwd, ".logician.json");
	const original = JSON.stringify({
		baseUrl: "http://localhost:9000",
		model: "test-model",
		permissionMode: "ask",
		mcpServers: { docs: { url: "http://localhost:3000" } },
	});
	writeFileSync(configPath, original, "utf8");

	const report = await buildDoctorReport(cwd);

	assert.equal(report.config.valid, true);
	assert.equal(report.config.path, configPath);
	assert.equal(report.backend.baseUrl, "http://localhost:9000");
	assert.equal(report.backend.model, "test-model");
	assert.equal(report.permissions.mode, "ask");
	assert.equal(report.mcp.configured, 1);
	assert.equal(report.mcp.liveHealthChecked, false);
	assert.equal(report.sandbox.enforced, false);
	assert.equal(readFileSync(configPath, "utf8"), original);
});

void test("doctor returns structured invalid-config evidence", async () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-doctor-invalid-"));
	writeFileSync(path.join(cwd, ".logician.json"), "{invalid", "utf8");

	const report = await buildDoctorReport(cwd);

	assert.equal(report.config.valid, false);
	assert.match(report.config.error ?? "", /Failed to read/);
	assert.doesNotThrow(() => JSON.stringify(report));
});

void test("doctor text states that backend and sandbox are not verified", async () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-doctor-text-"));
	const text = formatDoctorReport(await buildDoctorReport(cwd));

	assert.match(text, /not probed/);
	// sandbox line shows either "none" or "bubblewrap" depending on platform
	assert.match(text, /sandbox:\s*(none|bubblewrap)/);
});
