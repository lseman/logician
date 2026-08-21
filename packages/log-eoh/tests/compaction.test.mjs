import assert from "node:assert";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import {
	buildEohCompactionSummary,
	eohSummaryPathsFor,
} from "../src/compaction.ts";

test("buildEohCompactionSummary includes header", async () => {
	const tmpDir = await mkdtemp(path.join(os.tmpdir(), "logician-eoh-test-"));
	try {
		await mkdir(path.join(tmpDir, ".eoh"), { recursive: true });
		await writeFile(
			path.join(tmpDir, ".eoh", "log.jsonl"),
			'{"type": "eoh_config", "name": "test"}\n{"run": 1, "fitness": 0.5, "generation": 1, "createdBy": "init", "status": "keep", "description": "first", "timestamp": 1000}\n',
		);
		const paths = eohSummaryPathsFor(tmpDir);
		const summary = buildEohCompactionSummary(paths);
		assert.ok(summary.includes("EoH Compaction Summary"));
		assert.ok(summary.includes("test"));
		assert.ok(summary.includes("0.5"));
	} finally {
		await rm(tmpDir, { recursive: true, force: true });
	}
});

test("buildEohCompactionSummary handles empty log", async () => {
	const tmpDir = await mkdtemp(path.join(os.tmpdir(), "logician-eoh-test-"));
	try {
		await mkdir(path.join(tmpDir, ".eoh"), { recursive: true });
		await writeFile(path.join(tmpDir, ".eoh", "log.jsonl"), "");
		const paths = eohSummaryPathsFor(tmpDir);
		const summary = buildEohCompactionSummary(paths);
		assert.ok(summary.includes("EoH Compaction Summary"));
		assert.ok(summary.includes("No runs yet"));
	} finally {
		await rm(tmpDir, { recursive: true, force: true });
	}
});

test("buildEohCompactionSummary includes problem section", async () => {
	const tmpDir = await mkdtemp(path.join(os.tmpdir(), "logician-eoh-test-"));
	try {
		await mkdir(path.join(tmpDir, ".eoh"), { recursive: true });
		await writeFile(
			path.join(tmpDir, ".eoh", "log.jsonl"),
			'{"type": "eoh_config", "name": "test"}\n{"run": 1, "fitness": 0.5, "generation": 1, "createdBy": "init", "status": "keep", "description": "first", "timestamp": 1000}\n',
		);
		await writeFile(
			path.join(tmpDir, ".eoh", "problem.json"),
			"Bin packing problem",
		);
		const paths = eohSummaryPathsFor(tmpDir);
		const summary = buildEohCompactionSummary(paths);
		assert.ok(summary.includes("Bin packing problem"));
	} finally {
		await rm(tmpDir, { recursive: true, force: true });
	}
});
