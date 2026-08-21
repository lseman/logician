import assert from "node:assert/strict";
import * as fs from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { test } from "node:test";

import { AutoresearchSession } from "../src/index.ts";

function freshDir() {
	return fs.mkdtempSync(path.join(tmpdir(), "logician-autoresearch-run-"));
}

async function initializedSession(dir) {
	const session = new AutoresearchSession(dir);
	await session.initExperiment({ name: "checks", metric_name: "score" });
	return session;
}

test("run_experiment executes checks.sh and rejects a failed check", async () => {
	const dir = freshDir();
	const session = await initializedSession(dir);
	fs.writeFileSync(
		path.join(dir, ".auto", "checks.sh"),
		"#!/usr/bin/env bash\necho check-failed >&2\nexit 7\n",
	);

	const result = await session.runExperiment({
		command: "echo METRIC score=1",
	});

	assert.equal(result.details?.checksPass, false);
	assert.equal(result.details?.passed, false);
	assert.match(String(result.details?.checksOutput), /check-failed/);
	assert.match(result.content, /CHECKS FAILED/);
});

test("run_experiment terminates checks that exceed their deadline", async () => {
	const dir = freshDir();
	const session = await initializedSession(dir);
	fs.writeFileSync(
		path.join(dir, ".auto", "checks.sh"),
		"#!/usr/bin/env bash\ntrap '' TERM\nsleep 5\n",
	);

	const started = Date.now();
	const result = await session.runExperiment({
		command: "echo METRIC score=1",
		checks_timeout_seconds: 0.02,
	});

	assert.equal(result.details?.checksPass, false);
	assert.equal(result.details?.passed, false);
	assert.ok(Date.now() - started < 2000);
});
