import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
	getTrustRequiringPaths,
	hasTrustRequiringProjectResources,
} from "../trust/checker.ts";

void test("global home skills do not stall or require project trust", () => {
	const cwd = mkdtempSync(join(homedir(), ".logician-trust-test-"));
	try {
		assert.equal(hasTrustRequiringProjectResources(cwd), false);
	} finally {
		rmSync(cwd, { recursive: true, force: true });
	}
});

void test("project-local agent skills require trust", () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-trust-test-"));
	const skills = join(cwd, ".agents", "skills");
	mkdirSync(skills, { recursive: true });
	try {
		assert.equal(hasTrustRequiringProjectResources(cwd), true);
		assert.ok(getTrustRequiringPaths(cwd).includes(skills));
	} finally {
		rmSync(cwd, { recursive: true, force: true });
	}
});
