import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { test } from "node:test";
import {
	ensureInsideCwd,
	resolvePath,
	resolveReadPath,
} from "../tools/shared/path-utils.ts";

void test("resolvePath normalizes pasted agent paths", () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-path-"));
	assert.equal(
		resolvePath(cwd, "@dir\u202Fname/file.ts"),
		resolve(cwd, "dir name/file.ts"),
	);
});

void test("ensureInsideCwd rejects sibling directories with shared prefixes", () => {
	const cwd = resolve(tmpdir(), "logician-path-root");
	const sibling = `${cwd}-sibling/file.ts`;
	assert.throws(() => ensureInsideCwd(cwd, sibling), /outside CWD/);
});

void test("resolveReadPath falls back to macOS-style filename variants", () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-path-"));
	mkdirSync(join(cwd, "screenshots"));
	const actual = join(
		cwd,
		"screenshots",
		"Capture d\u2019ecran 10.15.30\u202FAM.png",
	);
	writeFileSync(actual, "image-ish", "utf8");

	const requested = "screenshots/Capture d'ecran 10.15.30 AM.png";
	assert.equal(resolveReadPath(requested, cwd), actual);
});
