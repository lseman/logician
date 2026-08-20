import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
	getProjectVirtualEnv,
	getShellEnv,
	getVirtualEnvPythonVersion,
} from "../infrastructure/tools/utils/shell.ts";

void test("project .venv is activated in shell environments", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-venv-"));
	const virtualEnv = path.join(cwd, ".venv");
	const executables = path.join(
		virtualEnv,
		process.platform === "win32" ? "Scripts" : "bin",
	);
	mkdirSync(executables, { recursive: true });
	writeFileSync(path.join(virtualEnv, "pyvenv.cfg"), "version = 3.12.4\n");

	const env = getShellEnv(cwd);
	assert.equal(getProjectVirtualEnv(cwd), virtualEnv);
	assert.equal(getVirtualEnvPythonVersion(virtualEnv), "3.12.4");
	assert.equal(env.VIRTUAL_ENV, virtualEnv);
	assert.equal(env.PATH?.split(path.delimiter)[0], executables);
});

void test("missing project .venv leaves the environment unchanged", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-no-venv-"));
	const env = getShellEnv(cwd);

	assert.equal(getProjectVirtualEnv(cwd), undefined);
	assert.equal(env.VIRTUAL_ENV, process.env.VIRTUAL_ENV);
	assert.equal(env.PATH, process.env.PATH);
});
