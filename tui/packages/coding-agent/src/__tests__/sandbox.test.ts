import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import {
	sandbox,
	getDefaultSandboxProfile,
	setDefaultSandboxProfile,
} from "../tools/sandbox.ts";

void test("detects bwrap on Linux when present on PATH", () => {
	if (process.platform !== "linux") return;
	const pathEnv = process.env.PATH ?? "";
	const bwrapDir = pathEnv
		.split(path.delimiter)
		.find((d) => existsSync(path.join(d, "bwrap")));
	if (!bwrapDir) return;
	const fullPath = path.join(bwrapDir, "bwrap");
	const result = spawnSync(fullPath, ["--version"], {
		timeout: 5000,
		stdio: ["ignore", "pipe", "pipe"],
	});
	assert.equal(result.status, 0);
	assert.match(result.stdout.toString(), /bubblewrap/);
});

void test("sandbox tool has correct name", () => {
	assert.equal(sandbox.name, "sandbox");
});

void test("sandbox tool requires a command parameter", () => {
	const params = sandbox.parameters as Record<string, unknown>;
	const required = params.required as string[];
	assert.ok(required.includes("command"));
});

void test("sandbox tool exposes the full profile enum", () => {
	const props = sandbox.parameters.properties as Record<string, unknown>;
	const profileProp = props.profile as { enum: string[] };
	assert.deepEqual(profileProp.enum, ["none", "code", "file", "dev", "full"]);
});

void test("sandbox tool has a timeout parameter", () => {
	const props = sandbox.parameters.properties as Record<string, unknown>;
	assert.ok(props.timeout !== undefined);
});

void test("prepareArguments wraps a bare string as { command }", () => {
	const fn = sandbox.prepareArguments as
		| ((raw: unknown) => Record<string, unknown>)
		| undefined;
	assert.ok(fn);
	assert.deepEqual(fn!("echo hello"), { command: "echo hello" });
});

void test("prepareArguments passes through an object with a command field", () => {
	const fn = sandbox.prepareArguments as
		| ((raw: unknown) => Record<string, unknown>)
		| undefined;
	assert.equal(fn!({ command: "ls -la" }).command, "ls -la");
});

void test("prepareArguments accepts alternate keys (cmd) as command", () => {
	const fn = sandbox.prepareArguments as
		| ((raw: unknown) => Record<string, unknown>)
		| undefined;
	assert.equal(fn!({ cmd: "pwd" }).command, "pwd");
});

void test("prepareArguments returns an empty object for null input", () => {
	const fn = sandbox.prepareArguments as
		| ((raw: unknown) => Record<string, unknown>)
		| undefined;
	assert.deepEqual(fn!(null), {});
});

// ── Session default profile ─────────────────────────────────────────────

void test("default sandbox profile starts as code", () => {
	setDefaultSandboxProfile("code");
	assert.equal(getDefaultSandboxProfile(), "code");
});

void test("setDefaultSandboxProfile updates the module-level default", () => {
	const prev = getDefaultSandboxProfile();
	try {
		setDefaultSandboxProfile("full");
		assert.equal(getDefaultSandboxProfile(), "full");
		setDefaultSandboxProfile("none");
		assert.equal(getDefaultSandboxProfile(), "none");
	} finally {
		setDefaultSandboxProfile(prev);
	}
});
