import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { bash } from "../../infrastructure/tools/bash.ts";

void test("bash includes output and exit code for non-zero commands", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-bash-"));
	const result = await bash.execute(
		{ command: "printf 'before failure'; exit 7" },
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /before failure/);
	assert.match(content, /Command exited with code 7/);
});

void test("bash executes a structured sequential batch and returns ordered details", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-bash-batch-"));
	const result = await bash.execute(
		{
			commands: [
				{ id: "first", command: "printf one" },
				{ id: "second", command: "printf two; exit 2" },
				{ id: "third", command: "printf three" },
			],
		},
		{ cwd },
	);
	assert.equal(typeof result, "object");
	if (typeof result === "string") return;
	const commands = result.details?.commands as Array<Record<string, unknown>>;
	assert.deepEqual(
		commands.map(entry => entry.id),
		["first", "second", "third"],
	);
	assert.deepEqual(
		commands.map(entry => entry.status),
		["completed", "failed", "completed"],
	);
	assert.deepEqual(
		commands.map(entry => entry.exitCode),
		[0, 2, 0],
	);
});

void test("bash stopOnFailure skips remaining sequential commands", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-bash-stop-"));
	const result = await bash.execute(
		{
			commands: [{ command: "exit 3" }, { command: "printf should-not-run" }],
			stopOnFailure: true,
		},
		{ cwd },
	);
	assert.equal(typeof result, "object");
	if (typeof result === "string") return;
	const commands = result.details?.commands as Array<Record<string, unknown>>;
	assert.deepEqual(
		commands.map(entry => entry.status),
		["failed", "skipped"],
	);
	assert.equal(commands[1].content, "Skipped after failure");
});

void test("bash parallel batches preserve input order", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-bash-parallel-"));
	const result = await bash.execute(
		{
			commands: [
				{ id: "slow", command: "sleep 0.05; printf slow" },
				{ id: "fast", command: "printf fast" },
			],
			mode: "parallel",
			maxConcurrency: 2,
		},
		{ cwd },
	);
	assert.equal(typeof result, "object");
	if (typeof result === "string") return;
	const commands = result.details?.commands as Array<Record<string, unknown>>;
	assert.deepEqual(
		commands.map(entry => entry.id),
		["slow", "fast"],
	);
	assert.deepEqual(
		commands.map(entry => entry.content),
		["slow", "fast"],
	);
});

void test("bash rejects ambiguous and invalid structured inputs", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-bash-invalid-"));
	const ambiguous = await bash.execute(
		{ command: "true", commands: [{ command: "true" }] },
		{ cwd },
	);
	assert.match(
		typeof ambiguous === "string" ? ambiguous : ambiguous.content,
		/either command or commands/,
	);
	const duplicate = await bash.execute(
		{
			commands: [
				{ id: "same", command: "true" },
				{ id: "same", command: "true" },
			],
		},
		{ cwd },
	);
	assert.match(
		typeof duplicate === "string" ? duplicate : duplicate.content,
		/duplicate command id/,
	);
});

void test("bash waitMsBeforeAsync moves long-running command to background task", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-bash-async-"));
	const result = await bash.execute(
		{ command: "sleep 2; echo done", waitMsBeforeAsync: 100 },
		{ cwd },
	);
	assert.equal(typeof result, "object");
	if (typeof result === "string") return;
	assert.match(result.content, /running in the background as task/);
	assert.equal(result.details?.background, true);
	assert.ok(typeof result.details?.taskId === "string");
});

void test("bash runPersistent executes inside persistent shell session", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-bash-persist-"));
	const res1 = await bash.execute(
		{ command: "export MY_VAR='foo123'", runPersistent: true, terminalId: "unit-term" },
		{ cwd },
	);
	assert.equal(typeof res1, "object");

	const res2 = await bash.execute(
		{ command: "echo $MY_VAR", runPersistent: true, terminalId: "unit-term" },
		{ cwd },
	);
	assert.equal(typeof res2, "object");
	if (typeof res2 === "string") return;
	assert.match(res2.content, /foo123/);
});

