import { test } from "bun:test";
import assert from "node:assert/strict";
import type { ToolCall } from "../agent/types.ts";
import {
	PermissionManager,
	primaryArgString,
} from "../tools/shared/permissions.ts";

function call(name: string, args: Record<string, unknown>): ToolCall {
	return { id: "t1", name, arguments: JSON.stringify(args) };
}

void test("deny rules win over allow rules and modes", () => {
	const pm = new PermissionManager({
		mode: "acceptAll",
		rules: { deny: ["bash(rm *)"], allow: ["bash"] },
	});
	const verdict = pm.evaluate(call("bash", { command: "rm -rf /tmp/x" }), {
		command: "rm -rf /tmp/x",
	});
	assert.equal(verdict.decision, "deny");
	assert.equal(verdict.source, "rule");
});

void test("allow rule matches glob against the command", () => {
	const pm = new PermissionManager({
		mode: "ask",
		rules: { allow: ["bash(git status*)"] },
	});
	assert.equal(
		pm.evaluate(call("bash", { command: "git status -sb" }), {
			command: "git status -sb",
		}).decision,
		"allow",
	);
	// Different command falls through to the ask mode.
	assert.equal(
		pm.evaluate(call("bash", { command: "git push" }), {
			command: "git push",
		}).decision,
		"ask",
	);
});

void test("plan mode allows read-only tools, denies the rest", () => {
	const pm = new PermissionManager({ mode: "plan" });
	assert.equal(
		pm.evaluate(call("read_file", { path: "a" }), { path: "a" }, {
			readOnly: true,
		} as never).decision,
		"allow",
	);
	const denied = pm.evaluate(call("write_file", { path: "a" }), { path: "a" });
	assert.equal(denied.decision, "deny");
	assert.match(denied.reason ?? "", /plan mode/i);
});

void test("acceptEdits allows edit tools, asks for bash", () => {
	const pm = new PermissionManager({ mode: "acceptEdits" });
	assert.equal(
		pm.evaluate(call("edit_file", { path: "a" }), { path: "a" }).decision,
		"allow",
	);
	assert.equal(
		pm.evaluate(call("bash", { command: "make" }), { command: "make" })
			.decision,
		"ask",
	);
});

void test("session allow persists after an 'always' decision", () => {
	const pm = new PermissionManager({ mode: "ask" });
	assert.equal(
		pm.evaluate(call("write_file", { path: "a" }), { path: "a" }).decision,
		"ask",
	);
	pm.addSessionAllow("write_file");
	assert.equal(
		pm.evaluate(call("write_file", { path: "a" }), { path: "a" }).decision,
		"allow",
	);
});

void test("primaryArgString prefers command, then path, then JSON", () => {
	assert.equal(primaryArgString({ command: "ls", path: "x" }), "ls");
	assert.equal(primaryArgString({ path: "src/a.ts" }), "src/a.ts");
	assert.equal(primaryArgString({ n: 1 }), '{"n":1}');
});

void test("batch bash permissions deny if any command is denied", () => {
	const pm = new PermissionManager({
		mode: "acceptAll",
		rules: { deny: ["bash(rm *)"] },
	});
	const args = {
		commands: [{ command: "printf safe" }, { command: "rm file.txt" }],
	};
	assert.equal(pm.evaluate(call("bash", args), args).decision, "deny");
});

void test("batch bash permissions require every command to match allow rules", () => {
	const pm = new PermissionManager({
		mode: "ask",
		rules: { allow: ["bash(npm test*)"] },
	});
	const allowed = {
		commands: [{ command: "npm test" }, { command: "npm test -- --runInBand" }],
	};
	const mixed = {
		commands: [{ command: "npm test" }, { command: "npm publish" }],
	};
	assert.equal(pm.evaluate(call("bash", allowed), allowed).decision, "allow");
	assert.equal(pm.evaluate(call("bash", mixed), mixed).decision, "ask");
});
