import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { AgentCoreBridge } from "../application/agent-bridge.ts";

function bridgeWithPathPolicy(
	cwd: string,
	options: { allowedPaths?: string[]; allowAllPaths?: boolean },
): AgentCoreBridge {
	return new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		cwd,
		runtimeHooksEnabled: false,
		mcpEager: false,
		...options,
	});
}

void test("bridge propagates allowedPaths to live file tools", async () => {
	const root = mkdtempSync(join(tmpdir(), "logician-path-policy-"));
	try {
		const cwd = join(root, "cwd");
		const allowed = join(root, "allowed");
		const file = join(allowed, "file.txt");
		mkdirSync(cwd);
		mkdirSync(allowed);
		writeFileSync(file, "allowed\n", { encoding: "utf8", flag: "wx" });

		const bridge = bridgeWithPathPolicy(cwd, { allowedPaths: [allowed] });
		const result = await bridge.getTools().execute({
			id: "read_allowed",
			name: "read_file",
			arguments: JSON.stringify({ path: file }),
		});

		assert.equal(result.isError, undefined);
		assert.match(result.content, /allowed/);
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
});

void test("bridge propagates allowAllPaths to live file tools", async () => {
	const root = mkdtempSync(join(tmpdir(), "logician-path-policy-"));
	try {
		const cwd = join(root, "cwd");
		const file = join(root, "outside.txt");
		mkdirSync(cwd);
		writeFileSync(file, "outside\n", "utf8");

		const bridge = bridgeWithPathPolicy(cwd, { allowAllPaths: true });
		const result = await bridge.getTools().execute({
			id: "read_outside",
			name: "read_file",
			arguments: JSON.stringify({ path: file }),
		});

		assert.equal(result.isError, undefined);
		assert.match(result.content, /outside/);
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
});
