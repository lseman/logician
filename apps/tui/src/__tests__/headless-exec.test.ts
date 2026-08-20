import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	type AgentProtocolNotification,
	createNotification,
	type RuntimeEvent,
} from "@logician/agent-protocol";
import {
	EXEC_STREAM_SCHEMA,
	type ExecBridge,
	parseExecArgs,
	runHeadlessExec,
} from "../app/headless-exec.ts";

class MemoryWriter {
	value = "";
	write(chunk: string | Uint8Array): boolean {
		this.value += String(chunk);
		return true;
	}
}

class FakeBridge implements ExecBridge {
	private callback:
		| ((notification: AgentProtocolNotification) => void)
		| undefined;
	stopped = false;
	constructor(private readonly events: RuntimeEvent[]) {}
	onNotification(
		callback: (notification: AgentProtocolNotification) => void,
	): () => void {
		this.callback = callback;
		return () => {
			this.callback = undefined;
		};
	}
	onError(_callback: (error: Error) => void): void {}
	async init(): Promise<Record<string, unknown>> {
		return {};
	}
	async sendMessage(_message: string): Promise<void> {
		for (const [index, event] of this.events.entries()) {
			this.callback?.(createNotification(event, index + 1));
		}
	}
	async stop(): Promise<void> {
		this.stopped = true;
	}
	getConfig(): { baseUrl: string; model: string } {
		return { baseUrl: "http://test", model: "test-model" };
	}
	respondToPermission(): boolean {
		return true;
	}
	respondToQuestion(): boolean {
		return true;
	}
}

void test("parseExecArgs accepts --jsonl before or after the prompt", () => {
	assert.deepEqual(parseExecArgs(["--jsonl", "fix", "tests"]), {
		jsonl: true,
		prompt: "fix tests",
	});
	assert.deepEqual(parseExecArgs(["fix tests", "--jsonl"]), {
		jsonl: true,
		prompt: "fix tests",
	});
	assert.deepEqual(parseExecArgs(["--jsonl", "--", "-prefixed prompt"]), {
		jsonl: true,
		prompt: "-prefixed prompt",
	});
	assert.throws(() => parseExecArgs([]), /Usage:/);
	assert.throws(() => parseExecArgs(["--wat", "hello"]), /Unknown exec option/);
});

void test("jsonl output is terminal-clean and ends with metadata then done", async () => {
	const stdout = new MemoryWriter();
	const stderr = new MemoryWriter();
	const bridge = new FakeBridge([
		{ type: "thinking_token", token: "private reasoning" },
		{ type: "token", token: "Hello" },
		{ type: "token", token: " world" },
		{
			type: "context_update",
			tokens: 42,
			maxTokens: 1000,
		},
	]);

	const exitCode = await runHeadlessExec(bridge, {
		prompt: "Say hello",
		jsonl: true,
		cwd: "/workspace",
		stdout,
		stderr,
		runId: "exec_test",
		now: () => 100,
	});

	assert.equal(exitCode, 0);
	assert.equal(stderr.value, "");
	assert.equal(bridge.stopped, true);
	const records = stdout.value
		.trim()
		.split("\n")
		.map(line => JSON.parse(line));
	assert.deepEqual(
		records.map(record => record.type),
		["content", "content", "metadata", "done"],
	);
	assert.ok(records.every(record => record.schema === EXEC_STREAM_SCHEMA));
	assert.ok(records.every(record => record.schema_version === 1));
	assert.ok(records.every(record => record.run_id === "exec_test"));
	assert.equal(records[2].meta.receipt_kind, "terminal");
	assert.equal(records[2].meta.visible_final_answer_chars, 11);
	assert.equal(records[2].meta.context_tokens, 42);
	assert.equal(stdout.value.includes("private reasoning"), false);
});

void test("interactive permission fails closed and still emits one done", async () => {
	const stdout = new MemoryWriter();
	const bridge = new FakeBridge([
		{
			type: "permission_request",
			toolName: "shell",
			toolCallId: "tool-1",
			args: {},
		},
	]);

	const exitCode = await runHeadlessExec(bridge, {
		prompt: "run it",
		jsonl: true,
		cwd: "/workspace",
		stdout,
		stderr: new MemoryWriter(),
		runId: "exec_test",
		now: () => 100,
	});

	const records = stdout.value
		.trim()
		.split("\n")
		.map(line => JSON.parse(line));
	assert.equal(exitCode, 1);
	assert.equal(records.filter(record => record.type === "done").length, 1);
	assert.deepEqual(
		records.slice(-2).map(record => record.type),
		["metadata", "done"],
	);
	assert.equal(records.at(-2)?.meta.status, "failed");
});
