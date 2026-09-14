import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { OpenAIBackend } from "../../capabilities/provider/backend.ts";
import {
	clearBashDebugger,
	getBashDebuggerReport,
	logStage,
	setBashDebugger,
	writeBashDebugEvent,
} from "../../capabilities/tools/bash-debugger.ts";
import { ToolRegistry } from "../../capabilities/tools/registry.ts";

const originalPath = process.env.LOGICIAN_BASH_DEBUG_FILE;
afterEach(() => {
	setBashDebugger(false);
	clearBashDebugger();
	if (originalPath === undefined) delete process.env.LOGICIAN_BASH_DEBUG_FILE;
	else process.env.LOGICIAN_BASH_DEBUG_FILE = originalPath;
});

test("records final arguments and successful output for prepared and direct calls", async () => {
	delete process.env.LOGICIAN_BASH_DEBUG_FILE;
	setBashDebugger(true);
	const registry = new ToolRegistry();
	registry.register({
		name: "bash",
		description: "test",
		parameters: { type: "object", properties: {} },
		prepareArguments: raw => {
			const args = raw as Record<string, unknown>;
			args.command = args.cmd;
			delete args.cmd;
			return args;
		},
		execute: async args => String(args.command),
	});
	for (const prepared of [false, true]) {
		clearBashDebugger();
		const call = {
			id: String(prepared),
			name: "bash",
			arguments: '{"cmd":"echo hello"}',
		};
		if (prepared) {
			const ready = registry.prepare(call);
			await registry.execute(ready.call, undefined, ready.args);
		} else await registry.execute(call);
		const report = getBashDebuggerReport();
		expect(report).toContain('Stage 1 (parsed):  {"cmd":"echo hello"}');
		expect(report).toContain('Stage 3 (final):   {"command":"echo hello"}');
		expect(report).toContain("result:            echo hello");
	}
});

test("writes full JSONL diagnostics only when enabled and survives file errors", () => {
	const dir = mkdtempSync(join(tmpdir(), "bash-debugger-"));
	try {
		const path = join(dir, "trace.jsonl");
		process.env.LOGICIAN_BASH_DEBUG_FILE = path;
		setBashDebugger(false);
		writeBashDebugEvent("disabled", {});
		setBashDebugger(true);
		const command = "x".repeat(1000);
		logStage("test", "bash", 0, JSON.stringify({ command }));
		const events = readFileSync(path, "utf8")
			.trim()
			.split("\n")
			.map(line => JSON.parse(line));
		expect(events).toHaveLength(1);
		expect(events[0].data.value).toBe(JSON.stringify({ command }));
		process.env.LOGICIAN_BASH_DEBUG_FILE = join(dir, "missing", "trace");
		expect(() => writeBashDebugEvent("test", {})).not.toThrow();
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test("flags blank commands without claiming the model caused the failure", () => {
	delete process.env.LOGICIAN_BASH_DEBUG_FILE;
	setBashDebugger(true);
	logStage("blank", "bash", 3, { command: "  " });
	expect(getBashDebuggerReport()).toContain("YES");
	expect(getBashDebuggerReport()).not.toContain("This means the model sent");
});

test("captures the exact provider prompt and empty arguments before normalization", async () => {
	const dir = mkdtempSync(join(tmpdir(), "bash-provider-"));
	const originalFetch = globalThis.fetch;
	try {
		const path = join(dir, "trace.jsonl");
		process.env.LOGICIAN_BASH_DEBUG_FILE = path;
		setBashDebugger(true);
		const chunk = {
			choices: [
				{
					delta: {
						tool_calls: [
							{
								index: 0,
								id: "empty-call",
								function: { name: "bash", arguments: "" },
							},
						],
					},
					finish_reason: "tool_calls",
				},
			],
		};
		globalThis.fetch = (async (
			_input: RequestInfo | URL,
			_init?: RequestInit,
		) =>
			new Response(
				"data: " + JSON.stringify(chunk) + "\n\ndata: [DONE]\n\n",
			)) as typeof fetch;
		const backend = new OpenAIBackend({
			baseUrl: "http://test.local",
			model: "test",
		});
		const messages = [{ role: "user", content: "run pwd" }];
		const response = await backend.generate(messages);
		expect(response.toolCalls[0]?.arguments).toBe("{}");
		const events = readFileSync(path, "utf8")
			.trim()
			.split("\n")
			.map(line => JSON.parse(line));
		const request = events.find(event => event.event === "provider.request");
		expect(request.data.body.messages).toEqual(messages);
		const assembled = events.find(
			event => event.event === "provider.assembled",
		);
		expect(assembled.data.toolCalls[0].arguments).toBe("");
		expect(assembled.data.requestId).toBe(request.data.requestId);
		expect(events.find(event => event.event === "provider.sse").data.data).toBe(
			JSON.stringify(chunk),
		);
	} finally {
		globalThis.fetch = originalFetch;
		rmSync(dir, { recursive: true, force: true });
	}
});
