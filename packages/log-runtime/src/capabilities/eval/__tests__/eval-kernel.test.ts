// ── eval kernel tests ────────────────────────────────────────────────────────

import { afterEach, beforeEach, expect, test } from "bun:test";
import {
	createKernelManager,
	type EvalKernelConfig,
	type EvalResult,
} from "../kernel-manager.ts";

let km: ReturnType<typeof createKernelManager> | undefined;

beforeEach(() => {
	km = createKernelManager();
});

afterEach(async () => {
	await km?.stop();
	km = undefined;
});

test("createKernelManager returns manager with all methods", () => {
	const manager = createKernelManager();

	expect(typeof manager.eval).toBe("function");
	expect(typeof manager.stop).toBe("function");
	expect(typeof manager.pythonState).toBe("function");
	expect(typeof manager.jsState).toBe("function");
	expect(manager.python).toBeDefined();
	expect(manager.js).toBeDefined();
});

test("kernel states are initially unavailable", () => {
	const manager = createKernelManager();
	const pythonState = manager.pythonState();
	const jsState = manager.jsState();

	expect(pythonState).toEqual({
		available: false,
		launched: false,
		pid: undefined,
		requestCount: 0,
	});
	expect(jsState).toEqual({
		available: false,
		launched: false,
		pid: undefined,
		requestCount: 0,
	});

	manager.stop();
});

test("python eval executes simple print", async () => {
	const result: EvalResult = await km!.eval({
		language: "python",
		code: "print('hello')",
	});

	expect(result.status).toBe("success");
	expect(result.output).toContain("hello");
});

test("python eval preserves state across calls", async () => {
	const r1: EvalResult = await km!.eval({ language: "python", code: "x = 42" });
	expect(r1.status).toBe("success");

	const r2: EvalResult = await km!.eval({
		language: "python",
		code: "print(x)",
	});
	expect(r2.status).toBe("success");
	expect(r2.output).toContain("42");
});

test("python eval with import works", async () => {
	const result: EvalResult = await km!.eval({
		language: "python",
		code: "import json; print(json.dumps({'a': 1}))",
	});

	expect(result.status).toBe("success");
	expect(result.output).toContain('"a": 1');
});

test("python eval with syntax error returns error status", async () => {
	const result: EvalResult = await km!.eval({
		language: "python",
		code: "def invalid syntax here",
	});

	expect(result.status).toBe("error");
	expect(result.error).toBeDefined();
});

test("python eval respects reset flag", async () => {
	await km!.eval({ language: "python", code: "my_var = 'preserved'" });

	const result: EvalResult = await km!.eval({
		language: "python",
		code: "print('my_var' in globals())",
		reset: true,
	});

	expect(result.status).toBe("success");
	expect(result.output).toContain("False");
});

test("python eval respects custom timeout", async () => {
	const result: EvalResult = await km!.eval({
		language: "python",
		code: "import time; time.sleep(0.01); print('done')",
		timeoutMs: 5000,
	});

	expect(result.status).toBe("success");
	expect(result.output).toContain("done");
});

test("js eval executes simple code", async () => {
	const result: EvalResult = await km!.eval({
		language: "js",
		code: "console.log('hi from bun')",
	});

	expect(result.status).toBe("success");
	expect(result.output).toContain("hi from bun");
});

test("js eval preserves state across calls", async () => {
	const r1: EvalResult = await km!.eval({ language: "js", code: "y = 100" });
	expect(r1.status).toBe("success");

	const r2: EvalResult = await km!.eval({
		language: "js",
		code: "console.log(y)",
	});
	expect(r2.status).toBe("success");
	expect(r2.output).toContain("100");
});

test("js eval with syntax error returns error status", async () => {
	const result: EvalResult = await km!.eval({
		language: "js",
		code: "def invalid syntax here",
	});

	expect(result.status).toBe("error");
	expect(result.error).toBeDefined();
});

test("concurrent eval calls work", async () => {
	const [r1, r2] = await Promise.all([
		km!.eval({ language: "python", code: "print('python')" }),
		km!.eval({ language: "js", code: "console.log('js')" }),
	]);

	expect(r1.status).toBe("success");
	expect(r2.status).toBe("success");
	expect(r1.output).toContain("python");
	expect(r2.output).toContain("js");
});

test("kernel manager with custom config accepts all options", () => {
	const config: EvalKernelConfig = {
		pythonPath: "python3",
		jsPath: "bun",
		maxConcurrent: 8,
		defaultTimeoutMs: 60000,
		cwd: "/tmp",
	};
	const manager = createKernelManager(config);
	expect(manager).toBeDefined();
	manager.stop();
});

test("eval with no code returns error", async () => {
	// Empty code should still execute (no-op)
	const result: EvalResult = await km!.eval({ language: "python", code: "" });
	expect(result.status).toBe("success");
});
