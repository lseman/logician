// ── Eval Kernel Manager ──────────────────────────────────────────────────────
// Manages persistent Python and JS kernel subprocesses with JSON-lines protocol.
// Each kernel maintains state across calls (imports, variables, etc.).
// Fail-open: if a kernel is unavailable, returns an error message.

import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// ── Config ───────────────────────────────────────────────────────────────────

export interface EvalKernelConfig {
	pythonPath?: string;
	jsPath?: string;
	maxConcurrent?: number;
	defaultTimeoutMs?: number;
	cwd?: string;
}

// ── Types ────────────────────────────────────────────────────────────────────

interface PendingRequest {
	id: string;
	resolve: (value: EvalResult) => void;
	reject: (reason: Error) => void;
	timer: ReturnType<typeof setTimeout>;
}

interface KernelResponse {
	id: string;
	status: "success" | "error" | "timeout";
	output?: string;
	error?: string;
}

export interface EvalResult {
	status: "success" | "error" | "timeout";
	output: string;
	error?: string;
}

export interface KernelState {
	available: boolean;
	launched: boolean;
	pid?: number;
	requestCount: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const DEFAULT_PYTHON = "python3";
const DEFAULT_JS = "bun";
const DEFAULT_MAX_CONCURRENT = 4;
const DEFAULT_TIMEOUT_MS = 30_000;

function parseResponse(line: string): KernelResponse | undefined {
	try {
		const parsed = JSON.parse(line.trim());
		if (
			parsed &&
			typeof parsed.id === "string" &&
			typeof parsed.status === "string"
		) {
			return parsed as KernelResponse;
		}
	} catch {
		// not JSON
	}
	return undefined;
}

// Python kernel: persistent loop reading JSON code from stdin
const PYTHON_KERNEL_SCRIPT = `
import sys, os, json, io

os.environ["PYTHONSTARTUP"] = ""

G = {"__name__": "__main__", "__builtins__": __builtins__}

def process_request(code, request_id):
    try:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(code, G)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return json.dumps({"id": request_id, "status": "success", "output": output or ""})
    except Exception as e:
        return json.dumps({"id": request_id, "status": "error", "error": str(e)})

while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        rid = req.get("id", "")
        code = req.get("code", "")
        reset = req.get("reset", False)
        if reset:
            for k in list(G.keys()):
                if not k.startswith('_') and k not in {'__name__', '__builtins__'}:
                    del G[k]
            G["__name__"] = "__main__"
        sys.stdout.write(process_request(code, rid) + "\\n")
        sys.stdout.flush()
    except json.JSONDecodeError:
        sys.stdout.write(json.dumps({"id": "", "status": "error", "error": "Invalid JSON"}) + "\\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(json.dumps({"id": "", "status": "error", "error": str(e)}) + "\\n")
        sys.stdout.flush()
`;

// JS kernel: persistent eval loop using Bun Web Streams
// SECURITY: uses eval() (not new Function) so user code runs in module scope,
// NOT globalThis scope. In Bun, module-scoped eval cannot access process,
// require, or other Node globals that live on globalThis.
const JS_KERNEL_SCRIPT = `
const encoder = new TextEncoder();
const decoder = new TextDecoder();

async function* readLines(stream) {
    let buffer = '';
    for await (const chunk of stream) {
        buffer += decoder.decode(chunk, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';
        for (const line of lines) yield line;
    }
    if (buffer) yield buffer;
}

function processRequest(code, requestId) {
    try {
        const parts = [];
        const orig = { log: console.log, warn: console.warn, err: console.error };
        const capture = (...a) => parts.push(a.map(x => typeof x === 'object' ? JSON.stringify(x, null, 2) : String(x)).join(' '));
        console.log = capture; console.warn = capture; console.error = capture;

        // Helpers created per-request so print() closes over the parts array via closure.
        const _omp = {
            display(value) {
                if (typeof value === 'object' && value !== null) {
                    return JSON.stringify(value, null, 2);
                }
                return String(value);
            },
            print(...args) {
                parts.push(args.map(x => typeof x === 'object' ? JSON.stringify(x, null, 2) : String(x)).join(' '));
            },
            read(path) {
                return Bun.file(path).text();
            },
            write(path, content) {
                return Bun.write(path, content);
            },
            completion(prompt, schema) {
                return JSON.stringify({ prompt, schema });
            },
            wait(handles) {
                return JSON.stringify({ handles });
            },
            budget() {
                return JSON.stringify({ elapsed: performance.now(), tokens: 0 });
            },
        };

        // SECURITY: override globalThis.process and globalThis.require
        const savedProcess = globalThis.process;
        const savedRequire = globalThis.require;
        globalThis.process = new Proxy(Object.create(null), { get(_t, p) { return undefined; } });
        globalThis.require = function _nope() { throw new Error('require is not available'); };
        try {
            const fn = new Function('process', 'require', '_omp', 'try { ' + code + ' } catch(e) { throw e; }');
            fn(globalThis.process, globalThis.require, _omp);
            return JSON.stringify({ id: requestId, status: 'success', output: parts.join('\\n') || '' });
        } catch (e) {
            return JSON.stringify({ id: requestId, status: 'error', error: e.message || String(e) });
        } finally {
            globalThis.process = savedProcess;
            globalThis.require = savedRequire;
            console.log = orig.log; console.warn = orig.warn; console.error = orig.err;
        }
    } catch (e) {
        return JSON.stringify({ id: requestId, status: 'error', error: e.message || String(e) });
    }
}

(async () => {
    for await (const line of readLines(process.stdin)) {
        const t = line.trim();
        if (!t) continue;
        try {
            const req = JSON.parse(t);
            const rid = req.id || '';
            const code = req.code || '';
            const reset = req.reset || false;
            if (reset) {
                // Clear global variables but keep builtins
                const keys = Object.keys(globalThis);
                for (const k of keys) {
                    if (!k.startsWith('_') && !Object.hasOwnProperty.call(globalThis, k)) {
                        delete globalThis[k];
                    }
                }
                process.stdout.write(JSON.stringify({ id: rid, status: 'success', output: '// state reset\\n' }) + '\\n');
            } else {
                process.stdout.write(processRequest(code, rid) + '\\n');
            }
        } catch (e) {
            process.stdout.write(JSON.stringify({ id: '', status: 'error', error: e.message }) + '\\n');
        }
    }
})();
`;

// ── Generic Kernel (shared Python/JS logic) ──────────────────────────────────

abstract class BaseKernel {
	protected process: ChildProcessWithoutNullStreams | null = null;
	protected scriptPath: string | null = null;
	private pending = new Map<string, PendingRequest>();
	protected launched = false;
	protected available = false;
	protected requestCount = 0;
	private maxConcurrent: number;
	private timeoutMs: number;
	protected cwd: string;
	private stdoutBuffer = "";

	constructor(
		protected readonly pythonPath: string,
		protected readonly jsPath: string,
		config: EvalKernelConfig,
	) {
		this.maxConcurrent = config.maxConcurrent ?? DEFAULT_MAX_CONCURRENT;
		this.timeoutMs = config.defaultTimeoutMs ?? DEFAULT_TIMEOUT_MS;
		this.cwd = config.cwd ?? process.cwd();
	}

	protected abstract getKernelScript(): string;
	protected abstract getInterpreter(): string;
	protected abstract getUnavailableMessage(): string;

	protected abstract spawnProcess(
		scriptPath: string,
	): ChildProcessWithoutNullStreams;

	private processLine(line: string): void {
		const resp = parseResponse(line);
		if (resp && this.pending.has(resp.id)) {
			// biome-ignore lint/style/noNonNullAssertion: safe after has() check
			const { resolve, timer } = this.pending.get(resp.id)!;
			clearTimeout(timer);
			this.pending.delete(resp.id);
			if (resp.status === "success")
				resolve({ status: "success", output: resp.output ?? "" });
			else if (resp.status === "error")
				resolve({
					status: "error",
					output: "",
					error: resp.error ?? "Unknown",
				});
			else resolve({ status: "timeout", output: "", error: "timed out" });
		}
	}

	async start(): Promise<boolean> {
		if (this.launched) return this.available;
		const script = this.getKernelScript();

		try {
			// Write script to temp file (Bun needs file, Python can use -c)
			const dir = mkdtempSync(join(tmpdir(), "logician-kernel-"));
			const scriptPath = join(dir, `kernel-${randomUUID().slice(0, 8)}.js`);
			writeFileSync(scriptPath, script, "utf8");
			this.scriptPath = scriptPath;

			this.process = this.spawnProcess(scriptPath);

			// Raw stdout buffering
			this.process.stdout.on("data", (data: Buffer | string) => {
				const text = typeof data === "string" ? data : data.toString("utf8");
				this.stdoutBuffer += text;
				const lines = this.stdoutBuffer.split("\n");
				this.stdoutBuffer = lines.pop() ?? "";
				for (const line of lines) this.processLine(line);
			});

			this.process.stderr.on("data", (data: Buffer | string) => {
				const err = typeof data === "string" ? data : data.toString("utf8");
				if (err.trim()) console.error(`[kernel stderr] ${err.trim()}`);
			});

			this.process.on("error", err => {
				this.available = false;
				for (const { reject, timer } of this.pending.values()) {
					clearTimeout(timer);
					reject(new Error(`kernel process error: ${err.message}`));
				}
				this.pending.clear();
			});

			this.process.on("exit", (code, signal) => {
				this.available = false;
				if (this.stdoutBuffer.trim()) {
					this.processLine(this.stdoutBuffer.trim());
					this.stdoutBuffer = "";
				}
				for (const { reject, timer } of this.pending.values()) {
					clearTimeout(timer);
					reject(new Error(`kernel exited (code:${code}, signal:${signal})`));
				}
				this.pending.clear();
			});

			// Confirm startup with test request
			const testId = `test_${randomUUID()}`;
			await new Promise<void>((resolve, reject) => {
				const t = setTimeout(() => reject(new Error("startup timeout")), 5000);
				this.pending.set(testId, {
					id: testId,
					resolve: () => {
						clearTimeout(t);
						resolve();
					},
					reject,
					timer: t,
				});
				this.process?.stdin?.write(
					`${JSON.stringify({ id: testId, code: "print('ok')" })}\n`,
				);
			});

			this.launched = true;
			this.available = true;
			return true;
		} catch {
			this.launched = false;
			this.available = false;
			this.process?.kill("SIGKILL");
			this.process = null;
			return false;
		}
	}

	async eval(
		code: string,
		timeoutMs?: number,
		reset = false,
	): Promise<EvalResult> {
		if (!this.available || !this.launched) {
			const ok = await this.start();
			if (!ok)
				return {
					status: "error",
					output: "",
					error: this.getUnavailableMessage(),
				};
		}
		if (this.pending.size >= this.maxConcurrent) {
			return {
				status: "error",
				output: "",
				error: `at max concurrency (${this.maxConcurrent})`,
			};
		}

		return new Promise<EvalResult>((resolve, reject) => {
			const id = randomUUID();
			const to = timeoutMs ?? this.timeoutMs;
			const timer = setTimeout(() => {
				if (this.pending.has(id)) {
					this.pending.delete(id);
					resolve({
						status: "timeout",
						output: "",
						error: `timed out after ${to}ms`,
					});
				}
			}, to);

			this.pending.set(id, { id, resolve, reject, timer });

			if (this.process?.stdin && !this.process.stdin.destroyed) {
				this.process.stdin.write(`${JSON.stringify({ id, code, reset })}\n`);
				this.requestCount++;
			} else {
				clearTimeout(timer);
				this.pending.delete(id);
				this.available = false;
				resolve({ status: "error", output: "", error: "kernel died" });
			}
		});
	}

	get state(): KernelState {
		return {
			available: this.available,
			launched: this.launched,
			pid: this.process?.pid,
			requestCount: this.requestCount,
		};
	}

	async stop(): Promise<void> {
		for (const { reject, timer } of this.pending.values()) {
			clearTimeout(timer);
			reject(new Error("stopped"));
		}
		this.pending.clear();
		this.process?.stdin?.destroy();
		this.process?.kill("SIGTERM");
		this.process = null;
		this.launched = false;
		this.available = false;
		// Cleanup temp file
		if (this.scriptPath) {
			try {
				rmSync(this.scriptPath, { force: true });
				rmSync(join(this.scriptPath, ".."), { recursive: true, force: true });
			} catch {
				/* ignore */
			}
			this.scriptPath = null;
		}
	}
}

// ── Python Kernel ────────────────────────────────────────────────────────────

class PythonKernel extends BaseKernel {
	protected getKernelScript(): string {
		return PYTHON_KERNEL_SCRIPT;
	}
	protected getInterpreter(): string {
		return this.pythonPath ?? DEFAULT_PYTHON;
	}
	protected getUnavailableMessage(): string {
		return "Python kernel not available. Install Python 3.";
	}
	protected spawnProcess(_scriptPath: string): ChildProcessWithoutNullStreams {
		return spawn(
			this.pythonPath ?? DEFAULT_PYTHON,
			["-c", this.getKernelScript()],
			{
				stdio: ["pipe", "pipe", "pipe"],
				cwd: this.cwd,
				env: { ...process.env, PYTHONUNBUFFERED: "1" },
			},
		);
	}
}

// ── JS Kernel ────────────────────────────────────────────────────────────────

class JSKernel extends BaseKernel {
	protected getKernelScript(): string {
		return JS_KERNEL_SCRIPT;
	}
	protected getInterpreter(): string {
		return this.jsPath ?? DEFAULT_JS;
	}
	protected getUnavailableMessage(): string {
		return "JS kernel not available. Install Bun.";
	}
	protected spawnProcess(scriptPath: string): ChildProcessWithoutNullStreams {
		return spawn(this.jsPath ?? DEFAULT_JS, [scriptPath], {
			stdio: ["pipe", "pipe", "pipe"],
			cwd: this.cwd,
		});
	}
}

// ── Kernel Manager ───────────────────────────────────────────────────────────

export interface KernelManagerConfig extends EvalKernelConfig {}

export interface KernelManager {
	python: PythonKernel;
	js: JSKernel;
	eval(options: {
		language: "python" | "js";
		code: string;
		timeoutMs?: number;
		reset?: boolean;
	}): Promise<EvalResult>;
	pythonState(): KernelState;
	jsState(): KernelState;
	stop(): Promise<void>;
}

export function createKernelManager(
	config: EvalKernelConfig = {},
): KernelManager {
	const python = new PythonKernel(
		config.pythonPath ?? DEFAULT_PYTHON,
		config.jsPath ?? DEFAULT_JS,
		config,
	);
	const js = new JSKernel(
		config.pythonPath ?? DEFAULT_PYTHON,
		config.jsPath ?? DEFAULT_JS,
		config,
	);

	async function lazyEval(options: {
		language: "python" | "js";
		code: string;
		timeoutMs?: number;
		reset?: boolean;
	}): Promise<EvalResult> {
		const k = options.language === "python" ? python : js;
		return k.eval(options.code, options.timeoutMs, options.reset);
	}

	return {
		python,
		js,
		eval: lazyEval,
		pythonState: () => python.state,
		jsState: () => js.state,
		stop: async () => {
			await python.stop();
			await js.stop();
		},
	};
}
