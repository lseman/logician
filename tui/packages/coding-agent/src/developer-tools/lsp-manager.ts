import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

export interface LspDiagnostic {
	line: number;
	column: number;
	message: string;
	code?: number | string;
	severity?: number;
	source?: string;
}

export interface LspServerDefinition {
	command: string;
	args?: string[];
	languageId: string;
}

interface JsonRpcMessage {
	id?: number;
	method?: string;
	result?: unknown;
	error?: { message?: string };
	params?: Record<string, unknown>;
}

const SERVERS: Record<string, LspServerDefinition> = {
	".rs": { command: "rust-analyzer", args: [], languageId: "rust" },
	".go": { command: "gopls", args: ["serve"], languageId: "go" },
	".py": { command: "pyright-langserver", args: ["--stdio"], languageId: "python" },
	".pyi": { command: "pyright-langserver", args: ["--stdio"], languageId: "python" },
	".ts": { command: "typescript-language-server", args: ["--stdio"], languageId: "typescript" },
	".tsx": { command: "typescript-language-server", args: ["--stdio"], languageId: "typescriptreact" },
	".js": { command: "typescript-language-server", args: ["--stdio"], languageId: "javascript" },
	".jsx": { command: "typescript-language-server", args: ["--stdio"], languageId: "javascriptreact" },
	".java": { command: "jdtls", args: [], languageId: "java" },
	".php": { command: "intelephense", args: ["--stdio"], languageId: "php" },
	".vue": { command: "vue-language-server", args: ["--stdio"], languageId: "vue" },
	".c": { command: "clangd", args: [], languageId: "c" },
	".h": { command: "clangd", args: [], languageId: "c" },
	".cpp": { command: "clangd", args: [], languageId: "cpp" },
	".cc": { command: "clangd", args: [], languageId: "cpp" },
	".cxx": { command: "clangd", args: [], languageId: "cpp" },
	".hpp": { command: "clangd", args: [], languageId: "cpp" },
};

class LspClient {
	private child: ChildProcessWithoutNullStreams;
	private buffer = Buffer.alloc(0);
	private nextId = 1;
	private version = 0;
	private opened = new Set<string>();
	private pending = new Map<number, {
		resolve: (value: unknown) => void;
		reject: (error: Error) => void;
	}>();
	private diagnostics = new Map<string, (items: LspDiagnostic[]) => void>();
	readonly ready: Promise<void>;

	constructor(
		private definition: LspServerDefinition,
		private cwd: string,
		initializeTimeoutMs: number,
	) {
		this.child = spawn(definition.command, definition.args ?? [], {
			cwd,
			stdio: ["pipe", "pipe", "pipe"],
		});
		this.child.stdout.on("data", (chunk: Buffer) => this.consume(chunk));
		this.child.stdin.on("error", () => {});
		this.ready = new Promise<void>((resolve, reject) => {
			const onError = (error: Error): void => reject(error);
			this.child.once("error", onError);
			const timer = setTimeout(() => {
				this.child.kill();
				reject(new Error(`LSP initialize timed out: ${definition.command}`));
			}, initializeTimeoutMs);
			this.request("initialize", {
				processId: process.pid,
				rootUri: pathToFileURL(cwd).href,
				capabilities: { textDocument: { publishDiagnostics: {} } },
			})
				.then(() => {
					clearTimeout(timer);
					this.child.off("error", onError);
					this.notify("initialized", {});
					resolve();
				})
				.catch((error) => {
					clearTimeout(timer);
					reject(error);
				});
		});
	}

	async diagnose(filePath: string, timeoutMs: number): Promise<LspDiagnostic[]> {
		await this.ready;
		const uri = pathToFileURL(filePath).href;
		const text = await readFile(filePath, "utf8");
		this.version++;
		const result = new Promise<LspDiagnostic[]>((resolve) => {
			const timer = setTimeout(() => {
				this.diagnostics.delete(uri);
				resolve([]);
			}, timeoutMs);
			this.diagnostics.set(uri, (items) => {
				clearTimeout(timer);
				this.diagnostics.delete(uri);
				resolve(items);
			});
		});
		if (this.opened.has(uri)) {
			this.notify("textDocument/didChange", {
				textDocument: { uri, version: this.version },
				contentChanges: [{ text }],
			});
		} else {
			this.opened.add(uri);
			this.notify("textDocument/didOpen", {
				textDocument: {
					uri,
					languageId: this.definition.languageId,
					version: this.version,
					text,
				},
			});
		}
		return result;
	}

	close(): void {
		this.child.kill();
	}

	private request(method: string, params: Record<string, unknown>): Promise<unknown> {
		const id = this.nextId++;
		const promise = new Promise<unknown>((resolve, reject) => {
			this.pending.set(id, { resolve, reject });
		});
		this.send({ jsonrpc: "2.0", id, method, params });
		return promise;
	}

	private notify(method: string, params: Record<string, unknown>): void {
		this.send({ jsonrpc: "2.0", method, params });
	}

	private send(message: Record<string, unknown>): void {
		const body = JSON.stringify(message);
		this.child.stdin.write(`Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
	}

	private consume(chunk: Buffer): void {
		this.buffer = Buffer.concat([this.buffer, chunk]);
		while (true) {
			const headerEnd = this.buffer.indexOf("\r\n\r\n");
			if (headerEnd < 0) return;
			const header = this.buffer.subarray(0, headerEnd).toString("ascii");
			const match = /Content-Length:\s*(\d+)/i.exec(header);
			if (!match) {
				this.buffer = this.buffer.subarray(headerEnd + 4);
				continue;
			}
			const length = Number(match[1]);
			const bodyStart = headerEnd + 4;
			if (this.buffer.length < bodyStart + length) return;
			const raw = this.buffer.subarray(bodyStart, bodyStart + length).toString("utf8");
			this.buffer = this.buffer.subarray(bodyStart + length);
			try {
				this.handle(JSON.parse(raw) as JsonRpcMessage);
			} catch {
				// Ignore malformed server messages; diagnostics remain advisory.
			}
		}
	}

	private handle(message: JsonRpcMessage): void {
		if (message.id !== undefined) {
			const pending = this.pending.get(message.id);
			if (!pending) return;
			this.pending.delete(message.id);
			if (message.error) pending.reject(new Error(message.error.message || "LSP error"));
			else pending.resolve(message.result);
			return;
		}
		if (message.method !== "textDocument/publishDiagnostics") return;
		const uri = String(message.params?.uri ?? "");
		const callback = this.diagnostics.get(uri);
		if (!callback) return;
		const raw = Array.isArray(message.params?.diagnostics)
			? message.params.diagnostics as Array<Record<string, unknown>>
			: [];
		callback(raw.slice(0, 10).map((item) => {
			const range = item.range as { start?: { line?: number; character?: number } } | undefined;
			return {
				line: Number(range?.start?.line ?? 0) + 1,
				column: Number(range?.start?.character ?? 0) + 1,
				message: String(item.message ?? "Language server diagnostic"),
				code: typeof item.code === "number" || typeof item.code === "string"
					? item.code
					: undefined,
				severity: typeof item.severity === "number" ? item.severity : undefined,
				source: typeof item.source === "string" ? item.source : undefined,
			};
		}));
	}
}

/** Lazy, per-language LSP transport pool. Missing servers fail silently. */
export class LspManager {
	private clients = new Map<string, LspClient>();
	private timeoutMs: number;
	private servers: Record<string, LspServerDefinition>;

	constructor(
		private cwd: string,
		options: {
			timeoutMs?: number;
			servers?: Record<string, LspServerDefinition>;
		} = {},
	) {
		this.timeoutMs = options.timeoutMs ?? 2_000;
		this.servers = { ...SERVERS, ...(options.servers ?? {}) };
	}

	async diagnosticsFor(filePath: string): Promise<LspDiagnostic[] | null> {
		const extension = path.extname(filePath).toLowerCase();
		const definition = this.servers[extension];
		if (!definition) return null;
		const clientKey = `${definition.command}:${definition.languageId}`;
		let client = this.clients.get(clientKey);
		if (!client) {
			client = new LspClient(definition, this.cwd, this.timeoutMs);
			this.clients.set(clientKey, client);
		}
		try {
			return await client.diagnose(filePath, this.timeoutMs);
		} catch {
			client.close();
			this.clients.delete(clientKey);
			return null;
		}
	}

	close(): void {
		for (const client of this.clients.values()) client.close();
		this.clients.clear();
	}
}
