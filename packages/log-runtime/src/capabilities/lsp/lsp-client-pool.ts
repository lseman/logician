import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
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

export interface LspLocation {
	file: string;
	line: number;
	column: number;
}

export interface LspHover {
	contents: string;
	signature?: string;
}

export interface LspSymbol {
	name: string;
	kind: string;
	file: string;
	line: number;
	column: number;
	children?: LspSymbol[];
}

export interface LspCodeAction {
	title: string;
	kind?: string;
	edit?: LspWorkspaceEdit;
	command?: string;
}

export interface LspTextEdit {
	file: string;
	range: { startLine: number; startCol: number; endLine: number; endCol: number };
	newText: string;
}

export interface LspWorkspaceEdit {
	changes: LspTextEdit[];
}

export interface LspServerInfo {
	command: string;
	languageId: string;
	ready: boolean;
}

export interface LspServerDefinition {
	command: string;
	args?: string[];
	languageId: string;
}
// Internal LSP result types for parsing JSON-RPC responses.
interface LspLocationResult {
	uri?: string;
	range?: { start?: { line?: number; character?: number } };
	targetUri?: string;
}

interface HoverResult {
	contents?: string | Array<string | { language?: string; value?: string }> | { language?: string; value?: string };
	range?: { start?: { line?: number } };
}

interface SymbolResult {
	uri?: string;
	name?: string;
	kind?: number;
	range?: { start?: { line?: number; character?: number } };
}

interface SymbolInfoResult {
	name?: string;
	kind?: number;
	location?: { uri?: string; range?: { start?: { line?: number; character?: number } }; selectionRange?: { start?: { line?: number; character?: number } } };
	containerName?: string;
}

interface CodeActionResult {
	title?: string;
	kind?: string;
	command?: { title?: string };
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
	".py": {
		command: "pyright-langserver",
		args: ["--stdio"],
		languageId: "python",
	},
	".pyi": {
		command: "pyright-langserver",
		args: ["--stdio"],
		languageId: "python",
	},
	".ts": {
		command: "typescript-language-server",
		args: ["--stdio"],
		languageId: "typescript",
	},
	".tsx": {
		command: "typescript-language-server",
		args: ["--stdio"],
		languageId: "typescriptreact",
	},
	".js": {
		command: "typescript-language-server",
		args: ["--stdio"],
		languageId: "javascript",
	},
	".jsx": {
		command: "typescript-language-server",
		args: ["--stdio"],
		languageId: "javascriptreact",
	},
	".java": { command: "jdtls", args: [], languageId: "java" },
	".php": { command: "intelephense", args: ["--stdio"], languageId: "php" },
	".vue": {
		command: "vue-language-server",
		args: ["--stdio"],
		languageId: "vue",
	},
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
	private pending = new Map<
		number,
		{
			resolve: (value: unknown) => void;
			reject: (error: Error) => void;
		}
	>();
	private diagnostics = new Map<string, (items: LspDiagnostic[]) => void>();
	private isReady = false;
	private capabilitiesResult: Record<string, unknown> | null = null;
	readonly ready: Promise<void>;

	constructor(
		private definition: LspServerDefinition,
		cwd: string,
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
				.then(result => {
					clearTimeout(timer);
					this.child.off("error", onError);
					this.isReady = true;
					this.capabilitiesResult = (result as { capabilities?: Record<string, unknown> })?.capabilities ?? null;
					this.notify("initialized", {});
					resolve();
				})
				.catch(error => {
					clearTimeout(timer);
					reject(error);
				});
		});
	}

	async diagnose(
		filePath: string,
		timeoutMs: number,
	): Promise<LspDiagnostic[]> {
		await this.ready;
		const uri = pathToFileURL(filePath).href;
		const text = await readFile(filePath, "utf8");
		this.version++;
		const result = new Promise<LspDiagnostic[]>(resolve => {
			const timer = setTimeout(() => {
				this.diagnostics.delete(uri);
				resolve([]);
			}, timeoutMs);
			this.diagnostics.set(uri, items => {
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

	private async openFile(filePath: string): Promise<string> {
		const uri = pathToFileURL(filePath).href;
		if (!this.opened.has(uri)) {
			const text = await readFile(filePath, "utf8");
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
		return uri;
	}

	async goToDefinition(filePath: string, line: number, column: number): Promise<LspLocation[]> {
		await this.ready;
		const uri = await this.openFile(filePath);
		const result = await this.request("textDocument/definition", {
			textDocument: { uri },
			position: { line: line - 1, character: column - 1 },
		});
		return this.locationsFromResult(result as unknown as LspLocationResult);
	}

	async references(filePath: string, line: number, column: number): Promise<LspLocation[]> {
		await this.ready;
		const uri = await this.openFile(filePath);
		const result = await this.request("textDocument/references", {
			textDocument: { uri },
			position: { line: line - 1, character: column - 1 },
			context: { includeDeclaration: false },
		});
		return this.locationsFromResult(result as unknown as LspLocationResult);
	}

	async hover(filePath: string, line: number, column: number): Promise<LspHover | null> {
		await this.ready;
		const uri = await this.openFile(filePath);
		const result = await this.request("textDocument/hover", {
			textDocument: { uri },
			position: { line: line - 1, character: column - 1 },
		});
		return this.hoverFromResult(result as HoverResult | null);
	}

	async symbols(filePath: string): Promise<LspSymbol[]> {
		await this.ready;
		const uri = await this.openFile(filePath);
		const result = await this.request("textDocument/documentSymbols", {
			textDocument: { uri },
		});
		return this.symbolsFromResult(result as unknown as SymbolResult | null);
	}

	async workspaceSymbols(query: string, limit: number = 100): Promise<LspSymbol[]> {
		await this.ready;
		const result = await this.request("workspace/symbols", {
			query,
		});
		return this.workspaceSymbolsFromResult(
			result as unknown as SymbolInfoResult | null,
			limit,
		);
	}

	async codeActions(
		filePath: string,
		line: number,
		column: number,
		kind?: string,
	): Promise<LspCodeAction[]> {
		await this.ready;
		const uri = await this.openFile(filePath);
		const context: { diagnostics?: unknown[]; only?: string } = {};
		if (kind) context.only = kind;
		const result = await this.request("textDocument/codeAction", {
			textDocument: { uri },
			range: {
				start: { line: line - 1, character: column - 1 },
				end: { line: line - 1, character: column - 1 },
			},
			context,
		});
		return this.codeActionsFromResult(result as unknown as CodeActionResult | null);
	}

	async applyWorkspaceEdit(edit: LspWorkspaceEdit): Promise<boolean> {
		await this.ready;
		const documentChanges = edit.changes.map(change => ({
			edits: [
				{
					range: {
						start: { line: change.range.startLine - 1, character: change.range.startCol - 1 },
						end: { line: change.range.endLine - 1, character: change.range.endCol - 1 },
					},
					newText: change.newText,
				},
			],
			uri: pathToFileURL(change.file).href,
		}));
		const result = await this.request("workspace/applyEdit", {
			edit: { documentChanges },
		});
		return (result as { applied?: boolean })?.applied !== false;
	}

	async typeDefinition(filePath: string, line: number, column: number): Promise<LspLocation[]> {
		await this.ready;
		const uri = await this.openFile(filePath);
		const result = await this.request("textDocument/typeDefinition", {
			textDocument: { uri },
			position: { line: line - 1, character: column - 1 },
		});
		return this.locationsFromResult(result as unknown as LspLocationResult);
	}

	async implementation(filePath: string, line: number, column: number): Promise<LspLocation[]> {
		await this.ready;
		const uri = await this.openFile(filePath);
		const result = await this.request("textDocument/implementation", {
			textDocument: { uri },
			position: { line: line - 1, character: column - 1 },
		});
		return this.locationsFromResult(result as unknown as LspLocationResult);
	}

	async capabilities(): Promise<Record<string, unknown> | null> {
		await this.ready;
		// Capabilities are returned during initialize handshake, stored in serverCapabilities
		return this.capabilitiesResult as Record<string, unknown> | null;
	}

	async rawRequest(method: string, payload: Record<string, unknown>): Promise<unknown> {
		await this.ready;
		return this.request(method, payload);
	}

	get status(): LspServerInfo {
		return {
			command: this.definition.command,
			languageId: this.definition.languageId,
			ready: this.isReady,
		};
	}

	private locationsFromResult(result: unknown): LspLocation[] {
		if (!result) return [];
		const items = Array.isArray(result) ? result : result instanceof Object && "targetUri" in result ? [result] : [];
		return items.map((item: unknown) => {
			if (typeof item !== "object" || item === null) return null;
			const entry = item as Record<string, unknown>;
			const uri = String(entry.uri ?? entry.targetUri ?? "");
			const range = entry.range as { start?: { line?: number; character?: number } } | undefined;
			if (!uri) return null;
			const filePath = uri.startsWith("file://") ? uri.slice(7) : uri;
			return {
				file: filePath,
				line: Number(range?.start?.line ?? 0) + 1,
				column: Number(range?.start?.character ?? 0) + 1,
			};
		}).filter((loc): loc is LspLocation => loc !== null);
	}

	private hoverFromResult(result: HoverResult | null): LspHover | null {
		if (!result || !("contents" in result)) return null;
		const contents = result.contents;
		let text = "";
		if (typeof contents === "string") {
			text = contents;
		} else if (Array.isArray(contents)) {
			text = contents
				.map((c: unknown) => (typeof c === "string" ? c : typeof c === "object" && c !== null && "value" in c ? String((c as { value: unknown }).value) : ""))
				.join("\n");
		} else if (typeof contents === "object" && contents !== null) {
			text = String((contents as { value?: string }).value ?? "");
		}
		const signature = typeof (result as { range?: unknown }).range === "object"
			? ` at line ${(result as { range: { start?: { line?: number } } }).range?.start?.line ?? 0}`
			: "";
		return { contents: text, signature };
	}

	private symbolsFromResult(result: unknown): LspSymbol[] {
		if (!result || !Array.isArray(result)) return [];
		const kindNames = this.kindNames;
		return result
			.map((item: unknown) => {
				if (typeof item !== "object" || item === null) return null;
				const entry = item as Record<string, unknown>;
				const uri = String(entry.uri ?? "");
				const range = entry.range as { start?: { line?: number; character?: number } } | undefined;
				const name = String(entry.name ?? "Unknown");
				const kind = kindNames[Number(entry.kind) ?? 0] ?? "Unknown";
				if (!uri) return null;
				const filePath = uri.startsWith("file://") ? uri.slice(7) : uri;
				return {
					name,
					kind,
					file: filePath,
					line: Number(range?.start?.line ?? 0) + 1,
					column: Number(range?.start?.character ?? 0) + 1,
				};
			})
			.filter((s): s is LspSymbol => s !== null);
	}

	private workspaceSymbolsFromResult(result: unknown, limit: number): LspSymbol[] {
		if (!result || !Array.isArray(result)) return [];
		const kindNames = this.kindNames;
		return result
			.slice(0, limit)
			.map((item: unknown) => {
				if (typeof item !== "object" || item === null) return null;
				const entry = item as Record<string, unknown>;
				const location = entry.location as { uri?: string; range?: { start?: { line?: number; character?: number } }; selectionRange?: { start?: number; character?: number } } | undefined;
				const uri = location?.uri ?? "";
				const range = location?.range ?? location?.selectionRange;
				const start = range as { line?: number; character?: number } | undefined;
				const name = String(entry.name ?? "Unknown");
				const kind = kindNames[Number(entry.kind) ?? 0] ?? "Unknown";
				if (!uri) return null;
				const filePath = uri.startsWith("file://") ? uri.slice(7) : uri;
				return {
					name,
					kind,
					file: filePath,
					line: Number(start?.line ?? 0) + 1,
					column: Number(start?.character ?? 0) + 1,
				};
			})
			.filter((s): s is LspSymbol => s !== null);
	}

	private codeActionsFromResult(result: unknown): LspCodeAction[] {
		if (!result || !Array.isArray(result)) return [];
		const actions: LspCodeAction[] = [];
		for (const item of result) {
			if (typeof item !== "object" || item === null) continue;
			const entry = item as Record<string, unknown>;
			const title = String(entry.title ?? "Unknown action");
			if (title === "") continue;
			actions.push({
				title,
				kind: typeof entry.kind === "string" ? entry.kind : undefined,
				command: typeof entry.command === "object" && entry.command !== null ? String((entry.command as Record<string, unknown>).title ?? "") : undefined,
			});
		}
		return actions;
	}


	private readonly kindNames: Record<number, string> = {
		1: "File", 2: "Module", 3: "Namespace", 4: "Package", 5: "Class", 6: "Method",
		7: "Property", 8: "Field", 9: "Constructor", 10: "Enum", 11: "Interface",
		12: "Function", 13: "Variable", 14: "Constant", 15: "String", 16: "Number",
		17: "Boolean", 18: "Array", 19: "Object", 20: "Key", 21: "Null", 22: "EnumMember",
		23: "Struct", 24: "Event", 25: "Operator", 26: "TypeParameter",
	};

	close(): void {
		this.child.kill();
	}

	private request(
		method: string,
		params: Record<string, unknown>,
	): Promise<unknown> {
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
		this.child.stdin.write(
			`Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`,
		);
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
			const raw = this.buffer
				.subarray(bodyStart, bodyStart + length)
				.toString("utf8");
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
			if (message.error)
				pending.reject(new Error(message.error.message || "LSP error"));
			else pending.resolve(message.result);
			return;
		}
		if (message.method !== "textDocument/publishDiagnostics") return;
		const uri = String(message.params?.uri ?? "");
		const callback = this.diagnostics.get(uri);
		if (!callback) return;
		const raw = Array.isArray(message.params?.diagnostics)
			? (message.params.diagnostics as Array<Record<string, unknown>>)
			: [];
		callback(
			raw.slice(0, 10).map(item => {
				const range = item.range as
					| { start?: { line?: number; character?: number } }
					| undefined;
				return {
					line: Number(range?.start?.line ?? 0) + 1,
					column: Number(range?.start?.character ?? 0) + 1,
					message: String(item.message ?? "Language server diagnostic"),
					code:
						typeof item.code === "number" || typeof item.code === "string"
							? item.code
							: undefined,
					severity:
						typeof item.severity === "number" ? item.severity : undefined,
					source: typeof item.source === "string" ? item.source : undefined,
				};
			}),
		);
	}
}

/** Lazy, per-language LSP transport pool. Missing servers fail silently. */
export class LspClientPool {
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

	// -- Full LSP action surface --

	async action(
		action: string,
		filePath?: string,
		line?: number,
		column?: number,
		query?: string,
		newName?: string,
		_apply?: boolean,
	): Promise<unknown> {
		if (!filePath) {
			switch (action.toLowerCase()) {
				case "status": return Array.from(this.clients.values()).map(c => c.status);
				case "reload": {
					for (const client of this.clients.values()) client.close();
					this.clients.clear();
					return "Servers reloaded";
				}
				case "capabilities": {
					const results: Record<string, unknown> = {};
					for (const [key, client] of this.clients) {
						results[key] = client.capabilities();
					}
					return results;
				}
			}
			return { error: `Action '${action}' requires a file path` };
		}

		const extension = path.extname(filePath).toLowerCase();
		const definition = this.servers[extension];
		if (!definition) return { error: `No LSP server configured for ${extension}` };
		const clientKey = `${definition.command}:${definition.languageId}`;
		let client = this.clients.get(clientKey);
		if (!client) {
			client = new LspClient(definition, this.cwd, this.timeoutMs);
			this.clients.set(clientKey, client);
		}
		try {
			const l = line ?? 1;
			const c = column ?? 1;
			const q = query ?? "";
			const n = newName ?? "";
			switch (action.toLowerCase()) {
				case "diagnostics": return await client.diagnose(filePath, this.timeoutMs);
				case "definition": return await client.goToDefinition(filePath, l, c);
				case "references": return await client.references(filePath, l, c);
				case "hover": return await client.hover(filePath, l, c);
				case "symbols": return await client.symbols(filePath);
				case "workspace-symbols": return await client.workspaceSymbols(q);
				case "code-actions": return await client.codeActions(filePath, l, c, q);
				case "rename": return await this.doRename(client, filePath, l, c, n);
				case "type-definition": return await client.typeDefinition(filePath, l, c);
				case "implementation": return await client.implementation(filePath, l, c);
				case "status": return client.status;
				case "capabilities": return client.capabilities();
				case "reload": {
					client.close();
					this.clients.delete(clientKey);
					return "Server reloaded";
				}
				default: return { error: `Unknown LSP action: ${action}` };
			}
		} catch (error) {
			return { error: String(error) };
		}
	}

	private async doRename(
		client: LspClient,
		filePath: string,
		line: number,
		column: number,
		newName: string,
	): Promise<LspWorkspaceEdit | null> {
		const refs = await client.references(filePath, line, column);
		if (refs.length === 0) return null;
		const changes: LspTextEdit[] = [];
		for (const ref of refs) {
			changes.push({
				file: ref.file,
				range: { startLine: ref.line, startCol: ref.column, endLine: ref.line, endCol: ref.column },
				newText: newName,
			});
		}
		return { changes };
	}

	close(): void {
		for (const client of this.clients.values()) client.close();
		this.clients.clear();
	}
}
