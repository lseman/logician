// ── MCP Client implementations ─────────────────────────────────────────────
// Stdio and HTTP MCP clients, JSON-RPC message handling, tool definition parsing.

import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { resolve } from "node:path";

const MCP_PROTOCOL_VERSION = "2025-03-26";

interface JsonRpcRequest {
	jsonrpc: "2.0";
	id?: number;
	method: string;
	params?: Record<string, unknown>;
}

interface PendingRequest {
	resolve: (value: Record<string, unknown>) => void;
	reject: (error: Error) => void;
	timer: NodeJS.Timeout;
}

export interface McpServerConfig {
	enabled?: boolean;
	type?: string;
	command?: string;
	args?: string[];
	env?: Record<string, string>;
	cwd?: string;
	url?: string;
	headers?: Record<string, string>;
	timeout?: number;
}

export interface McpToolDefinition {
	name: string;
	title?: string;
	description: string;
	inputSchema: Record<string, unknown>;
}

export interface McpClient {
	name: string;
	initialize(): Promise<void>;
	listTools(): Promise<McpToolDefinition[]>;
	callTool(name: string, args: Record<string, unknown>): Promise<unknown>;
	close(): void;
}

/**
 * Stdio-based MCP client. Spawns a child process and communicates via
 * JSON-RPC with Content-Length framing.
 */
class StdioMcpClient implements McpClient {
	readonly name: string;
	private config: McpServerConfig;
	private proc: ChildProcessWithoutNullStreams | null = null;
	private nextId = 0;
	private buffer: Buffer<ArrayBufferLike> = Buffer.alloc(0);
	private pending = new Map<number, PendingRequest>();
	private stderr = "";
	private cwd: string;
	private timeoutMs: number;

	constructor(name: string, config: McpServerConfig, cwd: string) {
		this.name = name;
		this.config = config;
		this.cwd = cwd;
		this.timeoutMs = Number(config.timeout || 30) * 1000;
	}

	async initialize(): Promise<void> {
		if (!this.config.command) {
			throw new Error("stdio MCP server is missing command");
		}
		this.proc = spawn(this.config.command, this.config.args || [], {
			cwd: this.config.cwd ? resolve(this.cwd, this.config.cwd) : this.cwd,
			env: { ...process.env, ...expandEnvMap(this.config.env || {}) },
			stdio: "pipe",
		});
		this.proc.stdout.on("data", (chunk) => this.handleStdout(chunk));
		this.proc.stderr.on("data", (chunk) => {
			this.stderr = `${this.stderr}${chunk.toString("utf8")}`.slice(-2000);
		});
		this.proc.on("exit", (code, signal) => {
			const suffix = this.stderr.trim() ? `: ${this.stderr.trim()}` : "";
			const exitStatus = signal || (code ?? "unknown");
			const error = new Error(`MCP server exited (${exitStatus})${suffix}`);
			for (const pending of this.pending.values()) {
				clearTimeout(pending.timer);
				pending.reject(error);
			}
			this.pending.clear();
		});

		await this.rpc("initialize", {
			protocolVersion: MCP_PROTOCOL_VERSION,
			capabilities: { tools: {} },
			clientInfo: { name: "tui", version: "0.2.0" },
		});
		this.notify("notifications/initialized");
	}

	async listTools(): Promise<McpToolDefinition[]> {
		const tools: McpToolDefinition[] = [];
		let cursor: string | undefined;
		do {
			const result = await this.rpc(
				"tools/list",
				cursor ? { cursor } : undefined,
			);
			const rawTools = Array.isArray(result.tools) ? result.tools : [];
			for (const raw of rawTools) {
				tools.push(parseMcpToolDefinition(raw));
			}
			cursor =
				typeof result.nextCursor === "string" ? result.nextCursor : undefined;
		} while (cursor);
		return tools;
	}

	async callTool(
		name: string,
		args: Record<string, unknown>,
	): Promise<unknown> {
		return this.rpc("tools/call", { name, arguments: args });
	}

	close(): void {
		if (!this.proc) return;
		this.proc.kill();
		this.proc = null;
	}

	private rpc(
		method: string,
		params?: Record<string, unknown>,
	): Promise<Record<string, unknown>> {
		if (!this.proc) throw new Error("MCP server is not running");
		const id = ++this.nextId;
		const request: JsonRpcRequest = { jsonrpc: "2.0", id, method };
		if (params) request.params = params;
		const payload = encodeMcpMessage(request);
		const promise = new Promise<Record<string, unknown>>((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`MCP request timed out: ${method}`));
			}, this.timeoutMs);
			this.pending.set(id, { resolve, reject, timer });
		});
		this.proc.stdin.write(payload);
		return promise;
	}

	private notify(method: string, params?: Record<string, unknown>): void {
		if (!this.proc) return;
		const request: JsonRpcRequest = { jsonrpc: "2.0", method };
		if (params) request.params = params;
		this.proc.stdin.write(encodeMcpMessage(request));
	}

	private handleStdout(chunk: Buffer): void {
		this.buffer = Buffer.concat([this.buffer, chunk]);
		while (true) {
			let parsed: {
				message: Record<string, unknown>;
				rest: Buffer;
			} | null;
			try {
				parsed = tryDecodeMcpMessage(this.buffer);
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				for (const pending of this.pending.values()) {
					clearTimeout(pending.timer);
					pending.reject(new Error(`MCP decode failed: ${message}`));
				}
				this.pending.clear();
				this.close();
				return;
			}
			if (!parsed) return;
			this.buffer = parsed.rest;
			this.handleMessage(parsed.message);
		}
	}

	private handleMessage(message: Record<string, unknown>): void {
		const id = typeof message.id === "number" ? message.id : undefined;
		if (id === undefined) return;
		const pending = this.pending.get(id);
		if (!pending) return;
		this.pending.delete(id);
		clearTimeout(pending.timer);
		const error = message.error;
		if (error && typeof error === "object") {
			const err = error as { code?: unknown; message?: unknown };
			pending.reject(
				new Error(
					`MCP error ${String(err.code ?? "?")}: ${String(err.message ?? error)}`,
				),
			);
			return;
		}
		const result = message.result;
		pending.resolve(
			result && typeof result === "object"
				? (result as Record<string, unknown>)
				: {},
		);
	}
}

/**
 * HTTP (streamable) MCP client. Communicates via POST to an HTTP endpoint
 * with SSE response support.
 */
class HttpMcpClient implements McpClient {
	readonly name: string;
	private config: McpServerConfig;
	private nextId = 0;
	private sessionId: string | null = null;
	private timeoutMs: number;

	constructor(name: string, config: McpServerConfig) {
		this.name = name;
		this.config = config;
		this.timeoutMs = Number(config.timeout || 30) * 1000;
	}

	async initialize(): Promise<void> {
		await this.rpc("initialize", {
			protocolVersion: MCP_PROTOCOL_VERSION,
			capabilities: { tools: {} },
			clientInfo: { name: "tui", version: "0.2.0" },
		});
		await this.notify("notifications/initialized");
	}

	async listTools(): Promise<McpToolDefinition[]> {
		const tools: McpToolDefinition[] = [];
		let cursor: string | undefined;
		do {
			const result = await this.rpc(
				"tools/list",
				cursor ? { cursor } : undefined,
			);
			const rawTools = Array.isArray(result.tools) ? result.tools : [];
			for (const raw of rawTools) {
				tools.push(parseMcpToolDefinition(raw));
			}
			cursor =
				typeof result.nextCursor === "string" ? result.nextCursor : undefined;
		} while (cursor);
		return tools;
	}

	async callTool(
		name: string,
		args: Record<string, unknown>,
	): Promise<unknown> {
		return this.rpc("tools/call", { name, arguments: args });
	}

	close(): void {
		this.sessionId = null;
	}

	private async rpc(
		method: string,
		params?: Record<string, unknown>,
	): Promise<Record<string, unknown>> {
		const id = ++this.nextId;
		const request: JsonRpcRequest = { jsonrpc: "2.0", id, method };
		if (params) request.params = params;
		return this.send(request, id);
	}

	private async notify(
		method: string,
		params?: Record<string, unknown>,
	): Promise<void> {
		const request: JsonRpcRequest = { jsonrpc: "2.0", method };
		if (params) request.params = params;
		await this.send(request, undefined);
	}

	private async send(
		request: JsonRpcRequest,
		id: number | undefined,
	): Promise<Record<string, unknown>> {
		if (!this.config.url) throw new Error("HTTP MCP server is missing url");
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), this.timeoutMs);
		try {
			const response = await fetch(this.config.url, {
				method: "POST",
				headers: this.headers(),
				body: JSON.stringify(request),
				signal: controller.signal,
			});
			const sessionId = response.headers.get("Mcp-Session-Id");
			if (sessionId && !this.sessionId) this.sessionId = sessionId;
			const text = await response.text();
			if (!response.ok) {
				throw new Error(`MCP HTTP ${response.status}: ${text.slice(0, 600)}`);
			}
			if (id === undefined) return {};
			const contentType = response.headers.get("Content-Type") || "";
			const envelope = contentType.includes("text/event-stream")
				? parseSseEnvelope(text, id)
				: JSON.parse(text || "{}");
			return unwrapEnvelope(envelope);
		} finally {
			clearTimeout(timer);
		}
	}

	private headers(): Record<string, string> {
		const headers: Record<string, string> = {
			"Content-Type": "application/json",
			Accept: "application/json, text/event-stream",
			...expandEnvMap(this.config.headers || {}),
		};
		if (this.sessionId) headers["Mcp-Session-Id"] = this.sessionId;
		return headers;
	}
}

export function createMcpClient(
	name: string,
	config: McpServerConfig,
	cwd: string,
): McpClient {
	if (
		config.url ||
		config.type === "http" ||
		config.type === "streamable-http"
	) {
		return new HttpMcpClient(name, config);
	}
	return new StdioMcpClient(name, config, cwd);
}

export function createMcpTool(client: McpClient, def: McpToolDefinition): unknown {
	const safeName = safeToolName(def.name);
	const name = `mcp__${safeToolName(client.name)}__${safeName}`;
	return {
		name,
		label: def.title || `MCP: ${def.name}`,
		description:
			def.description || `MCP tool '${def.name}' from server '${client.name}'.`,
		promptSnippet: def.description
			? def.description.length > 80
				? def.description.slice(0, 80) + "..."
				: def.description
			: `MCP tool '${def.name}'`,
		parameters: def.inputSchema,
		execute: async (args: Record<string, unknown>) => {
			const result = await client.callTool(def.name, args);
			return formatMcpToolResult(result);
		},
	};
}

// ── JSON-RPC message helpers ─────────────────────────────────────────────

export function encodeMcpMessage(message: JsonRpcRequest): Buffer {
	// MCP stdio uses one JSON-RPC message per line. Content-Length framing is
	// the old LSP convention and current SDK servers (including Context Mode)
	// wait forever for a JSON line when they receive it.
	return Buffer.from(`${JSON.stringify(message)}\n`, "utf8");
}

export function tryDecodeMcpMessage(
	buffer: Buffer,
): { message: Record<string, unknown>; rest: Buffer } | null {
	const marker = buffer.indexOf("\r\n\r\n");
	const newline = buffer.indexOf("\n");
	if (marker < 0 || (newline >= 0 && newline < marker)) {
		if (newline < 0) return null;
		const line = buffer.slice(0, newline).toString("utf8").trim();
		const rest = buffer.slice(newline + 1);
		if (!line) return { message: {}, rest };
		return {
			message: JSON.parse(line) as Record<string, unknown>,
			rest,
		};
	}
	const header = buffer.slice(0, marker).toString("utf8");
	const lengthMatch = header.match(/Content-Length:\s*(\d+)/i);
	if (!lengthMatch) {
		throw new Error("Malformed MCP message header");
	}
	const length = Number(lengthMatch[1]);
	const bodyStart = marker + 4;
	const bodyEnd = bodyStart + length;
	if (buffer.length < bodyEnd) return null;
	const body = buffer.slice(bodyStart, bodyEnd).toString("utf8");
	return {
		message: JSON.parse(body) as Record<string, unknown>,
		rest: buffer.slice(bodyEnd),
	};
}

// ── Utility functions ────────────────────────────────────────────────────

export function parseMcpToolDefinition(raw: unknown): McpToolDefinition {
	const item =
		raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
	const inputSchema = item.inputSchema ||
		item.input_schema || { type: "object", properties: {} };
	return {
		name: String(item.name || "unknown_tool"),
		title: typeof item.title === "string" ? item.title : undefined,
		description: String(item.description || ""),
		inputSchema:
			inputSchema && typeof inputSchema === "object"
				? (inputSchema as Record<string, unknown>)
				: { type: "object", properties: {} },
	};
}

export function formatMcpToolResult(result: unknown): string {
	if (result && typeof result === "object") {
		const record = result as Record<string, unknown>;
		const content = record.content;
		const isError = record.isError === true;
		if (Array.isArray(content)) {
			const text = content.map(formatContentItem).filter(Boolean).join("\n");
			return isError ? `Error: ${text || JSON.stringify(result)}` : text;
		}
		return isError
			? `Error: ${JSON.stringify(result, null, 2)}`
			: JSON.stringify(result, null, 2);
	}
	return String(result ?? "");
}

function formatContentItem(item: unknown): string {
	if (!item || typeof item !== "object") return String(item ?? "");
	const record = item as Record<string, unknown>;
	if (typeof record.text === "string") return record.text;
	if (typeof record.data === "string") return record.data;
	return JSON.stringify(record);
}

function parseSseEnvelope(
	raw: string,
	requestId: number,
): Record<string, unknown> {
	let last: Record<string, unknown> = {};
	for (const line of raw.split(/\r?\n/)) {
		if (!line.startsWith("data:")) continue;
		const data = line.slice(5).trim();
		if (!data || data === "[DONE]") continue;
		const envelope = JSON.parse(data) as Record<string, unknown>;
		if ("result" in envelope || "error" in envelope) {
			last = envelope;
			if (envelope.id === requestId) break;
		}
	}
	return last;
}

function unwrapEnvelope(
	envelope: Record<string, unknown>,
): Record<string, unknown> {
	const error = envelope.error;
	if (error && typeof error === "object") {
		const err = error as { code?: unknown; message?: unknown };
		throw new Error(
			`MCP error ${String(err.code ?? "?")}: ${String(err.message ?? error)}`,
		);
	}
	const result = envelope.result;
	return result && typeof result === "object"
		? (result as Record<string, unknown>)
		: {};
}

function expandEnvMap(values: Record<string, string>): Record<string, string> {
	const expanded: Record<string, string> = {};
	for (const [key, value] of Object.entries(values)) {
		expanded[key] = value.replace(
			/\$\{([A-Z0-9_]+)\}/gi,
			(_, name: string) => process.env[name] || "",
		);
	}
	return expanded;
}

function safeToolName(value: string): string {
	// OpenAI/Anthropic tool names allow hyphens. Preserve them because plugin
	// startup hooks refer to the Claude MCP namespace verbatim (for example
	// plugin_context-mode_context-mode); rewriting it makes valid guidance call
	// a tool name that does not exist.
	const safe = value.replace(/[^a-zA-Z0-9_-]/g, "_");
	return /^[a-zA-Z_]/.test(safe) ? safe : `_${safe}`;
}
