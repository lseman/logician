// ── Tool registry ──────────────────────────────────────────────────────────────────
// Manages tool registration and execution. Mirrors Python ToolRegistry.
// Adds opt-in LRU result caching (tools must set cacheable: true — most tools
// have side effects or time-varying output, so caching is never applied by
// default), a per-tool execution timeout, and a result size cap so a
// misbehaving tool (MCP/extension) cannot flood the conversation context.

import { parseToolInput } from "./parser.ts";
import type {
	Tool,
	ToolCall,
	ToolContext,
	ToolResult,
} from "../../core/types.ts";
import type { AskUserContext } from "../../core/types/types-tools.ts";
import { ToolResultCache } from "../../core/tool-cache.ts";
import { withTimeout } from "./async-utils.ts";
import { statSync } from "node:fs";
import { resolve } from "node:path";
import { DEFAULT_TRUNCATION } from "../../core/types/types-truncation.ts";

/** Default cap on tool execution time. Tools can override via timeoutMs. */
const DEFAULT_TOOL_TIMEOUT_MS = 600_000;

/** Default cap on tool result size appended to context (~25k tokens). */
const DEFAULT_MAX_RESULT_CHARS = DEFAULT_TRUNCATION.toolResultMaxChars;

function truncateResultMiddle(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	const half = Math.max(1, Math.floor((maxChars - 64) / 2));
	return (
		`${text.slice(0, half)}\n` +
		`...[tool result truncated: ${text.length - half * 2} chars elided]...\n` +
		`${text.slice(-half)}`
	);
}

export interface PreparedToolCall {
	call: ToolCall;
	args: Record<string, unknown>;
	error?: string;
}

export interface ToolRegistryOptions {
	cwd?: string;
	allowAllPaths?: boolean;
	signal?: AbortSignal;
	onQuestionRequest?: (ctx: AskUserContext) => Promise<string>;
	/** Cache for tool results (P0-1). Pass null to disable caching. */
	cache?: ToolResultCache | null;
	/** Per-tool execution timeout (default 10 min). 0 disables. Tools override via timeoutMs. */
	defaultToolTimeoutMs?: number;
	/** Cap on result content appended to context (default 100k chars). 0 disables. */
	maxResultChars?: number;
}

export class ToolRegistry {
	private tools = new Map<string, Tool>();
	private ctx: ToolContext;
	private cache: ToolResultCache | null;
	private cwd: string;
	private defaultToolTimeoutMs: number;
	private maxResultChars: number;

	constructor(options?: ToolRegistryOptions) {
		this.ctx = options || {};
		this.cwd = options?.cwd ?? process.cwd();
		this.cache = options?.cache ?? new ToolResultCache(2000, 60_000);
		this.defaultToolTimeoutMs =
			options?.defaultToolTimeoutMs ?? DEFAULT_TOOL_TIMEOUT_MS;
		this.maxResultChars = options?.maxResultChars ?? DEFAULT_MAX_RESULT_CHARS;
	}

	register(tool: Tool): void {
		this.tools.set(tool.name, tool);
	}

	registerMany(tools: Tool[]): void {
		for (const tool of tools) {
			this.register(tool);
		}
	}

	unregister(name: string): void {
		this.tools.delete(name);
	}

	has(name: string): boolean {
		return this.tools.has(name);
	}

	get(name: string): Tool | undefined {
		return this.tools.get(name);
	}

	list(): Tool[] {
		return Array.from(this.tools.values());
	}

	prepare(call: ToolCall): PreparedToolCall {
		const tool = this.tools.get(call.name);
		if (!tool) {
			return {
				call,
				args: parseToolInput(call.arguments),
				error: `Error: Unknown tool: ${call.name}`,
			};
		}

		try {
			let args = parseToolInput(call.arguments);
			if (tool.prepareArguments) {
				args = tool.prepareArguments(args);
			}
			return {
				call: { ...call, arguments: JSON.stringify(args) },
				args,
			};
		} catch (e: unknown) {
			const error = e as Error;
			return {
				call,
				args: parseToolInput(call.arguments),
				error: `Error preparing ${call.name}: ${error.message}`,
			};
		}
	}

	async execute(
		call: ToolCall,
		context?: ToolContext,
		preparedArgs?: Record<string, unknown>,
	): Promise<ToolResult> {
		const tool = this.tools.get(call.name);
		if (!tool) {
			return { content: `Error: Unknown tool: ${call.name}` };
		}

		const args = preparedArgs ?? this.prepare(call).args;

		// ── Cache lookup — opt-in only. Tools with side effects (edits, bash,
		// read_file's read-tracking) or time-varying output must never be
		// served from cache, so caching requires an explicit cacheable: true.
		const useCache =
			this.cache != null &&
			tool.cacheable === true &&
			!call.arguments?.includes("__nocache__");
		if (useCache) {
			const cached = this.cache!.get(
				call.name,
				JSON.stringify(args),
				this.extractMtimeKey(call.name, args) ?? undefined,
			);
			if (cached) {
				return { content: cached.result };
			}
		}

		try {
			const timeoutMs =
				tool.resolveTimeoutMs?.(args) ??
				tool.timeoutMs ??
				this.defaultToolTimeoutMs;
			const run = tool.execute(args, { ...this.ctx, ...context });
			const raw = timeoutMs > 0 ? await withTimeout(run, timeoutMs) : await run;
			// Normalize the string | ToolResult union to a ToolResult.
			const result: ToolResult =
				typeof raw === "string" ? { content: raw } : raw;

			// Cap result size so a misbehaving tool cannot flood the context.
			if (
				this.maxResultChars > 0 &&
				result.content.length > this.maxResultChars
			) {
				result.content = truncateResultMiddle(
					result.content,
					this.maxResultChars,
				);
			}

			if (useCache && !result.isError) {
				this.cache!.put(
					call.name,
					JSON.stringify(args),
					result.content,
					false,
					this.extractMtimeKey(call.name, args) ?? undefined,
				);
			}

			return result;
		} catch (e: unknown) {
			const error = e as Error;
			return {
				content: `Error executing ${call.name}: ${error.message}`,
				isError: true,
			};
		}
	}

	/** Extract mtime-based cache key for file-based tools. Returns null for non-file tools. */
	private extractMtimeKey(
		toolName: string,
		args: Record<string, unknown>,
	): string | null {
		const paths: string[] = [];

		if (
			toolName === "read_file" ||
			toolName === "edit_file" ||
			toolName === "write_file"
		) {
			const p = (args as any).path;
			if (p) paths.push(p);
		} else if (toolName === "list_files") {
			const p = (args as any).path;
			if (p) paths.push(p);
		} else if (toolName === "file_diff") {
			const p = (args as any).path;
			if (p) paths.push(p);
		}

		if (paths.length === 0) return null;

		// Build mtime-based key: "mtime:<file1>:<file2>:..."
		const mtimeParts: string[] = [];
		for (const p of paths) {
			try {
				const absolute = resolve(this.cwd, p);
				const st = statSync(absolute);
				mtimeParts.push(`${st.mtimeMs}`);
			} catch (e: unknown) {
				// File doesn't exist — use "0" as sentinel
				mtimeParts.push("0");
			}
		}
		return `mtime:${mtimeParts.join(":")}`;
	}

	/** Get cache statistics for observability. */
	getCacheStats() {
		return this.cache ? this.cache.stats() : null;
	}

	/** Clear the tool result cache. */
	clearCache() {
		this.cache?.clear();
	}

	toToolDefinitions(): Record<string, unknown>[] {
		return this.list().map((tool) => {
			const fn: Record<string, unknown> = {
				name: tool.name,
				description: tool.description,
				parameters: tool.parameters,
			};
			if (tool.promptSnippet) fn.promptSnippet = tool.promptSnippet;
			if (tool.label) fn.label = tool.label;
			return { type: "function" as const, function: fn };
		});
	}

	/** Return tool info for system prompt tool list section. */
	toolSnippets(): Record<string, string> {
		const snippets: Record<string, string> = {};
		for (const tool of this.list()) {
			if (tool.promptSnippet) {
				snippets[tool.name] = tool.promptSnippet;
			}
		}
		return snippets;
	}

	/** Return tool-level guidelines for system prompt. */
	toolGuidelines(): string[] {
		const guidelines: string[] = [];
		for (const tool of this.list()) {
			if (tool.promptGuidelines) {
				guidelines.push(...tool.promptGuidelines);
			}
		}
		return guidelines;
	}
}
