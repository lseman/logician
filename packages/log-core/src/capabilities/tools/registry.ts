// ── Tool registry ──────────────────────────────────────────────────────────────────
// Manages tool registration and execution. Mirrors Python ToolRegistry.
// Adds opt-in LRU result caching (tools must set cacheable: true — most tools
// have side effects or time-varying output, so caching is never applied by
// default), a per-tool execution timeout, and a result size cap so a
// misbehaving tool (MCP/extension) cannot flood the conversation context.

import { statSync } from "node:fs";
import { resolve } from "node:path";
import { CancellationScope } from "../../runtime/control/cancellation-scope.ts";
import { DEFAULT_TRUNCATION } from "../../system/types/types-config.ts";
import type {
	AskUserContext,
	Tool,
	ToolCall,
	ToolContext,
	ToolResult,
} from "../../system/types/types-messages.ts";
import { parseToolInput } from "./internal/parser.ts";
import { normalizeProviderToolSchema } from "./provider-schema.ts";
import { ToolResultCache } from "./tool-result-cache.ts";

/** Default cap on tool execution time. Tools can override via timeoutMs. */
const DEFAULT_TOOL_TIMEOUT_MS = 600_000;

/** Default cap on tool result size appended to context (~25k tokens). */
const DEFAULT_MAX_RESULT_CHARS = DEFAULT_TRUNCATION.toolResultMaxChars;

/** Name of the always-on tool used to resolve deferred tools' full schemas. */
const SEARCH_TOOLS_NAME = "search_tools";

/**
 * Tools whose full JSON schema is withheld from the provider request until
 * search_tools resolves them. MCP servers (playwright, github, ...) can each
 * contribute dozens of tools; shipping every schema on every turn regardless
 * of relevance is the actual context-bloat source, not the small built-in set.
 */
function isDeferredByDefault(tool: Tool): boolean {
	return tool.origin?.kind === "mcp";
}

function truncateResultMiddle(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	const half = Math.max(1, Math.floor((maxChars - 64) / 2));
	return (
		`${text.slice(0, half)}\n` +
		`...[tool result truncated: exceeded the ${maxChars}-char registry cap, ${text.length - half * 2} chars elided — ` +
		`this is separate from any pagination the tool itself supports; ` +
		`re-run with narrower arguments or a smaller scope]...\n` +
		`${text.slice(-half)}`
	);
}

/** Common OS error codes translated into an actionable next step for the model. */
const ERROR_CODE_HINTS: Record<string, string> = {
	ENOENT:
		"the path doesn't exist — double-check it, e.g. with list_files or find.",
	EACCES: "permission denied — check the path is within an allowed directory.",
	EISDIR: "that path is a directory, not a file.",
	ENOTDIR: "a parent segment of that path is a file, not a directory.",
	ETIMEDOUT:
		"the operation timed out — retry, or narrow the scope of the request.",
	ABORT_ERR: "the operation was aborted.",
};

/** Append an actionable hint for a recognized error code, if any. */
function describeToolError(error: Error): string {
	const code = (error as NodeJS.ErrnoException).code;
	const hint = code ? ERROR_CODE_HINTS[code] : undefined;
	return hint ? `${error.message} — ${hint}` : error.message;
}

export interface PreparedToolCall {
	call: ToolCall;
	args: Record<string, unknown>;
	error?: string;
}

export interface ToolRegistryOptions {
	cwd?: string;
	allowedPaths?: string[];
	allowAllPaths?: boolean;
	signal?: AbortSignal;
	onQuestionRequest?: (ctx: AskUserContext) => Promise<string>;
	/** Cache for tool results (P0-1). Pass null to disable caching. */
	cache?: ToolResultCache | null;
	/** Maximum entries for the default cache. */
	cacheSize?: number;
	/** TTL in milliseconds for the default cache. */
	cacheTtlMs?: number;
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
	/** Deferred tool names resolved this session — get full schemas from here on. */
	private resolvedDeferred = new Set<string>();

	constructor(options?: ToolRegistryOptions) {
		this.ctx = options || {};
		this.cwd = options?.cwd ?? process.cwd();
		this.cache =
			options && Object.hasOwn(options, "cache")
				? (options.cache ?? null)
				: new ToolResultCache(
						options?.cacheSize ?? 2000,
						options?.cacheTtlMs ?? 60_000,
					);
		this.defaultToolTimeoutMs =
			options?.defaultToolTimeoutMs ?? DEFAULT_TOOL_TIMEOUT_MS;
		this.maxResultChars = options?.maxResultChars ?? DEFAULT_MAX_RESULT_CHARS;
	}

	register(tool: Tool): void {
		this.tools.set(tool.name, tool);
		if (isDeferredByDefault(tool) && !this.tools.has(SEARCH_TOOLS_NAME)) {
			this.tools.set(SEARCH_TOOLS_NAME, this.createSearchToolsTool());
		}
	}

	registerMany(tools: Tool[]): void {
		for (const tool of tools) {
			this.register(tool);
		}
	}

	private createSearchToolsTool(): Tool {
		return {
			name: SEARCH_TOOLS_NAME,
			description:
				"Look up additional tools (e.g. from connected MCP servers) by " +
				"keyword before using them — their full schemas aren't loaded " +
				"into context until requested here. Returns matching tool names " +
				"and descriptions; once returned, call them directly like any " +
				"other tool.",
			promptSnippet:
				"find and unlock additional tools (MCP, etc.) by keyword before calling them",
			parameters: {
				type: "object",
				properties: {
					query: {
						type: "string",
						description:
							"Keywords describing the capability you need, e.g. " +
							'"browser screenshot" or "github pull request".',
					},
				},
				required: ["query"],
			},
			readOnly: true,
			execute: async args => {
				const query = typeof args.query === "string" ? args.query : "";
				const matches = this.resolveDeferredTools(query);
				if (matches.length === 0) {
					const remaining = this.deferredTools();
					return remaining.length > 0
						? `No matching tools for "${query}". Available: ${remaining
								.map(tool => tool.name)
								.join(", ")}`
						: "No additional tools available.";
				}
				return matches
					.map(tool => `${tool.name}: ${tool.description}`)
					.join("\n");
			},
		};
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

	/** Tools currently withheld from the provider request pending search_tools resolution. */
	private deferredTools(): Tool[] {
		return this.list().filter(
			tool =>
				isDeferredByDefault(tool) && !this.resolvedDeferred.has(tool.name),
		);
	}

	/**
	 * Keyword search over withheld tools' name/description/label. Matches are
	 * promoted to resolved (their full schema is included from the next
	 * toToolDefinitions() call on) and returned for the caller to report.
	 */
	resolveDeferredTools(query: string): Tool[] {
		const terms = query
			.toLowerCase()
			.split(/[^a-z0-9]+/)
			.filter(Boolean);
		const matches = this.deferredTools().filter(tool => {
			const haystack =
				`${tool.name} ${tool.label ?? ""} ${tool.description}`.toLowerCase();
			return terms.length === 0 || terms.some(term => haystack.includes(term));
		});
		for (const tool of matches) this.resolvedDeferred.add(tool.name);
		return matches;
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
		} catch (_e: unknown) {
			const error = _e as Error;
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
			this.cache !== null &&
			tool.cacheable === true &&
			!call.arguments?.includes("__nocache__");
		if (useCache) {
			const cached = this.cache?.get(
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
			const executionScope = new CancellationScope({
				operation: `tool ${call.name}`,
				parent: context?.signal ?? this.ctx.signal,
				timeoutMs,
			});
			const raw: string | ToolResult = await executionScope.run(signal =>
				tool.execute(args, {
					...this.ctx,
					...context,
					signal,
				}),
			);
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
				this.cache?.put(
					call.name,
					JSON.stringify(args),
					result.content,
					false,
					this.extractMtimeKey(call.name, args) ?? undefined,
				);
			}

			return result;
		} catch (_e: unknown) {
			const error = _e as Error;
			return {
				content: `Error executing ${call.name}: ${describeToolError(error)}`,
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
			toolName === "write_file" ||
			toolName === "list_files" ||
			toolName === "file_diff"
		) {
			const p = args.path;
			if (typeof p === "string") paths.push(p);
		}

		if (paths.length === 0) return null;

		// Build mtime-based key: "mtime:<file1>:<file2>:..."
		const mtimeParts: string[] = [];
		for (const p of paths) {
			try {
				const absolute = resolve(this.cwd, p);
				const st = statSync(absolute);
				mtimeParts.push(`${st.mtimeMs}`);
			} catch (_e: unknown) {
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
		// Deferred, unresolved tools are excluded from the request entirely —
		// their name+description already appear as plain text in the (cached)
		// system prompt tool list, and search_tools promotes them into this
		// array once the model asks for them by name/topic.
		const deferredNames = new Set(this.deferredTools().map(tool => tool.name));
		return this.list()
			.filter(tool => !deferredNames.has(tool.name))
			.map(tool => {
				const fn: Record<string, unknown> = {
					name: tool.name,
					description: tool.description,
					parameters: normalizeProviderToolSchema(tool.parameters),
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
