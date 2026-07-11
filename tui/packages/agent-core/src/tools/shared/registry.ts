// ── Tool registry ──────────────────────────────────────────────────────────────────
// Manages tool registration and execution. Mirrors Python ToolRegistry.
// Adds LRU result caching (P0-1) with mtime-based invalidation for file-based tools.

import { parseToolInput } from "./parser.ts";
import type {
	Tool,
	ToolCall,
	ToolContext,
	ToolResult,
} from "../../core/types.ts";
import type { AskUserContext } from "../../core/types/types-tools.ts";
import {
	ToolResultCache,
	computeContentFingerprint,
} from "../../core/tool-cache.ts";
import { statSync } from "node:fs";
import { resolve } from "node:path";

export interface PreparedToolCall {
	call: ToolCall;
	args: Record<string, unknown>;
	error?: string;
}

export interface ToolRegistryOptions {
	cwd?: string;
	signal?: AbortSignal;
	onQuestionRequest?: (ctx: AskUserContext) => Promise<string>;
	/** Cache for tool results (P0-1). Pass null to disable caching. */
	cache?: ToolResultCache | null;
}

export class ToolRegistry {
	private tools = new Map<string, Tool>();
	private ctx: ToolContext;
	private cache: ToolResultCache | null;
	private cwd: string;

	constructor(options?: ToolRegistryOptions) {
		this.ctx = options || {};
		this.cwd = options?.cwd ?? process.cwd();
		this.cache = options?.cache ?? new ToolResultCache(2000, 60_000);
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

		// ── P0-1: Cache lookup ───────────────────────────────────────
		if (this.cache && !call.arguments?.includes("__nocache__")) {
			const cacheArgs = JSON.stringify(args);
			let contentFp: string | undefined;
			if (call.name === "read_file") {
				const p = (args as any).path;
				if (p) {
					// Compute fingerprint from resolved path + args for semantic matching
					contentFp = computeContentFingerprint(`${p}:${cacheArgs}`);
				}
			}
			const cached = this.cache.get(call.name, cacheArgs, contentFp);
			if (cached) {
				return { content: cached.result };
			}
		}

		try {
			const raw = await tool.execute(args, { ...this.ctx, ...context });
			// Normalize the string | ToolResult union to a ToolResult.
			const result: ToolResult =
				typeof raw === "string" ? { content: raw } : raw;

			// ── P0-1: Cache successful results ───────────────────────
			if (this.cache && !result.isError) {
				const mtimeKey = this.extractMtimeKey(call.name, args);
				let contentFp: string | undefined;
				if (call.name === "read_file") {
					contentFp = computeContentFingerprint(result.content);
				}
				if (mtimeKey) {
					this.cache.put(
						call.name,
						JSON.stringify(args),
						result.content,
						false,
						mtimeKey,
						contentFp,
					);
				} else {
					this.cache.put(
						call.name,
						JSON.stringify(args),
						result.content,
						false,
						undefined,
						contentFp,
					);
				}
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
			} catch {
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
