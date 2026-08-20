// ── Tool registry ──────────────────────────────────────────────────────────────────
// Manages tool registration and execution. Mirrors Python ToolRegistry.
// Adds a per-tool execution timeout and a result size cap so a misbehaving
// tool (MCP/extension) cannot flood the conversation context.

import type {
	Tool,
	ToolCall,
	ToolContext,
	ToolResult,
} from "../../types/index.ts";
import { DEFAULT_TRUNCATION } from "../../types/types-config.ts";
import type { AskUserContext } from "../../types/types-messages.ts";
import { withTimeout } from "./async-utils.ts";
import { parseToolInput } from "./parser.ts";

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
	allowedPaths?: string[];
	allowAllPaths?: boolean;
	signal?: AbortSignal;
	onQuestionRequest?: (ctx: AskUserContext) => Promise<string>;
	/** Per-tool execution timeout (default 10 min). 0 disables. Tools override via timeoutMs. */
	defaultToolTimeoutMs?: number;
	/** Cap on result content appended to context (default 100k chars). 0 disables. */
	maxResultChars?: number;
}

export class ToolRegistry {
	private tools = new Map<string, Tool>();
	private ctx: ToolContext;
	private cwd: string;
	private defaultToolTimeoutMs: number;
	private maxResultChars: number;

	constructor(options?: ToolRegistryOptions) {
		this.ctx = options || {};
		this.cwd = options?.cwd ?? process.cwd();
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

		try {
			const timeoutMs =
				tool.resolveTimeoutMs?.(args) ??
				tool.timeoutMs ??
				this.defaultToolTimeoutMs;
			const parentSignal = context?.signal ?? this.ctx.signal;
			const executionController = new AbortController();
			const abortFromParent = () =>
				executionController.abort(parentSignal?.reason);
			if (parentSignal?.aborted) abortFromParent();
			else
				parentSignal?.addEventListener("abort", abortFromParent, {
					once: true,
				});
			let raw: string | ToolResult;
			try {
				const run = tool.execute(args, {
					...this.ctx,
					...context,
					signal: executionController.signal,
				});
				raw =
					timeoutMs > 0
						? await withTimeout(run, timeoutMs, () =>
								executionController.abort(
									new Error(`Tool execution timed out after ${timeoutMs}ms`),
								),
							)
						: await run;
			} finally {
				parentSignal?.removeEventListener("abort", abortFromParent);
			}
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

			return result;
		} catch (_e: unknown) {
			const error = _e as Error;
			return {
				content: `Error executing ${call.name}: ${error.message}`,
				isError: true,
			};
		}
	}

	toToolDefinitions(): Record<string, unknown>[] {
		return this.list().map(tool => {
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
