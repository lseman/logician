// ── Tool registry ──────────────────────────────────────────────────────────────────
// Manages tool registration and execution. Mirrors Python ToolRegistry.

import { parseToolInput } from "./parser.ts";
import type {
	Tool,
	ToolCall,
	ToolContext,
	ToolResult,
} from "../../core/types.ts";

export interface PreparedToolCall {
	call: ToolCall;
	args: Record<string, unknown>;
	error?: string;
}

export class ToolRegistry {
	private tools = new Map<string, Tool>();
	private ctx: ToolContext;

	constructor(ctx?: ToolContext) {
		this.ctx = ctx || {};
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

		try {
			const args = preparedArgs ?? this.prepare(call).args;
			const raw = await tool.execute(args, { ...this.ctx, ...context });
			// Normalize the string | ToolResult union to a ToolResult.
			return typeof raw === "string" ? { content: raw } : raw;
		} catch (e: unknown) {
			const error = e as Error;
			return { content: `Error executing ${call.name}: ${error.message}` };
		}
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
