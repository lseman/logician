// ── Tool registry ─────────────────────────────────────────────────────────
// Register/list/enable-disable container for AgentHarness's HarnessTool[].
// Unlike old agent-core's ToolRegistry (packages/agent-core/src/harness/tools/registry.ts),
// this does not reimplement execution, timeouts, or abort wiring — beta's
// agent-loop.ts (agent/agent-loop.ts: prepareToolCall/executePreparedToolCall)
// already owns per-call abort-signal propagation, thrown-error-to-error-result
// conversion, and per-tool executionMode (sequential/parallel). The registry's
// job is purely bookkeeping (which tools exist, which are active) plus one
// safety property agent-loop.ts doesn't provide: a cap on result size so a
// misbehaving tool can't flood the conversation context.

import type {
	AgentTool,
	AgentToolResult,
	AgentToolUpdateCallback,
} from "../../agent/types.ts";
import type { TextContent } from "../../ai/types.ts";
import type { HarnessTool } from "../agent-harness.ts";

/** Default cap on tool result content length appended to context (~25k tokens worth of chars). */
export const DEFAULT_MAX_RESULT_CHARS = 100_000;

function textLength(content: AgentToolResult<unknown>["content"]): number {
	let length = 0;
	for (const block of content) {
		if (block.type === "text") length += block.text.length;
	}
	return length;
}

function truncateResultMiddle(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	const half = Math.max(1, Math.floor((maxChars - 64) / 2));
	return `${text.slice(0, half)}\n...[tool result truncated: ${text.length - half * 2} chars elided]...\n${text.slice(-half)}`;
}

/** Truncate the text content of a tool result in place, preserving non-text blocks (e.g. images) untouched. */
function capResultSize<T>(
	result: AgentToolResult<T>,
	maxChars: number,
): AgentToolResult<T> {
	if (maxChars <= 0 || textLength(result.content) <= maxChars) return result;
	let remaining = maxChars;
	const content: AgentToolResult<T>["content"] = result.content.map(block => {
		if (block.type !== "text") return block;
		if (remaining <= 0) return { type: "text", text: "" } satisfies TextContent;
		const truncated = truncateResultMiddle(block.text, remaining);
		remaining -= truncated.length;
		return { type: "text", text: truncated } satisfies TextContent;
	});
	return { ...result, content };
}

export interface ToolRegistryOptions {
	/** Cap on a single tool result's text content, in characters. 0 disables. Default 100_000. */
	maxResultChars?: number;
}

/**
 * Register/list/enable-disable container for HarnessTool[]. Feeds AgentHarness's
 * `tools`/`activeToolNames` constructor options; does not itself execute tools —
 * that stays agent-loop.ts's job once tools are passed into AgentHarness.
 */
export class ToolRegistry {
	private readonly tools = new Map<string, HarnessTool>();
	private readonly active = new Set<string>();
	private readonly maxResultChars: number;

	constructor(options: ToolRegistryOptions = {}) {
		this.maxResultChars = options.maxResultChars ?? DEFAULT_MAX_RESULT_CHARS;
	}

	/** Register a tool. Active by default. Wraps execute() with the result-size cap. */
	register(tool: AgentTool, options: { active?: boolean } = {}): void {
		const wrapped: HarnessTool = {
			...tool,
			execute: async (
				toolCallId: string,
				params: unknown,
				signal?: AbortSignal,
				onUpdate?: AgentToolUpdateCallback<unknown>,
			) => {
				const result = await tool.execute(toolCallId, params, signal, onUpdate);
				return capResultSize(result, this.maxResultChars);
			},
		};
		this.tools.set(tool.name, wrapped);
		if (options.active ?? true) this.active.add(tool.name);
		else this.active.delete(tool.name);
	}

	registerMany(tools: AgentTool[], options: { active?: boolean } = {}): void {
		for (const tool of tools) this.register(tool, options);
	}

	unregister(name: string): void {
		this.tools.delete(name);
		this.active.delete(name);
	}

	has(name: string): boolean {
		return this.tools.has(name);
	}

	get(name: string): HarnessTool | undefined {
		return this.tools.get(name);
	}

	/** All registered tools, active or not. */
	list(): HarnessTool[] {
		return [...this.tools.values()];
	}

	/** Names of active (enabled) tools. */
	activeToolNames(): string[] {
		return [...this.active].filter(name => this.tools.has(name));
	}

	/** Only the currently-active tools. */
	activeTools(): HarnessTool[] {
		return this.activeToolNames()
			.map(name => this.tools.get(name))
			.filter((tool): tool is HarnessTool => tool !== undefined);
	}

	enable(name: string): void {
		if (this.tools.has(name)) this.active.add(name);
	}

	disable(name: string): void {
		this.active.delete(name);
	}

	setActive(names: string[]): void {
		this.active.clear();
		for (const name of names) {
			if (this.tools.has(name)) this.active.add(name);
		}
	}

	/** Convenience: the { tools, activeToolNames } pair AgentHarnessOptions expects. */
	toHarnessOptions(): { tools: HarnessTool[]; activeToolNames: string[] } {
		return { tools: this.list(), activeToolNames: this.activeToolNames() };
	}
}
