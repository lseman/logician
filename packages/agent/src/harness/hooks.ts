// ── Hooks ─────────────────────────────────────────────────────────────────
// In-memory implementation of AgentHarness's Hooks (intercepting, can
// mutate/block) surface. The passive fire-and-forget Events surface lives in
// events.ts's HarnessEventBus instead — Hooks and Events are deliberately
// separate concepts (see agent-harness.ts's Hooks/Events interfaces).
// Event/result shapes here follow pi coding agent's harness.md §5.6 spec
// verbatim, since that's the authoritative design for this port — but unlike
// pi's spec, dispatch is plain synchronous/async in-memory composition with
// no durable replay/action-log integration (pi itself never implemented that
// part; building it would be new ground, not a port). before_tool/after_tool
// bridge directly onto agent-loop.ts's existing beforeToolCall/afterToolCall
// config callbacks, which already use the same sentinel-based (non-throwing)
// contract pi's spec describes.

import type {
	AfterToolCallContext,
	AfterToolCallResult,
	AgentMessage,
	BeforeToolCallContext,
	BeforeToolCallResult,
} from "../agent/types.ts";
import type { HookName, Hooks } from "./agent-harness.ts";
import { HarnessClosed } from "./agent-harness.ts";

/** Event payload shapes per hook, keyed by HookName. Mirrors pi harness.md §5.6's per-hook event table. */
export interface HookEventMap {
	before_run: { prompt: AgentMessage[]; systemPrompt: string };
	before_resume: Record<string, never>;
	before_run_end: { runId: string; messages: AgentMessage[] };
	transform_context: { messages: AgentMessage[] };
	before_request: { attempt: number };
	before_payload: { payload: unknown };
	after_response: { message: AgentMessage };
	before_tool: BeforeToolCallContext;
	after_tool: AfterToolCallContext;
	before_compaction: { reason: string };
	before_navigation: { targetId: string };
}

/** Result shapes per hook. Hooks that don't transform anything resolve `undefined`. */
export interface HookResultMap {
	before_run: { messages?: AgentMessage[]; systemPrompt?: string } | undefined;
	before_resume: undefined;
	before_run_end: { followUp?: AgentMessage[] } | undefined;
	transform_context: { messages: AgentMessage[] };
	before_request: undefined;
	before_payload: { payload: unknown } | undefined;
	after_response: { message: AgentMessage } | undefined;
	before_tool: BeforeToolCallResult | { args?: unknown } | undefined;
	after_tool: AfterToolCallResult | undefined;
	before_compaction: undefined;
	before_navigation: undefined;
}

type HookHandler<K extends HookName = HookName> = (
	event: HookEventMap[K],
) => HookResultMap[K] | Promise<HookResultMap[K]>;

interface RegisteredHandler {
	id: string;
	handler: (event: unknown) => unknown | Promise<unknown>;
}

/**
 * In-memory hook dispatcher. Handlers for a given hook name run in registration order, each
 * seeing the previous handler's output where the hook chains (transform_context, before_tool
 * args rewrites). A thrown handler is reported and skipped — except before_tool, which fails
 * closed and blocks the tool, per pi's spec (harness.md:2562-2567).
 */
export class HookRegistry implements Hooks {
	private readonly handlers = new Map<string, RegisteredHandler[]>();
	private readonly isClosed: () => boolean;
	private nextId = 0;

	constructor(isClosed: () => boolean = () => false) {
		this.isClosed = isClosed;
	}

	on<K extends HookName>(
		name: K,
		handler: HookHandler<K>,
		options?: { id?: string },
	): () => void;
	on(
		name: HookName | string,
		handler: (event: unknown) => unknown | Promise<unknown>,
		options?: { id?: string },
	): () => void {
		if (this.isClosed()) throw new HarnessClosed();
		const id = options?.id ?? `hook_${this.nextId++}`;
		const list = this.handlers.get(name) ?? [];
		list.push({ id, handler });
		this.handlers.set(name, list);
		return () => {
			const current = this.handlers.get(name);
			if (!current) return;
			this.handlers.set(
				name,
				current.filter(entry => entry.id !== id),
			);
		};
	}

	/** True if any handler is registered for `name`. Lets callers skip building an event when unused. */
	has(name: HookName): boolean {
		return (this.handlers.get(name)?.length ?? 0) > 0;
	}

	/**
	 * Dispatch transform_context: each handler sees the previous handler's rewritten messages.
	 * A throwing handler is skipped (its input passes through unchanged).
	 */
	async transformContext(messages: AgentMessage[]): Promise<AgentMessage[]> {
		let current = messages;
		for (const { handler } of this.handlers.get("transform_context") ?? []) {
			try {
				const result = (await handler({
					messages: current,
				})) as HookResultMap["transform_context"];
				if (result?.messages) current = result.messages;
			} catch {
				// non-blocking hooks fail open: skip this handler, keep the chain going
			}
		}
		return current;
	}

	/**
	 * Dispatch before_tool: args rewrites chain across handlers; the first `block` result is
	 * terminal. A thrown handler fails closed and blocks the tool (unlike other hooks).
	 */
	async beforeToolCall(
		context: BeforeToolCallContext,
	): Promise<BeforeToolCallResult | undefined> {
		let args = context.args;
		for (const { handler } of this.handlers.get("before_tool") ?? []) {
			let result: HookResultMap["before_tool"];
			try {
				result = (await handler({
					...context,
					args,
				})) as HookResultMap["before_tool"];
			} catch (error) {
				return {
					block: true,
					reason: error instanceof Error ? error.message : String(error),
				};
			}
			if (!result) continue;
			if ("block" in result && result.block) return result;
			if ("args" in result && result.args !== undefined) args = result.args;
		}
		return args === context.args
			? undefined
			: ({ args } as BeforeToolCallResult & { args: unknown });
	}

	/**
	 * Dispatch after_tool: each handler's overrides layer onto the previous result field-by-field.
	 * A throwing handler is skipped (its overrides are dropped).
	 */
	async afterToolCall(
		context: AfterToolCallContext,
	): Promise<AfterToolCallResult | undefined> {
		let overrides: AfterToolCallResult | undefined;
		for (const { handler } of this.handlers.get("after_tool") ?? []) {
			try {
				const result = (await handler({
					...context,
					result: overrides
						? { ...context.result, ...overrides }
						: context.result,
					isError: overrides?.isError ?? context.isError,
				})) as HookResultMap["after_tool"];
				if (result) overrides = { ...overrides, ...result };
			} catch {
				// non-blocking: skip this handler's overrides
			}
		}
		return overrides;
	}
}
