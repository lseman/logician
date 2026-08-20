// ── Typed hook bus ─────────────────────────────────────────────────────────
// Unifies the single-handler contract hooks into one multi-handler bus with
// per-event reducer semantics, mirroring pi's hook design
// Multiple extensions register handlers for the
// same event and compose deterministically:
//
//   beforeToolCall   → early-block: first {content} short-circuits; {args}
//                      rewrites thread to later handlers.
//   afterToolCall    → patch-accumulate: each handler sees the prior patch;
//                      later non-undefined fields win.
//   prepareNextTurn  → transform: messages thread through all handlers.
//   shouldStopAfterTurn → first-true wins.
//
// The bus emits a single `AgentHooks` via `toHooks()`, so the runner still
// calls one handler per event. Handlers can be scoped with source metadata so
// a failing extension is identifiable; a thrown handler is skipped and
// reported via `onError` rather than aborting the chain.

import type {
	AfterProviderResponseContext,
	AfterToolCallContext,
	AfterToolCallResult,
	AgentHooks,
	BeforeAgentStartContext,
	BeforeAgentStartResult,
	BeforeCompactContext,
	BeforeCompactResult,
	BeforeProviderPayloadContext,
	BeforeProviderPayloadResult,
	BeforeProviderRequestContext,
	BeforeProviderRequestResult,
	BeforeToolCallContext,
	BeforeToolCallResult,
	GetFollowUpMessagesContext,
	GetSteeringMessagesContext,
	Message,
	PrepareNextTurnContext,
	PrepareNextTurnResult,
	ShouldStopAfterTurnContext,
	TransformContext,
	TransformContextResult,
} from "../types/index.ts";

export type HookEventName = keyof AgentHooks;

export interface HookRegistration {
	/** Stable identity used for diagnostics and duplicate detection. */
	id?: string;
	source?: string;
}

export interface HookBusOptions {
	onError?: (error: Error, event: HookEventName, source?: string) => void;
}

interface Entry<H> {
	handler: H;
	id: string;
	source?: string;
}

type BeforeHandler = NonNullable<AgentHooks["beforeToolCall"]>;
type AfterHandler = NonNullable<AgentHooks["afterToolCall"]>;
type PrepareHandler = NonNullable<AgentHooks["prepareNextTurn"]>;
type TransformHandler = NonNullable<AgentHooks["transformContext"]>;
type ProviderRequestHandler = NonNullable<AgentHooks["beforeProviderRequest"]>;
type ProviderPayloadHandler = NonNullable<AgentHooks["beforeProviderPayload"]>;
type AfterProviderHandler = NonNullable<AgentHooks["afterProviderResponse"]>;
type StopHandler = NonNullable<AgentHooks["shouldStopAfterTurn"]>;
type SteeringHandler = NonNullable<AgentHooks["getSteeringMessages"]>;
type FollowUpHandler = NonNullable<AgentHooks["getFollowUpMessages"]>;
type AgentStartHandler = NonNullable<AgentHooks["beforeAgentStart"]>;
type CompactHandler = NonNullable<AgentHooks["beforeCompact"]>;

export class HookBus {
	private before: Entry<BeforeHandler>[] = [];
	private after: Entry<AfterHandler>[] = [];
	private prepare: Entry<PrepareHandler>[] = [];
	private transform: Entry<TransformHandler>[] = [];
	private providerRequest: Entry<ProviderRequestHandler>[] = [];
	private providerPayload: Entry<ProviderPayloadHandler>[] = [];
	private afterProvider: Entry<AfterProviderHandler>[] = [];
	private stop: Entry<StopHandler>[] = [];
	private steering: Entry<SteeringHandler>[] = [];
	private followUp: Entry<FollowUpHandler>[] = [];
	private agentStart: Entry<AgentStartHandler>[] = [];
	private compact: Entry<CompactHandler>[] = [];
	private nextAnonymousId = 0;

	private onError?: HookBusOptions["onError"];

	constructor(options: HookBusOptions = {}) {
		this.onError = options.onError;
	}

	// Register one handler for an event, run in registration order. Returns an unsubscribe function.
	on<E extends HookEventName>(
		event: E,
		handler: NonNullable<AgentHooks[E]>,
		reg: HookRegistration = {},
	): () => void {
		const list = this.listFor(event) as Entry<AgentHooks[E]>[];
		const id = reg.id ?? `${String(event)}#${++this.nextAnonymousId}`;
		if (this.hasHandlerId(id))
			throw new Error(`Duplicate hook handler id: ${id}`);
		const entry = { handler, id, source: reg.source };
		list.push(entry);
		return () => {
			const i = list.indexOf(entry);
			if (i >= 0) list.splice(i, 1);
		};
	}

	// Register a whole AgentHooks object at once (each present handler).
	register(hooks: AgentHooks, reg: HookRegistration = {}): () => void {
		const offs: Array<() => void> = [];
		if (hooks.beforeAgentStart)
			offs.push(this.on("beforeAgentStart", hooks.beforeAgentStart, reg));
		if (hooks.beforeToolCall)
			offs.push(this.on("beforeToolCall", hooks.beforeToolCall, reg));
		if (hooks.afterToolCall)
			offs.push(this.on("afterToolCall", hooks.afterToolCall, reg));
		if (hooks.prepareNextTurn)
			offs.push(this.on("prepareNextTurn", hooks.prepareNextTurn, reg));
		if (hooks.transformContext)
			offs.push(this.on("transformContext", hooks.transformContext, reg));
		if (hooks.beforeProviderRequest)
			offs.push(
				this.on("beforeProviderRequest", hooks.beforeProviderRequest, reg),
			);
		if (hooks.beforeProviderPayload)
			offs.push(
				this.on("beforeProviderPayload", hooks.beforeProviderPayload, reg),
			);
		if (hooks.afterProviderResponse)
			offs.push(
				this.on("afterProviderResponse", hooks.afterProviderResponse, reg),
			);
		if (hooks.shouldStopAfterTurn)
			offs.push(this.on("shouldStopAfterTurn", hooks.shouldStopAfterTurn, reg));
		if (hooks.getSteeringMessages)
			offs.push(this.on("getSteeringMessages", hooks.getSteeringMessages, reg));
		if (hooks.getFollowUpMessages)
			offs.push(this.on("getFollowUpMessages", hooks.getFollowUpMessages, reg));
		if (hooks.beforeCompact)
			offs.push(this.on("beforeCompact", hooks.beforeCompact, reg));
		return () =>
			offs.forEach(off => {
				off();
			});
	}

	// Single composed AgentHooks for the runner. Each event runs its
	// reducer over all registered handlers.
	toHooks(): AgentHooks {
		return {
			beforeAgentStart: (ctx, signal) => this.runAgentStart(ctx, signal),
			beforeToolCall: (ctx, signal) => this.runBefore(ctx, signal),
			afterToolCall: (ctx, signal) => this.runAfter(ctx, signal),
			prepareNextTurn: (ctx, signal) => this.runPrepare(ctx, signal),
			transformContext: (ctx, signal) => this.runTransform(ctx, signal),
			beforeProviderRequest: (ctx, signal) =>
				this.runProviderRequest(ctx, signal),
			beforeProviderPayload: (ctx, signal) =>
				this.runProviderPayload(ctx, signal),
			afterProviderResponse: (ctx, signal) =>
				this.runAfterProvider(ctx, signal),
			shouldStopAfterTurn: (ctx, signal) => this.runStop(ctx, signal),
			getSteeringMessages: (ctx, signal) => this.runSteering(ctx, signal),
			getFollowUpMessages: (ctx, signal) => this.runFollowUp(ctx, signal),
			beforeCompact: (ctx, signal) => this.runCompact(ctx, signal),
		};
	}

	// ── Reducers ───────────────────────────────────────────────────────────
	private async runAgentStart(
		ctx: BeforeAgentStartContext,
		signal?: AbortSignal,
	): Promise<BeforeAgentStartResult | undefined> {
		let messages = ctx.messages;
		let systemPrompt = ctx.systemPrompt;
		let changed = false;
		for (const { handler, id, source } of this.agentStart) {
			const result = await this.guard(
				scopedSignal =>
					handler({ ...ctx, messages, systemPrompt }, scopedSignal),
				"beforeAgentStart",
				source,
				id,
				signal,
			);
			if (result?.messages) {
				messages = result.messages;
				changed = true;
			}
			if (result?.systemPrompt !== undefined) {
				systemPrompt = result.systemPrompt;
				changed = true;
			}
		}
		return changed ? { messages, systemPrompt } : undefined;
	}

	private async runBefore(
		ctx: BeforeToolCallContext,
		signal?: AbortSignal,
	): Promise<BeforeToolCallResult | undefined> {
		if (!this.before.length) return undefined;
		let current = ctx;
		let rewritten: Record<string, unknown> | undefined;
		for (const { handler, id, source } of this.before) {
			const r = await this.guard(
				scopedSignal => handler(current, scopedSignal),
				"beforeToolCall",
				source,
				id,
				signal,
			);
			if (!r) continue;
			// A content result short-circuits: tool is not run.
			if (r.content !== undefined) {
				return rewritten ? { ...r, args: r.args ?? rewritten } : r;
			}
			if (r.args !== undefined) {
				rewritten = r.args;
				current = { ...current, args: r.args };
			}
		}
		return rewritten ? { args: rewritten } : undefined;
	}

	private async runAfter(
		ctx: AfterToolCallContext,
		signal?: AbortSignal,
	): Promise<AfterToolCallResult | undefined> {
		if (!this.after.length) return undefined;
		let current = ctx;
		let modified = false;
		let terminate = false;
		for (const { handler, id, source } of this.after) {
			const r = await this.guard(
				scopedSignal => handler(current, scopedSignal),
				"afterToolCall",
				source,
				id,
				signal,
			);
			if (!r) continue;
			// terminate from ANY handler wins; the loop applies its
			// all-tools-in-batch gate.
			terminate = terminate || r.terminate === true;
			current = {
				...current,
				result: r.content ?? current.result,
				isError: r.isError ?? current.isError,
			};
			modified = true;
		}
		return modified
			? { content: current.result, isError: current.isError, terminate }
			: undefined;
	}

	private async runPrepare(
		ctx: PrepareNextTurnContext,
		signal?: AbortSignal,
	): Promise<PrepareNextTurnResult | undefined> {
		if (!this.prepare.length) return undefined;
		let messages = ctx.messages;
		for (const { handler, id, source } of this.prepare) {
			const r = await this.guard(
				scopedSignal => handler({ ...ctx, messages }, scopedSignal),
				"prepareNextTurn",
				source,
				id,
				signal,
			);
			if (r?.messages) messages = r.messages;
		}
		return messages === ctx.messages ? undefined : { messages };
	}

	private async runTransform(
		ctx: TransformContext,
		signal?: AbortSignal,
	): Promise<TransformContextResult | undefined> {
		if (!this.transform.length) return undefined;
		let messages = ctx.messages;
		for (const { handler, id, source } of this.transform) {
			const r = await this.guard(
				scopedSignal => handler({ ...ctx, messages }, scopedSignal),
				"transformContext",
				source,
				id,
				signal,
			);
			if (r?.messages) messages = r.messages;
		}
		return messages === ctx.messages ? undefined : { messages };
	}

	private async runProviderRequest(
		ctx: BeforeProviderRequestContext,
		signal?: AbortSignal,
	): Promise<BeforeProviderRequestResult | undefined> {
		if (!this.providerRequest.length) return undefined;
		const collectedHeaders: Record<string, string | undefined> = {};
		let timeoutMs: number | undefined;
		let maxRetries: number | undefined;
		let cacheRetention: string | undefined;
		let metadata: Record<string, unknown> | undefined;
		let transport: string | undefined;
		for (const { handler, id, source } of this.providerRequest) {
			const r = await this.guard(
				scopedSignal => handler(ctx, scopedSignal),
				"beforeProviderRequest",
				source,
				id,
				signal,
			);
			if (!r) continue;
			if (r.headers) {
				for (const [k, v] of Object.entries(r.headers)) {
					collectedHeaders[k] = v;
				}
			}
			if (r.timeoutMs !== undefined) timeoutMs = r.timeoutMs;
			if (r.maxRetries !== undefined) maxRetries = r.maxRetries;
			if (r.cacheRetention !== undefined) cacheRetention = r.cacheRetention;
			if (r.metadata !== undefined)
				metadata = { ...(metadata ?? {}), ...r.metadata };
			if (r.transport !== undefined) transport = r.transport;
		}
		return Object.keys(collectedHeaders).length ||
			timeoutMs !== undefined ||
			maxRetries !== undefined ||
			cacheRetention !== undefined ||
			metadata !== undefined ||
			transport !== undefined
			? {
					headers: collectedHeaders,
					timeoutMs,
					maxRetries,
					cacheRetention,
					metadata,
					transport,
				}
			: undefined;
	}

	private async runProviderPayload(
		ctx: BeforeProviderPayloadContext,
		signal?: AbortSignal,
	): Promise<BeforeProviderPayloadResult | undefined> {
		if (!this.providerPayload.length) return undefined;
		let payload = ctx.payload;
		for (const { handler, id, source } of this.providerPayload) {
			const r = await this.guard(
				scopedSignal => handler({ ...ctx, payload }, scopedSignal),
				"beforeProviderPayload",
				source,
				id,
				signal,
			);
			if (r?.payload) payload = r.payload;
		}
		return payload === ctx.payload ? undefined : { payload };
	}

	private async runAfterProvider(
		ctx: AfterProviderResponseContext,
		signal?: AbortSignal,
	): Promise<void> {
		for (const { handler, id, source } of this.afterProvider) {
			await this.guard(
				scopedSignal => handler(ctx, scopedSignal),
				"afterProviderResponse",
				source,
				id,
				signal,
			);
		}
	}

	private async runStop(
		ctx: ShouldStopAfterTurnContext,
		signal?: AbortSignal,
	): Promise<boolean | undefined> {
		for (const { handler, id, source } of this.stop) {
			const r = await this.guard(
				scopedSignal => handler(ctx, scopedSignal),
				"shouldStopAfterTurn",
				source,
				id,
				signal,
			);
			if (r === true) return true;
		}
		return undefined;
	}

	private async runSteering(
		ctx: GetSteeringMessagesContext,
		signal?: AbortSignal,
	): Promise<Message[] | undefined> {
		const out: Message[] = [];
		for (const { handler, id, source } of this.steering) {
			const r = await this.guard(
				scopedSignal =>
					handler(
						{ ...ctx, messages: [...ctx.messages, ...out] },
						scopedSignal,
					),
				"getSteeringMessages",
				source,
				id,
				signal,
			);
			if (r?.length) out.push(...r);
		}
		return out.length ? out : undefined;
	}

	private async runFollowUp(
		ctx: GetFollowUpMessagesContext,
		signal?: AbortSignal,
	): Promise<Message[] | undefined> {
		const out: Message[] = [];
		for (const { handler, id, source } of this.followUp) {
			const r = await this.guard(
				scopedSignal =>
					handler(
						{
							...ctx,
							messages: [...ctx.messages, ...out],
						},
						scopedSignal,
					),
				"getFollowUpMessages",
				source,
				id,
				signal,
			);
			if (r?.length) out.push(...r);
		}
		return out.length ? out : undefined;
	}

	private async runCompact(
		ctx: BeforeCompactContext,
		signal?: AbortSignal,
	): Promise<BeforeCompactResult | undefined> {
		let summary: string | undefined;
		for (const { handler, id, source } of this.compact) {
			const result = await this.guard(
				scopedSignal => handler(ctx, scopedSignal),
				"beforeCompact",
				source,
				id,
				signal,
			);
			if (result?.cancel) return { cancel: true };
			if (result?.summary !== undefined) summary = result.summary;
		}
		return summary === undefined ? undefined : { summary };
	}

	// ── Internals ──────────────────────────────────────────────────────────

	private listFor(event: HookEventName): Entry<unknown>[] {
		switch (event) {
			case "beforeAgentStart":
				return this.agentStart as Entry<unknown>[];
			case "beforeToolCall":
				return this.before as Entry<unknown>[];
			case "afterToolCall":
				return this.after as Entry<unknown>[];
			case "prepareNextTurn":
				return this.prepare as Entry<unknown>[];
			case "transformContext":
				return this.transform as Entry<unknown>[];
			case "beforeProviderRequest":
				return this.providerRequest as Entry<unknown>[];
			case "beforeProviderPayload":
				return this.providerPayload as Entry<unknown>[];
			case "afterProviderResponse":
				return this.afterProvider as Entry<unknown>[];
			case "shouldStopAfterTurn":
				return this.stop as Entry<unknown>[];
			case "getSteeringMessages":
				return this.steering as Entry<unknown>[];
			case "getFollowUpMessages":
				return this.followUp as Entry<unknown>[];
			case "beforeCompact":
				return this.compact as Entry<unknown>[];
		}
		return [];
	}

	// Runs a handler scoped to the parent abort signal; a thrown handler is
	// caught, reported via onError, and skipped rather than aborting the chain.
	private async guard<T>(
		fn: (signal: AbortSignal) => T | Promise<T>,
		event: HookEventName,
		source: string | undefined,
		id: string,
		parentSignal?: AbortSignal,
	): Promise<T | undefined> {
		const controller = new AbortController();
		const abort = () => controller.abort(parentSignal?.reason);
		if (parentSignal?.aborted) abort();
		else parentSignal?.addEventListener("abort", abort, { once: true });
		try {
			if (controller.signal.aborted)
				throw controller.signal.reason ?? new Error("Hook handler aborted");
			return await fn(controller.signal);
		} catch (e) {
			const error = e as Error;
			this.onError?.(error, event, source);
			return undefined;
		} finally {
			parentSignal?.removeEventListener("abort", abort);
		}
	}

	private hasHandlerId(id: string): boolean {
		return (
			[
				...this.agentStart,
				...this.before,
				...this.after,
				...this.prepare,
				...this.transform,
				...this.providerRequest,
				...this.providerPayload,
				...this.afterProvider,
				...this.stop,
				...this.steering,
				...this.followUp,
				...this.compact,
			] as Array<Entry<unknown>>
		).some(entry => entry.id === id);
	}
}
