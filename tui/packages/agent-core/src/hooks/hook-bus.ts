// ── Typed hook bus ─────────────────────────────────────────────────────────
// Unifies the single-handler contract hooks into one multi-handler bus with
// per-event reducer semantics, mirroring pi's hook design
// (packages/agent/docs/hooks.md). Multiple extensions register handlers for the
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
// calls one handler per event. `observe()` is a read-only
// firehose over every event. Handlers can be scoped with source metadata so a
// failing extension is identifiable, and `errorMode` controls whether a thrown
// handler aborts the chain or is skipped.

import { withTimeout } from "../tools/shared/async-utils.ts";
import { HookMetricsCollector } from "./hook-metrics.ts";
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
} from "../core/types.ts";

export type HookEventName = keyof AgentHooks;

export interface HookRegistration {
	source?: string;
	/** Per-handler timeout. Overrides the bus default; 0 disables. */
	timeoutMs?: number;
}

export type HookErrorMode = "continue" | "throw";

export interface HookBusOptions {
	errorMode?: HookErrorMode;
	onError?: (error: Error, event: HookEventName, source?: string) => void;
	/** Default per-handler timeout — one slow handler must not stall the turn.
	 *  A timed-out handler is treated like a thrown one (skipped + reported).
	 *  0 = no default timeout. */
	defaultTimeoutMs?: number;
}

interface Entry<H> {
	handler: H;
	source?: string;
	timeoutMs?: number;
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

// Read-only observer: sees every event with its name, return ignored.
export type HookObserver = (
	event: HookEventName,
	ctx: unknown,
) => void | Promise<void>;

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
	private observers: HookObserver[] = [];
	private metrics = new HookMetricsCollector();

	private errorMode: HookErrorMode;
	private onError?: HookBusOptions["onError"];
	private defaultTimeoutMs: number;

	constructor(options: HookBusOptions = {}) {
		this.errorMode = options.errorMode ?? "continue";
		this.onError = options.onError;
		this.defaultTimeoutMs = options.defaultTimeoutMs ?? 0;
	}

	/** Get hook execution metrics. */
	getMetrics(): HookMetricsCollector {
		return this.metrics;
	}

	// Register one handler for an event. Returns an unsubscribe function.
	on<E extends HookEventName>(
		event: E,
		handler: NonNullable<AgentHooks[E]>,
		reg: HookRegistration = {},
	): () => void {
		const list = this.listFor(event) as Entry<AgentHooks[E]>[];
		const entry = { handler, source: reg.source, timeoutMs: reg.timeoutMs };
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
			offs.forEach((off) => {
				off();
			});
	}

	observe(observer: HookObserver): () => void {
		this.observers.push(observer);
		return () => {
			const i = this.observers.indexOf(observer);
			if (i >= 0) this.observers.splice(i, 1);
		};
	}

	clear(): void {
		this.before = [];
		this.after = [];
		this.prepare = [];
		this.transform = [];
		this.providerRequest = [];
		this.providerPayload = [];
		this.afterProvider = [];
		this.stop = [];
		this.steering = [];
		this.followUp = [];
		this.agentStart = [];
		this.compact = [];
		this.observers = [];
	}

	// Single composed AgentHooks for the runner. Each event runs its
	// reducer over all registered handlers.
	toHooks(): AgentHooks {
		return {
			beforeAgentStart: (ctx) => this.runAgentStart(ctx),
			beforeToolCall: (ctx) => this.runBefore(ctx),
			afterToolCall: (ctx) => this.runAfter(ctx),
			prepareNextTurn: (ctx) => this.runPrepare(ctx),
			transformContext: (ctx) => this.runTransform(ctx),
			beforeProviderRequest: (ctx) => this.runProviderRequest(ctx),
			beforeProviderPayload: (ctx) => this.runProviderPayload(ctx),
			afterProviderResponse: (ctx) => this.runAfterProvider(ctx),
			shouldStopAfterTurn: (ctx) => this.runStop(ctx),
			getSteeringMessages: (ctx) => this.runSteering(ctx),
			getFollowUpMessages: (ctx) => this.runFollowUp(ctx),
			beforeCompact: (ctx) => this.runCompact(ctx),
		};
	}

	// ── Reducers ───────────────────────────────────────────────────────────
	private async runAgentStart(
		ctx: BeforeAgentStartContext,
	): Promise<BeforeAgentStartResult | undefined> {
		await this.notify("beforeAgentStart", ctx);
		let messages = ctx.messages;
		let systemPrompt = ctx.systemPrompt;
		let changed = false;
		for (const { handler, source, timeoutMs } of this.agentStart) {
			const result = await this.guard(
				() => handler({ ...ctx, messages, systemPrompt }),
				"beforeAgentStart",
				source,
				timeoutMs,
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
	): Promise<BeforeToolCallResult | undefined> {
		await this.notify("beforeToolCall", ctx);
		if (!this.before.length) return undefined;
		let current = ctx;
		let rewritten: Record<string, unknown> | undefined;
		for (const { handler, source, timeoutMs } of this.before) {
			const r = await this.guard(
				() => handler(current),
				"beforeToolCall",
				source,
				timeoutMs,
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
	): Promise<AfterToolCallResult | undefined> {
		await this.notify("afterToolCall", ctx);
		if (!this.after.length) return undefined;
		let current = ctx;
		let modified = false;
		let terminate = false;
		for (const { handler, source, timeoutMs } of this.after) {
			const r = await this.guard(
				() => handler(current),
				"afterToolCall",
				source,
				timeoutMs,
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
	): Promise<PrepareNextTurnResult | undefined> {
		await this.notify("prepareNextTurn", ctx);
		if (!this.prepare.length) return undefined;
		let messages = ctx.messages;
		for (const { handler, source, timeoutMs } of this.prepare) {
			const r = await this.guard(
				() => handler({ ...ctx, messages }),
				"prepareNextTurn",
				source,
				timeoutMs,
			);
			if (r?.messages) messages = r.messages;
		}
		return messages === ctx.messages ? undefined : { messages };
	}

	private async runTransform(
		ctx: TransformContext,
	): Promise<TransformContextResult | undefined> {
		await this.notify("transformContext", ctx);
		if (!this.transform.length) return undefined;
		let messages = ctx.messages;
		for (const { handler, source, timeoutMs } of this.transform) {
			const r = await this.guard(
				() => handler({ ...ctx, messages }),
				"transformContext",
				source,
				timeoutMs,
			);
			if (r?.messages) messages = r.messages;
		}
		return messages === ctx.messages ? undefined : { messages };
	}

	private async runProviderRequest(
		ctx: BeforeProviderRequestContext,
	): Promise<BeforeProviderRequestResult | undefined> {
		await this.notify("beforeProviderRequest", ctx);
		if (!this.providerRequest.length) return undefined;
		const collectedHeaders: Record<string, string | undefined> = {};
		let timeoutMs: number | undefined;
		let maxRetries: number | undefined;
		let cacheRetention: string | undefined;
		let metadata: Record<string, unknown> | undefined;
		let transport: string | undefined;
		for (const { handler, source, timeoutMs: hookTimeoutMs } of this
			.providerRequest) {
			const r = await this.guard(
				() => handler(ctx),
				"beforeProviderRequest",
				source,
				hookTimeoutMs,
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
	): Promise<BeforeProviderPayloadResult | undefined> {
		await this.notify("beforeProviderPayload", ctx);
		if (!this.providerPayload.length) return undefined;
		let payload = ctx.payload;
		for (const { handler, source, timeoutMs } of this.providerPayload) {
			const r = await this.guard(
				() => handler({ ...ctx, payload }),
				"beforeProviderPayload",
				source,
				timeoutMs,
			);
			if (r?.payload) payload = r.payload;
		}
		return payload === ctx.payload ? undefined : { payload };
	}

	private async runAfterProvider(ctx: AfterProviderResponseContext): Promise<void> {
		await this.notify("afterProviderResponse", ctx);
		for (const { handler, source, timeoutMs } of this.afterProvider) {
			await this.guard(() => handler(ctx), "afterProviderResponse", source, timeoutMs);
		}
	}

	private async runStop(
		ctx: ShouldStopAfterTurnContext,
	): Promise<boolean | undefined> {
		await this.notify("shouldStopAfterTurn", ctx);
		for (const { handler, source, timeoutMs } of this.stop) {
			const r = await this.guard(
				() => handler(ctx),
				"shouldStopAfterTurn",
				source,
				timeoutMs,
			);
			if (r === true) return true;
		}
		return undefined;
	}

	private async runSteering(
		ctx: GetSteeringMessagesContext,
	): Promise<Message[] | undefined> {
		await this.notify("getSteeringMessages", ctx);
		const out: Message[] = [];
		for (const { handler, source, timeoutMs } of this.steering) {
			const r = await this.guard(
				() => handler({ ...ctx, messages: [...ctx.messages, ...out] }),
				"getSteeringMessages",
				source,
				timeoutMs,
			);
			if (r?.length) out.push(...r);
		}
		return out.length ? out : undefined;
	}

	private async runFollowUp(
		ctx: GetFollowUpMessagesContext,
	): Promise<Message[] | undefined> {
		await this.notify("getFollowUpMessages", ctx);
		const out: Message[] = [];
		for (const { handler, source, timeoutMs } of this.followUp) {
			const r = await this.guard(
				() =>
					handler({
						...ctx,
						messages: [...ctx.messages, ...out],
					}),
				"getFollowUpMessages",
				source,
				timeoutMs,
			);
			if (r?.length) out.push(...r);
		}
		return out.length ? out : undefined;
	}

	private async runCompact(
		ctx: BeforeCompactContext,
	): Promise<BeforeCompactResult | undefined> {
		await this.notify("beforeCompact", ctx);
		let summary: string | undefined;
		for (const { handler, source, timeoutMs } of this.compact) {
			const result = await this.guard(
				() => handler(ctx),
				"beforeCompact",
				source,
				timeoutMs,
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

	private async notify(event: HookEventName, ctx: unknown): Promise<void> {
		for (const observer of this.observers) {
			try {
				await observer(event, ctx);
			} catch {
				// Observers are read-only; their failures never affect a turn.
			}
		}
	}

	private async guard<T>(
		fn: () => T | Promise<T>,
		event: HookEventName,
		source?: string,
		timeoutMs?: number,
	): Promise<T | undefined> {
		const effective = timeoutMs ?? this.defaultTimeoutMs;
		const start = Date.now();
		try {
			const run = Promise.resolve(fn());
			const result = effective > 0 ? await withTimeout(run, effective) : await run;
			const duration = Date.now() - start;
			this.metrics.record(event, duration);
			return result;
		} catch (e) {
			const error = e as Error;
			const duration = Date.now() - start;
			this.metrics.record(event, duration, error);
			this.onError?.(error, event, source);
			if (this.errorMode === "throw") throw error;
			return undefined;
		}
	}
}
