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

import { withTimeout } from "../../tools/shared/async-utils.ts";
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
} from "../../agent/types.ts";

export type HookEventName = keyof AgentHooks;

export interface HookRegistration {
	/** Stable identity used for diagnostics and duplicate detection. */
	id?: string;
	source?: string;
	/** Higher priorities run first. Equal priorities retain registration order. */
	priority?: number;
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
	id: string;
	source?: string;
	priority: number;
	timeoutMs?: number;
	order: number;
}

export interface HookHandlerDiagnostics {
	id: string;
	source?: string;
	event: HookEventName;
	priority: number;
	count: number;
	errors: number;
	timeouts: number;
	totalMs: number;
	lastMs?: number;
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
	signal?: AbortSignal,
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
	private handlerStats = new Map<string, HookHandlerDiagnostics>();
	private cleanups = new Set<() => void | Promise<void>>();
	private nextOrder = 0;
	private nextAnonymousId = 0;
	private disposed = false;

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

	getDiagnostics(): HookHandlerDiagnostics[] {
		return [...this.handlerStats.values()].map((value) => ({ ...value }));
	}

	addCleanup(cleanup: () => void | Promise<void>): () => void {
		this.assertActive();
		this.cleanups.add(cleanup);
		return () => this.cleanups.delete(cleanup);
	}

	// Register one handler for an event. Returns an unsubscribe function.
	on<E extends HookEventName>(
		event: E,
		handler: NonNullable<AgentHooks[E]>,
		reg: HookRegistration = {},
	): () => void {
		this.assertActive();
		const list = this.listFor(event) as Entry<AgentHooks[E]>[];
		const id = reg.id ?? `${String(event)}#${++this.nextAnonymousId}`;
		if (this.hasHandlerId(id)) throw new Error(`Duplicate hook handler id: ${id}`);
		const entry = { handler, id, source: reg.source, priority: reg.priority ?? 0, timeoutMs: reg.timeoutMs, order: this.nextOrder++ };
		list.push(entry);
		list.sort((a, b) => b.priority - a.priority || a.order - b.order);
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

	async clear(): Promise<void> {
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
		const cleanups = [...this.cleanups];
		this.cleanups.clear();
		for (const cleanup of cleanups.reverse()) await cleanup();
	}

	async dispose(): Promise<void> {
		if (this.disposed) return;
		await this.clear();
		this.disposed = true;
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
			beforeProviderRequest: (ctx, signal) => this.runProviderRequest(ctx, signal),
			beforeProviderPayload: (ctx, signal) => this.runProviderPayload(ctx, signal),
			afterProviderResponse: (ctx, signal) => this.runAfterProvider(ctx, signal),
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
		await this.notify("beforeAgentStart", ctx, signal);
		let messages = ctx.messages;
		let systemPrompt = ctx.systemPrompt;
		let changed = false;
		for (const { handler, id, source, timeoutMs } of this.agentStart) {
			const result = await this.guard(
				(scopedSignal) => handler({ ...ctx, messages, systemPrompt }, scopedSignal),
				"beforeAgentStart",
				source,
				timeoutMs,
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
		await this.notify("beforeToolCall", ctx, signal);
		if (!this.before.length) return undefined;
		let current = ctx;
		let rewritten: Record<string, unknown> | undefined;
		for (const { handler, id, source, timeoutMs } of this.before) {
			const r = await this.guard(
				(scopedSignal) => handler(current, scopedSignal),
				"beforeToolCall",
				source,
				timeoutMs,
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
		await this.notify("afterToolCall", ctx, signal);
		if (!this.after.length) return undefined;
		let current = ctx;
		let modified = false;
		let terminate = false;
		for (const { handler, id, source, timeoutMs } of this.after) {
			const r = await this.guard(
				(scopedSignal) => handler(current, scopedSignal),
				"afterToolCall",
				source,
				timeoutMs,
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
		await this.notify("prepareNextTurn", ctx, signal);
		if (!this.prepare.length) return undefined;
		let messages = ctx.messages;
		for (const { handler, id, source, timeoutMs } of this.prepare) {
			const r = await this.guard(
				(scopedSignal) => handler({ ...ctx, messages }, scopedSignal),
				"prepareNextTurn",
				source,
				timeoutMs,
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
		await this.notify("transformContext", ctx, signal);
		if (!this.transform.length) return undefined;
		let messages = ctx.messages;
		for (const { handler, id, source, timeoutMs } of this.transform) {
			const r = await this.guard(
				(scopedSignal) => handler({ ...ctx, messages }, scopedSignal),
				"transformContext",
				source,
				timeoutMs,
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
		await this.notify("beforeProviderRequest", ctx, signal);
		if (!this.providerRequest.length) return undefined;
		const collectedHeaders: Record<string, string | undefined> = {};
		let timeoutMs: number | undefined;
		let maxRetries: number | undefined;
		let cacheRetention: string | undefined;
		let metadata: Record<string, unknown> | undefined;
		let transport: string | undefined;
		for (const { handler, id, source, timeoutMs: hookTimeoutMs } of this
			.providerRequest) {
			const r = await this.guard(
				(scopedSignal) => handler(ctx, scopedSignal),
				"beforeProviderRequest",
				source,
				hookTimeoutMs,
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
		await this.notify("beforeProviderPayload", ctx, signal);
		if (!this.providerPayload.length) return undefined;
		let payload = ctx.payload;
		for (const { handler, id, source, timeoutMs } of this.providerPayload) {
			const r = await this.guard(
				(scopedSignal) => handler({ ...ctx, payload }, scopedSignal),
				"beforeProviderPayload",
				source,
				timeoutMs,
				id,
				signal,
			);
			if (r?.payload) payload = r.payload;
		}
		return payload === ctx.payload ? undefined : { payload };
	}

	private async runAfterProvider(ctx: AfterProviderResponseContext, signal?: AbortSignal): Promise<void> {
		await this.notify("afterProviderResponse", ctx, signal);
		for (const { handler, id, source, timeoutMs } of this.afterProvider) {
			await this.guard((scopedSignal) => handler(ctx, scopedSignal), "afterProviderResponse", source, timeoutMs, id, signal);
		}
	}

	private async runStop(
		ctx: ShouldStopAfterTurnContext,
		signal?: AbortSignal,
	): Promise<boolean | undefined> {
		await this.notify("shouldStopAfterTurn", ctx, signal);
		for (const { handler, id, source, timeoutMs } of this.stop) {
			const r = await this.guard(
				(scopedSignal) => handler(ctx, scopedSignal),
				"shouldStopAfterTurn",
				source,
				timeoutMs,
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
		await this.notify("getSteeringMessages", ctx, signal);
		const out: Message[] = [];
		for (const { handler, id, source, timeoutMs } of this.steering) {
			const r = await this.guard(
				(scopedSignal) => handler({ ...ctx, messages: [...ctx.messages, ...out] }, scopedSignal),
				"getSteeringMessages",
				source,
				timeoutMs,
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
		await this.notify("getFollowUpMessages", ctx, signal);
		const out: Message[] = [];
		for (const { handler, id, source, timeoutMs } of this.followUp) {
			const r = await this.guard(
				(scopedSignal) =>
					handler({
						...ctx,
						messages: [...ctx.messages, ...out],
					}, scopedSignal),
				"getFollowUpMessages",
				source,
				timeoutMs,
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
		await this.notify("beforeCompact", ctx, signal);
		let summary: string | undefined;
		for (const { handler, id, source, timeoutMs } of this.compact) {
			const result = await this.guard(
				(scopedSignal) => handler(ctx, scopedSignal),
				"beforeCompact",
				source,
				timeoutMs,
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

	private async notify(event: HookEventName, ctx: unknown, signal?: AbortSignal): Promise<void> {
		for (const observer of this.observers) {
			try {
				await observer(event, ctx, signal);
			} catch (_e: unknown) {
				// Observers are read-only; their failures never affect a turn.
				console.error("[hook-bus] observer error:", _e);
			}
		}
	}

	private async guard<T>(
		fn: (signal: AbortSignal) => T | Promise<T>,
		event: HookEventName,
		source?: string,
		timeoutMs?: number,
		id = `${String(event)}#unknown`,
		parentSignal?: AbortSignal,
	): Promise<T | undefined> {
		const effective = timeoutMs ?? this.defaultTimeoutMs;
		const start = performance.now();
		const controller = new AbortController();
		const abort = () => controller.abort(parentSignal?.reason);
		if (parentSignal?.aborted) abort();
		else parentSignal?.addEventListener("abort", abort, { once: true });
		let timeout: ReturnType<typeof setTimeout> | undefined;
		if (effective > 0) timeout = setTimeout(() => controller.abort(new Error(`Hook handler timeout after ${effective}ms`)), effective);
		try {
			if (controller.signal.aborted) throw controller.signal.reason ?? new Error("Hook handler aborted");
			const run = Promise.resolve(fn(controller.signal));
			const result = effective > 0 ? await withTimeout(run, effective) : await run;
			if (controller.signal.aborted) throw controller.signal.reason ?? new Error("Hook handler aborted");
			const duration = performance.now() - start;
			this.metrics.record(event, duration);
			this.recordHandler(id, source, event, duration);
			return result;
		} catch (e) {
			const error = e as Error;
			const duration = performance.now() - start;
			this.metrics.record(event, duration, error);
			this.recordHandler(id, source, event, duration, error);
			this.onError?.(error, event, source);
			if (this.errorMode === "throw") throw error;
			return undefined;
		} finally {
			if (timeout) clearTimeout(timeout);
			parentSignal?.removeEventListener("abort", abort);
		}
	}

	private hasHandlerId(id: string): boolean {
		return ([...this.agentStart, ...this.before, ...this.after, ...this.prepare, ...this.transform, ...this.providerRequest, ...this.providerPayload, ...this.afterProvider, ...this.stop, ...this.steering, ...this.followUp, ...this.compact] as Array<Entry<unknown>>).some((entry) => entry.id === id);
	}

	private assertActive(): void {
		if (this.disposed) throw new Error("HookBus has been disposed");
	}

	private recordHandler(id: string, source: string | undefined, event: HookEventName, duration: number, error?: Error): void {
		const current = this.handlerStats.get(id) ?? { id, source, event, priority: this.findPriority(id), count: 0, errors: 0, timeouts: 0, totalMs: 0 };
		current.count++;
		current.totalMs += duration;
		current.lastMs = duration;
		if (error) {
			current.errors++;
			if (/timeout/i.test(error.message)) current.timeouts++;
		}
		this.handlerStats.set(id, current);
	}

	private findPriority(id: string): number {
		const entries = [...this.agentStart, ...this.before, ...this.after, ...this.prepare, ...this.transform, ...this.providerRequest, ...this.providerPayload, ...this.afterProvider, ...this.stop, ...this.steering, ...this.followUp, ...this.compact] as Array<Entry<unknown>>;
		return entries.find((entry) => entry.id === id)?.priority ?? 0;
	}
}
