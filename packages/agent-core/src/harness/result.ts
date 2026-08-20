// ── Harness Result Types ──────────────────────────────────────────────────
// Tagged errors, Result monad, phase machine, and runtime state.

import type { AgentEvent, RunOutcomeStatus } from "../types/types-messages.ts";

export type { RunOutcomeStatus };

// ── TaggedError / Result ──────────────────────────────────────────────────
// Pi-compatible error types. See Pi's harness/result.ts for reference.

export type Result<TValue, TError = Error> =
	| { ok: true; value: TValue }
	| { ok: false; error: TError };

export function ok<TValue, TError = Error>(
	value: TValue,
): Result<TValue, TError> {
	return { ok: true, value };
}

export function err<TValue, TError = Error>(
	error: TError,
): Result<never, TError> {
	return { ok: false, error };
}

export interface TaggedErrorValue<Tag extends string> extends Error {
	readonly _tag: Tag;
}

export interface TaggedErrorFactory<Tag extends string> {
	new (message: string): TaggedErrorValue<Tag>;
	is(value: unknown): value is TaggedErrorValue<Tag>;
}

class TaggedErrorClass extends Error {
	_tag!: string;
}

export function TaggedError<Tag extends string>(
	tag: Tag,
): TaggedErrorFactory<Tag> {
	const factory = ((message: string) => {
		const err = Object.create(TaggedErrorClass.prototype);
		err._tag = tag;
		err.message = message;
		err.name = tag;
		return err as TaggedErrorValue<Tag>;
	}) as unknown as TaggedErrorFactory<Tag>;
	factory.is = (value: unknown): value is TaggedErrorValue<Tag> =>
		value != null &&
		typeof value === "object" &&
		"_tag" in value &&
		(value as any)._tag === tag;
	return factory;
}

// ── Harness Phase ─────────────────────────────────────────────────────────

export type HarnessPhase = "idle" | "turn" | "compaction" | "branch_summary";

// ── Harness Busy Error ────────────────────────────────────────────────────

export class HarnessBusyError extends Error {
	readonly phase: HarnessPhase;
	readonly required: HarnessPhase;

	constructor(op: string, phase: HarnessPhase, required: HarnessPhase) {
		super(
			`AgentHarness cannot ${op}: phase is "${phase}", requires "${required}"`,
		);
		this.name = "HarnessBusyError";
		this.phase = phase;
		this.required = required;
	}
}

export function assertIdlePhase(phase: HarnessPhase, op: string): void {
	if (phase !== "idle") {
		throw new HarnessBusyError(op, phase, "idle");
	}
}

// ── Agent Runtime State ───────────────────────────────────────────────────

export type AgentRuntimeState = {
	phase: HarnessPhase;
	isStreaming: boolean;
	turnId?: string;
	streamingMessage?: { role: string; content?: string | null };
	pendingToolCalls: readonly string[];
	abortRequested: boolean;
	lastError?: string;
	lastEventSeq?: number;
	startedAt?: number;
	turnStartedAt?: number;
	lastTurnDurationMs?: number;
	lastRunDurationMs?: number;
	outcome?: {
		status: RunOutcomeStatus;
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	};
};

export function createRuntimeState(
	phase: HarnessPhase = "idle",
): AgentRuntimeState {
	return {
		phase,
		isStreaming: false,
		pendingToolCalls: [],
		abortRequested: false,
	};
}

/** In-memory event projection driving UI-facing runtime status. */
export function reduceRuntimeState(
	state: AgentRuntimeState,
	event: AgentEvent,
	phase: HarnessPhase = state.phase,
): AgentRuntimeState {
	const current =
		event.seq === undefined ? state : { ...state, lastEventSeq: event.seq };
	const now = event.ts ?? Date.now();
	switch (event.type) {
		case "agent_start":
			return {
				phase,
				isStreaming: true,
				pendingToolCalls: [],
				abortRequested: false,
				lastEventSeq: event.seq,
				startedAt: now,
			};
		case "run_outcome":
			return {
				...current,
				outcome: {
					status: event.status,
					summary: event.summary,
					source: event.source,
				},
			};
		case "turn_start":
			return { ...current, turnId: event.turnId, turnStartedAt: now };
		case "turn_end":
			return {
				...current,
				lastTurnDurationMs: current.turnStartedAt
					? Math.max(0, now - current.turnStartedAt)
					: undefined,
				turnStartedAt: undefined,
			};
		case "message_start":
			return event.role === "assistant"
				? { ...current, streamingMessage: { role: "assistant", content: "" } }
				: current;
		case "text_delta": {
			const content =
				current.streamingMessage?.role === "assistant"
					? (current.streamingMessage.content ?? "")
					: "";
			return {
				...current,
				streamingMessage: { role: "assistant", content: content + event.delta },
			};
		}
		case "message_update":
			return event.message.role === "assistant"
				? { ...current, streamingMessage: event.message }
				: current;
		case "message_end":
			return event.message?.role === "assistant"
				? { ...current, streamingMessage: undefined }
				: current;
		case "tool_call_start":
			return {
				...current,
				pendingToolCalls: Array.from(
					new Set([...current.pendingToolCalls, event.toolCallId]),
				),
			};
		case "tool_call_end":
			return {
				...current,
				pendingToolCalls: current.pendingToolCalls.filter(
					id => id !== event.toolCallId,
				),
			};
		case "error":
			return { ...current, lastError: event.message };
		case "agent_end":
			return {
				...current,
				lastRunDurationMs: current.startedAt
					? Math.max(0, now - current.startedAt)
					: undefined,
				isStreaming: false,
				turnId: undefined,
				streamingMessage: undefined,
				pendingToolCalls: [],
				turnStartedAt: undefined,
			};
		default:
			return current;
	}
}

// ── Abort Result ──────────────────────────────────────────────────────────

export interface AbortResult {
	clearedSteering: string[];
	clearedFollowUp: string[];
	clearedNextTurn: string[];
}

export interface HarnessQueues {
	steering: string[];
	followUp: string[];
	nextTurn: string[];
}
