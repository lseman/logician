// ── Runtime State Reducer ─────────────────────────────────────────────────
// Pure reducer for AgentRuntimeState driven by AgentEvent stream.
// Matches Pi's reducer.ts pattern.

import type { AgentEvent } from "../types/types-messages.ts";
import type {
	AgentRuntimeState,
	HarnessPhase,
	RunOutcomeStatus,
} from "./result.ts";
import { createRuntimeState } from "./result.ts";

export {
	type AgentRuntimeState,
	createRuntimeState,
	type HarnessPhase,
	type RunOutcomeStatus,
};

/**
 * Apply a single event to the runtime state, returning a new state object.
 * This is the single source of truth for UI-facing runtime status.
 */
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
