import type { RunOutcomeStatus } from "../../system/types/execution-policy.ts";
import type { AgentEvent, Message } from "../../system/types/types-messages.ts";

export type HarnessPhase = "idle" | "turn" | "compaction" | "branch_summary";

export interface AgentRuntimeState {
	phase: HarnessPhase;
	isStreaming: boolean;
	turnId?: string;
	streamingMessage?: Message;
	pendingToolCalls: readonly string[];
	retry?: { attempt: number; maxRetries: number; delayMs?: number };
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
	};
}

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
		case "agent_retry_start":
			return {
				...current,
				retry: {
					attempt: event.attempt,
					maxRetries: event.maxRetries,
					delayMs: event.delayMs,
				},
			};
		case "agent_retry_end":
			return { ...current, retry: undefined };
		case "error":
			return { ...current, lastError: event.message };
		case "agent_end":
			return {
				...current,
				...(event.status && event.summary
					? { outcome: { status: event.status, summary: event.summary } }
					: {}),
				lastRunDurationMs: current.startedAt
					? Math.max(0, now - current.startedAt)
					: undefined,
				isStreaming: false,
				turnId: undefined,
				streamingMessage: undefined,
				pendingToolCalls: [],
				retry: undefined,
				turnStartedAt: undefined,
			};
		default:
			return current;
	}
}
