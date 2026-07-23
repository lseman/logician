import type { ParsedBridgeEvent } from "@logician/coding-agent/events";

export type TurnPhase =
	| "idle"
	| "thinking"
	| "streaming"
	| "tool"
	| "verifying"
	| "waiting"
	| "approval"
	| "complete"
	| "failed";

export interface TurnState {
	phase: TurnPhase;
	turnId?: string;
	runningTools: number;
	startedAt?: number;
	settledAt?: number;
}

export const INITIAL_TURN_STATE: TurnState = {
	phase: "idle",
	runningTools: 0,
};

const VERIFY_PATTERN =
	/\b(test|check|lint|typecheck|pytest|cargo test|go test|ctest|make check)\b/i;

function isVerificationTool(event: ParsedBridgeEvent): boolean {
	if (event.type !== "tool_start" && event.type !== "tool_execution_start") {
		return false;
	}
	if (event.tool_name !== "bash" && event.tool !== "bash") return false;
	const command = String(event.tool_args?.command ?? "");
	return VERIFY_PATTERN.test(command);
}

/** Canonical lifecycle reducer. Renderers consume this state; they do not infer it. */
export function reduceTurnState(
	state: TurnState,
	event: ParsedBridgeEvent,
	now = Date.now(),
): TurnState {
	switch (event.type) {
		case "turn_start":
			return {
				phase: "thinking",
				turnId: event.turn_id,
				runningTools: 0,
				startedAt: now,
			};
		case "thinking_token":
			return { ...state, phase: "thinking" };
		case "token":
		case "text_start":
		case "message_update":
			return state.phase === "idle" || state.phase === "complete"
				? state
				: { ...state, phase: "streaming" };
		case "tool_start":
		case "tool_execution_start":
			return {
				...state,
				phase: isVerificationTool(event) ? "verifying" : "tool",
				runningTools: state.runningTools + 1,
			};
		case "tool_end":
		case "tool_execution_end": {
			const runningTools = Math.max(0, state.runningTools - 1);
			return {
				...state,
				phase: event.is_error
					? "failed"
					: runningTools > 0 ? state.phase : "thinking",
				runningTools,
			};
		}
		case "permission_request":
			return { ...state, phase: "approval" };
		case "question_request":
			return { ...state, phase: "waiting" };
		case "turn_end":
			return { ...state, phase: "complete", runningTools: 0, settledAt: now };
		case "notice":
			return event.level === "error"
				? { ...state, phase: "failed", runningTools: 0, settledAt: now }
				: state;
		case "phase":
			if (event.state === "ready") {
				return state.startedAt
					? { ...state, phase: "complete", runningTools: 0, settledAt: now }
					: { ...state, phase: "idle", runningTools: 0 };
			}
			if (event.state === "error") {
				return { ...state, phase: "failed", runningTools: 0, settledAt: now };
			}
			return state;
		default:
			return state;
	}
}

export function turnPhaseLabel(phase: TurnPhase): string {
	return phase === "idle" || phase === "complete" ? "ready" : phase;
}

export function turnPhaseIsActive(phase: TurnPhase): boolean {
	return !["idle", "complete", "failed", "waiting", "approval"].includes(phase);
}
