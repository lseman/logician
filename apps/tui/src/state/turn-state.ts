import type { RuntimeEvent } from "@logician/log-core/events";

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
	runningToolIds: readonly string[];
	startedAt?: number;
	settledAt?: number;
}

export const INITIAL_TURN_STATE: TurnState = {
	phase: "idle",
	runningTools: 0,
	runningToolIds: [],
};

/** Mark an accepted submission busy before the bridge assigns its turn id. */
export function beginPendingTurn(
	state: TurnState,
	now = Date.now(),
): TurnState {
	return {
		...state,
		phase: "thinking",
		turnId: undefined,
		runningTools: 0,
		runningToolIds: [],
		startedAt: now,
		settledAt: undefined,
	};
}

function toolCallId(event: RuntimeEvent): string | undefined {
	if (
		event.type !== "tool_execution_start" &&
		event.type !== "tool_execution_end"
	) {
		return undefined;
	}
	return event.toolCallId || undefined;
}

const VERIFY_PATTERN =
	/\b(test|check|lint|typecheck|pytest|cargo test|go test|ctest|make check)\b/i;

function isVerificationTool(event: RuntimeEvent): boolean {
	if (event.type !== "tool_execution_start") {
		return false;
	}
	if (event.toolName !== "bash") return false;
	const command = String(event.args?.command ?? "");
	return VERIFY_PATTERN.test(command);
}

/** Canonical lifecycle reducer. Renderers consume this state; they do not infer it. */
export function reduceTurnState(
	state: TurnState,
	event: RuntimeEvent,
	now = Date.now(),
): TurnState {
	switch (event.type) {
		case "turn_start":
			return {
				phase: "thinking",
				turnId: event.turnId,
				runningTools: 0,
				runningToolIds: [],
				startedAt: now,
			};
		case "thinking_token":
			return { ...state, phase: "thinking" };
		case "token":
		case "message_update":
		case "tool_call_start":
		case "tool_call_update":
		case "tool_call_id_update":
			return state.phase === "idle" || state.phase === "complete"
				? state
				: { ...state, phase: "streaming" };
		case "tool_execution_start": {
			const id = toolCallId(event);
			const duplicate = id !== undefined && state.runningToolIds.includes(id);
			return {
				...state,
				phase: isVerificationTool(event) ? "verifying" : "tool",
				runningTools: duplicate ? state.runningTools : state.runningTools + 1,
				runningToolIds:
					id === undefined || duplicate
						? state.runningToolIds
						: [...state.runningToolIds, id],
			};
		}
		case "tool_execution_end": {
			const id = toolCallId(event);
			const knownId = id !== undefined && state.runningToolIds.includes(id);
			const runningTools =
				id === undefined || knownId
					? Math.max(0, state.runningTools - 1)
					: state.runningTools;
			const runningToolIds = knownId
				? state.runningToolIds.filter(runningId => runningId !== id)
				: state.runningToolIds;
			return {
				...state,
				phase: event.isError
					? "failed"
					: runningTools > 0
						? state.phase
						: "thinking",
				runningTools,
				runningToolIds,
			};
		}
		case "permission_request":
			return { ...state, phase: "approval" };
		case "question_request":
			return { ...state, phase: "waiting" };
		case "turn_end":
			return {
				...state,
				phase: "complete",
				runningTools: 0,
				runningToolIds: [],
				settledAt: now,
			};
		case "notice":
			if (event.level === "error") {
				return {
					...state,
					phase: "failed",
					runningTools: 0,
					runningToolIds: [],
					settledAt: now,
				};
			}
			// A steerNow abort never gets a turn_end for the interrupted turn
			// (event-mapping.ts suppresses agent_end and surfaces this notice
			// instead, so a real completion notice doesn't also show). Without
			// this, phase stays stuck at "streaming"/"thinking" with the
			// animation still running until some later turn_end happens to
			// arrive — settle it here instead.
			if (event.label === "Steering") {
				return {
					...state,
					phase: "complete",
					runningTools: 0,
					runningToolIds: [],
					settledAt: now,
				};
			}
			return state;
		case "agent_error":
			return {
				...state,
				phase: event.recoverable ? state.phase : "failed",
				...(event.recoverable
					? {}
					: { runningTools: 0, runningToolIds: [], settledAt: now }),
			};
		case "phase":
			if (event.state === "ready") {
				// `ready` also describes bridge initialization and harness-idle
				// transitions. Only turn_end is authoritative for turn completion.
				return state;
			}
			if (event.state === "error") {
				return {
					...state,
					phase: "failed",
					runningTools: 0,
					runningToolIds: [],
					settledAt: now,
				};
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
