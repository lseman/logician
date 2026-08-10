export type HarnessInterventionKind =
	| "continuation"
	| "loop"
	| "budget"
	| "retry"
	| "verification"
	| "compaction";

export type HarnessInterventionAction =
	| "notice"
	| "recover"
	| "change_strategy"
	| "continue"
	| "pause"
	| "stop";

export interface HarnessInterventionEvidence {
	summary: string;
	signals?: readonly string[];
	counters?: Readonly<Record<string, number>>;
}

export interface HarnessIntervention {
	id: string;
	kind: HarnessInterventionKind;
	cause: string;
	action: HarnessInterventionAction;
	severity: "info" | "warning" | "error";
	detector: string;
	attempt: number;
	evidence: HarnessInterventionEvidence;
	limits?: Readonly<Record<string, number>>;
	nextAction?: string;
	iteration: number;
}

interface IncidentState {
	id: string;
	attempt: number;
	lastIteration: number;
}

export interface InterventionInput {
	kind: HarnessInterventionKind;
	cause: string;
	detector: string;
	message: string;
	iteration: number;
	signals?: readonly string[];
	counters?: Readonly<Record<string, number>>;
	limits?: Readonly<Record<string, number>>;
	action?: HarnessInterventionAction;
	nextAction?: string;
}

/**
 * Owns intervention incident identity and escalation for one agent run.
 * Repeated detections become recovery, strategy-change, then pause actions.
 */
export class HarnessInterventionController {
	private readonly incidents = new Map<string, IncidentState>();
	private sequence = 0;

	record(input: InterventionInput): HarnessIntervention {
		const key = `${input.kind}:${input.detector}:${input.cause}`;
		const previous = this.incidents.get(key);
		const attempt = (previous?.attempt ?? 0) + 1;
		const id = previous?.id ?? `intervention-${++this.sequence}`;
		this.incidents.set(key, { id, attempt, lastIteration: input.iteration });

		const action = input.action ?? this.escalatedAction(attempt);
		return {
			id,
			kind: input.kind,
			cause: input.cause,
			action,
			severity: action === "pause" || action === "stop" ? "error" : "warning",
			detector: input.detector,
			attempt,
			evidence: {
				summary: input.message,
				signals: input.signals,
				counters: input.counters,
			},
			limits: input.limits,
			nextAction: input.nextAction,
			iteration: input.iteration,
		};
	}

	/** Verified progress closes active incidents so later failures start fresh. */
	recordProgress(): void {
		this.incidents.clear();
	}

	private escalatedAction(attempt: number): HarnessInterventionAction {
		if (attempt <= 1) return "recover";
		if (attempt === 2) return "change_strategy";
		return "pause";
	}
}
