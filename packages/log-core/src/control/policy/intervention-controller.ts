import type {
	HarnessIntervention,
	HarnessInterventionAction,
	HarnessInterventionKind,
} from "../../system/types/intervention.ts";

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
		const key = this.incidentKey(input.kind, input.detector, input.cause);
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
			severity:
				action === "pause" || action === "stop"
					? "error"
					: action === "notice" || action === "continue"
						? "info"
						: "warning",
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

	/** Rebuild escalation state from a durable intervention trajectory. */
	replay(events: readonly HarnessIntervention[]): void {
		this.incidents.clear();
		for (const event of events) {
			const key = this.incidentKey(event.kind, event.detector, event.cause);
			const previous = this.incidents.get(key);
			if (!previous || event.attempt >= previous.attempt) {
				this.incidents.set(key, {
					id: event.id,
					attempt: event.attempt,
					lastIteration: event.iteration,
				});
			}
			const numericId = Number(event.id.match(/(\d+)$/)?.[1] ?? 0);
			this.sequence = Math.max(this.sequence, numericId);
		}
	}

	private incidentKey(
		kind: HarnessInterventionKind,
		detector: string,
		cause: string,
	): string {
		return `${kind}:${detector}:${cause}`;
	}

	private escalatedAction(attempt: number): HarnessInterventionAction {
		if (attempt <= 1) return "recover";
		if (attempt === 2) return "change_strategy";
		return "pause";
	}
}
