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
