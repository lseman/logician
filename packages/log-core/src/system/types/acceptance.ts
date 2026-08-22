export type EvidenceKind =
	| "changed-files"
	| "tests-added"
	| "commands-run"
	| "validation-output"
	| "residual-risks"
	| "no-staged-files"
	| "diff-summary"
	| "manual-notes";

export type CriterionSeverity = "required" | "recommended";

export interface AcceptanceCriterion {
	id: string;
	must: string;
	evidence?: EvidenceKind[];
	severity?: CriterionSeverity;
}

export interface AcceptanceVerification {
	id: string;
	command: string;
	cwd?: string;
	timeoutMs?: number;
	allowFailure?: boolean;
}

export interface AcceptanceConfig {
	criteria?: string[] | AcceptanceCriterion[];
	evidence?: EvidenceKind[];
	verify?: AcceptanceVerification[];
	stopRules?: string[];
}
