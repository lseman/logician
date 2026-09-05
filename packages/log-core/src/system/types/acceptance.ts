export type CriterionSeverity = "required" | "recommended";

export interface AcceptanceCriterion {
	id: string;
	must: string;
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
	verify?: AcceptanceVerification[];
	stopRules?: string[];
}
