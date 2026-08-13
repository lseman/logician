export const EVAL_SCHEMA_VERSION = 1 as const;

export type TaskKind =
	| "bugfix"
	| "feature"
	| "refactor"
	| "docs"
	| "investigation";

export interface CommandSpec {
	command: string;
	args?: string[];
	timeoutMs?: number;
}

export type GraderSpec =
	| ({ id: string; type: "command" } & CommandSpec)
	| {
			id: string;
			type: "file_contains";
			path: string;
			pattern: string;
	  }
	| { id: string; type: "file_absent"; path: string }
	| {
			id: string;
			type: "diff_scope";
			baseRef: string;
			allowedPaths: string[];
			maxChangedFiles?: number;
	  };

export interface EvalTask {
	schemaVersion: typeof EVAL_SCHEMA_VERSION;
	id: string;
	title: string;
	kind: TaskKind;
	prompt: string;
	fixture: { repository: string; revision: string };
	agent: CommandSpec;
	graders: GraderSpec[];
	tags?: string[];
	limits?: {
		wallTimeMs?: number;
		maxTokens?: number;
		maxCostUsd?: number;
	};
}

export interface EvalCorpus {
	schemaVersion: typeof EVAL_SCHEMA_VERSION;
	name: string;
	tasks: EvalTask[];
}

export interface GraderResult {
	id: string;
	type: GraderSpec["type"];
	passed: boolean;
	durationMs: number;
	summary: string;
	evidence?: string;
}

export interface TrialMetrics {
	durationMs: number;
	exitCode: number | null;
	timedOut: boolean;
	changedFiles: number;
	toolCalls?: number;
	contextTokens?: number;
	model?: string;
}

export interface EvalTrial {
	schemaVersion: typeof EVAL_SCHEMA_VERSION;
	taskId: string;
	trialId: string;
	startedAt: string;
	workspace: string;
	agentDeclaredComplete: boolean | null;
	environmentGradedPass: boolean;
	graders: GraderResult[];
	metrics: TrialMetrics;
	trajectoryPath?: string;
	/** Internal until the CLI persists it as a trajectory artifact. */
	agentOutput?: string;
}

export interface EvalReport {
	schemaVersion: typeof EVAL_SCHEMA_VERSION;
	generatedAt: string;
	trials: EvalTrial[];
	summary: {
		passed: number;
		failed: number;
		passRate: number;
		medianDurationMs: number;
	};
}
