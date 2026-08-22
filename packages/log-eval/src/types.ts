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
	permissionRequests?: number;
	compactions?: number;
	retries?: number;
}

/**
 * Snapshot of the harness config in effect for a trial, read from
 * ~/.logician/settings.json at the moment the trial ran. The agent runs as a
 * subprocess that inherits process.env and reads its own config from disk —
 * the eval runner doesn't control these settings, so this snapshot exists
 * purely so a report can be traced back to what config produced it, rather
 * than assuming it matches whatever the config file holds today.
 */
export interface HarnessConfigSnapshot {
	model?: string;
	permissionMode?: string;
	toolExecution?: string;
	maxIterations?: number;
	compaction?: Record<string, unknown>;
	mcpServerNames?: string[];
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
	harnessConfig?: HarnessConfigSnapshot;
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
