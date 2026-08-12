// ── Tool types ────────────────────────────────────────────────────────────

export interface ToolCall {
	id: string;
	name: string;
	arguments: string;
}

/** Structured tool result. */
export interface ToolResult {
	content: string;
	details?: Record<string, unknown>;
	isError?: boolean;
	terminate?: boolean;
	/** Provider receipt used to reconcile a crash before commit. */
	recoveryReceipt?: string;
}

export type ToolExecutionMode = "sequential" | "parallel";
export type ToolRecoverySemantics =
	| "pure"
	| "idempotent"
	| "receipt_recoverable"
	| "at_most_once_unknown";

export interface Tool {
	name: string;
	/** Human-readable label shown in UI/tool lists. */
	label?: string;
	description: string;
	/** One-line description for the "Available tools" section in the system prompt. */
	promptSnippet?: string;
	/** Guideline bullets for the system prompt Guidelines section. */
	promptGuidelines?: string[];
	parameters: Record<string, unknown>;
	prepareArguments?: (args: unknown) => Record<string, unknown>;
	executionMode?: ToolExecutionMode;
	/** Opt-in result caching. Only pure, side-effect-free tools should set this. */
	cacheable?: boolean;
	/** Execution timeout in ms. Overrides the registry default; 0 disables. */
	timeoutMs?: number;
	/**
	 * Per-call timeout in ms derived from the call's arguments (e.g. bash's
	 * timeout parameter). Takes precedence over timeoutMs; return undefined to
	 * fall through.
	 */
	resolveTimeoutMs?: (args: Record<string, unknown>) => number | undefined;
	hookAliases?: string[];
	readOnly?: boolean;
	/** Crash-recovery contract. Mutating tools default to at_most_once_unknown. */
	recoverySemantics?: ToolRecoverySemantics;
	execute: (
		args: Record<string, unknown>,
		ctx: ToolContext,
	) => Promise<string | ToolResult>;
}

export interface AskUserContext {
	questions: Array<{
		id: string;
		header?: string;
		question: string;
		choices: Array<{ value: string; label: string; description?: string }>;
	}>;
}

export interface ToolContext {
	cwd?: string;
	allowedPaths?: string[];
	allowAllPaths?: boolean;
	maxOutputChars?: number;
	signal?: AbortSignal;
	onUpdate?: (partialResult: string) => void;
	onQuestionRequest?: (ctx: AskUserContext) => Promise<string>;
	/** Stable across retries of the same logical operation. */
	idempotencyKey?: string;
	/** Run Kernel operation identity for receipts and correlation. */
	operationId?: string;
}
