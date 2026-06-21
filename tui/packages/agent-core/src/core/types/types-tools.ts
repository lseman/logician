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
}

export type ToolExecutionMode = "sequential" | "parallel";

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
	cacheable?: boolean;
	hookAliases?: string[];
	readOnly?: boolean;
	execute: (
		args: Record<string, unknown>,
		ctx: ToolContext,
	) => Promise<string | ToolResult>;
}

export interface AskUserContext {
	question: string;
	choices: Array<{ value: string; label: string }>;
}

export interface ToolContext {
	cwd?: string;
	maxOutputChars?: number;
	signal?: AbortSignal;
	onUpdate?: (partialResult: string) => void;
	onQuestionRequest?: (ctx: AskUserContext) => Promise<string>;
}
