/** Configuration, inference, queue, truncation, and agent error contracts. */

import type { AcceptanceConfig } from "../../control/guards/acceptance-contract.ts";
import type {
	AgentStopPolicy,
	ExecutionProfile,
} from "../../control/policy/execution-policy.ts";
import type { RunBudgetLimits } from "../../control/policy/run-budget.ts";
import type { TaskLedger } from "../../control/policy/task-ledger.ts";
import type { PermissionManager } from "../../capabilities/tools/permissions.ts";
import type { AgentHooks, EventHandler, Tool } from "./types-messages.ts";

export type QueueMode = "all" | "one-at-a-time";
export const QUEUE_MODES: readonly QueueMode[] = ["all", "one-at-a-time"];

export type ThinkingLevel =
	| "off"
	| "minimal"
	| "low"
	| "medium"
	| "high"
	| "xhigh";

export const THINKING_LEVELS: readonly ThinkingLevel[] = [
	"off",
	"minimal",
	"low",
	"medium",
	"high",
	"xhigh",
];

export const VALID_TOOL_EXECUTION: readonly string[] = [
	"sequential",
	"parallel",
];

export type InferenceMode =
	| "auto"
	| "none"
	| "thinking-general"
	| "thinking-coding"
	| "instruct-general"
	| "instruct-reasoning"
	| "instruct-coding"
	| "deterministic"
	| "creative"
	| "analytical";

export interface SamplingParams {
	temperature: number;
	top_p: number;
	top_k: number;
	min_p: number;
	presence_penalty: number;
	repetition_penalty: number;
}

export interface InferenceModeDef {
	label: string;
	description: string;
	useProviderDefaults: boolean;
	params: SamplingParams;
}

export const INFERENCE_MODE_ORDER: readonly InferenceMode[] = [
	"auto",
	"none",
	"thinking-general",
	"thinking-coding",
	"instruct-general",
	"instruct-reasoning",
	"instruct-coding",
	"deterministic",
	"creative",
	"analytical",
];

export const DEFAULT_INFERENCE_MODE: InferenceMode = "none";

export const INFERENCE_MODES: ReadonlyMap<InferenceMode, InferenceModeDef> =
	new Map([
		[
			"auto",
			{
				label: "Auto",
				description: "Automatically selects a preset from live task evidence.",
				useProviderDefaults: false,
				params: {
					temperature: 0.7,
					top_p: 0.8,
					top_k: 20,
					min_p: 0,
					presence_penalty: 1,
					repetition_penalty: 1,
				},
			},
		],
		[
			"none",
			{
				label: "Provider",
				description: "Pass nothing and let the provider use its defaults.",
				useProviderDefaults: true,
				params: {
					temperature: 0.7,
					top_p: 0.8,
					top_k: 20,
					min_p: 0,
					presence_penalty: 0,
					repetition_penalty: 1,
				},
			},
		],
		[
			"thinking-general",
			{
				label: "Think Gen",
				description: "General thinking with diverse sampling.",
				useProviderDefaults: false,
				params: {
					temperature: 1,
					top_p: 0.95,
					top_k: 20,
					min_p: 0,
					presence_penalty: 1.5,
					repetition_penalty: 1,
				},
			},
		],
		[
			"thinking-coding",
			{
				label: "Think Code",
				description: "Precise coding-oriented thinking.",
				useProviderDefaults: false,
				params: {
					temperature: 0.6,
					top_p: 0.95,
					top_k: 20,
					min_p: 0,
					presence_penalty: 0,
					repetition_penalty: 1,
				},
			},
		],
		[
			"instruct-general",
			{
				label: "Instruct",
				description: "Balanced non-thinking general sampling.",
				useProviderDefaults: false,
				params: {
					temperature: 0.7,
					top_p: 0.8,
					top_k: 20,
					min_p: 0,
					presence_penalty: 1.5,
					repetition_penalty: 1,
				},
			},
		],
		[
			"instruct-reasoning",
			{
				label: "Reason",
				description: "Non-thinking reasoning sampling.",
				useProviderDefaults: false,
				params: {
					temperature: 1,
					top_p: 0.95,
					top_k: 20,
					min_p: 0,
					presence_penalty: 1.5,
					repetition_penalty: 1,
				},
			},
		],
		[
			"instruct-coding",
			{
				label: "Code",
				description: "Precise non-thinking coding sampling.",
				useProviderDefaults: false,
				params: {
					temperature: 0.3,
					top_p: 0.9,
					top_k: 20,
					min_p: 0,
					presence_penalty: 0,
					repetition_penalty: 1,
				},
			},
		],
		[
			"deterministic",
			{
				label: "Exact",
				description: "Near-deterministic sampling.",
				useProviderDefaults: false,
				params: {
					temperature: 0,
					top_p: 0,
					top_k: 1,
					min_p: 0,
					presence_penalty: 0,
					repetition_penalty: 1,
				},
			},
		],
		[
			"creative",
			{
				label: "Creative",
				description: "High-diversity ideation sampling.",
				useProviderDefaults: false,
				params: {
					temperature: 1.3,
					top_p: 0.99,
					top_k: 40,
					min_p: 0,
					presence_penalty: 2,
					repetition_penalty: 0.9,
				},
			},
		],
		[
			"analytical",
			{
				label: "Analyze",
				description: "Tight sampling for analysis and review.",
				useProviderDefaults: false,
				params: {
					temperature: 0.2,
					top_p: 0.7,
					top_k: 20,
					min_p: 0,
					presence_penalty: 0.5,
					repetition_penalty: 1.1,
				},
			},
		],
	]);

export function getInferenceMode(
	mode: InferenceMode,
): InferenceModeDef | undefined {
	return INFERENCE_MODES.get(mode);
}

export function cycleInferenceMode(current: InferenceMode): InferenceMode {
	const index = INFERENCE_MODE_ORDER.indexOf(current);
	return INFERENCE_MODE_ORDER[(index + 1) % INFERENCE_MODE_ORDER.length];
}

export function isValidInferenceMode(value: string): value is InferenceMode {
	return INFERENCE_MODES.has(value as InferenceMode);
}

/** Curated provider request options owned by the harness and snapshotted per turn. */
export interface AgentHarnessStreamOptions {
	/** Timeout in milliseconds. */
	timeoutMs?: number;
	/** Maximum provider retry attempts. */
	maxRetries?: number;
	/** Optional cap for provider-requested retry delays. */
	maxRetryDelayMs?: number;
	/** Additional request headers merged with auth and lifecycle headers. */
	headers?: Record<string, string>;
	/** Provider metadata forwarded with requests. */
	metadata?: Record<string, unknown>;
	/** Provider cache retention hint. */
	cacheRetention?: string;
}

export interface AgentModelConfig {
	/** Display name for the model. */
	name: string;
	/** Model identifier sent to the API. */
	model: string;
	/** Optional per-model baseUrl override for cycling between endpoints. */
	url?: string;
}

export interface AgentConfig {
	baseUrl: string;
	model: string;
	models?: AgentModelConfig[];
	cwd?: string;
	temperature?: number;
	maxTokens?: number;
	chatTemplate?: string;
	stop?: string[];
	maxIterations?: number;
	/**
	 * `minimal` keeps the provider/tool/queue mechanism while disabling the
	 * runner's built-in continuation, acceptance, and repair policies.
	 */
	executionProfile?: ExecutionProfile;
	/** Policies evaluated externally when the agent loop naturally becomes idle. */
	stopPolicies?: AgentStopPolicy[];
	/** Optional task capability observed by autonomous policy. */
	taskLedger?: TaskLedger;
	loopDetectionWindow?: number;
	degenerateLoopThreshold?: number;
	stagnationThreshold?: number;
	contextWindowTokens?: number;
	systemPrompt?: string;
	tools?: Tool[];
	onEvent?: EventHandler;
	onHookEvent?: (event: string, ctx: unknown) => void;
	runtimeHooksEnabled?: boolean;
	hookSessionId?: string;
	hookTranscriptPath?: string;
	hooks?: AgentHooks;
	convertToLlm?: (
		messages: import("./types-messages.ts").AgentMessage[],
	) => import("./types-messages.ts").Message[];
	turnEndCallback?: (turnId: string) => void;
	guardsEnabled?: boolean;
	duplicateGuardEnabled?: boolean;
	failureGuardEnabled?: boolean;
	duplicateToolThreshold?: number;
	toolFailureLoopThreshold?: number;
	/** Enable evidence-based no-progress stopping. */
	progressStopEnabled?: boolean;
	proactiveCompactionEnabled?: boolean;
	proactiveCompactionFraction?: number;
	ariadneEnabled?: boolean;
	fffgrepEnabled?: boolean;
	continuationEnabled?: boolean;
	toolExecution?: "sequential" | "parallel";
	steeringQueueMode?: QueueMode;
	followUpQueueMode?: QueueMode;
	thinkingLevel?: ThinkingLevel;
	autoRetryEnabled?: boolean;
	maxRetries?: number;
	retryBaseDelayMs?: number;
	turnTimeoutMs?: number;
	webSearch?: WebSearchConfig;
	cacheSize?: number;
	cacheTtlMs?: number;
	permissions?: PermissionManager;
	onPermissionRequest?: (ctx: {
		toolName: string;
		toolCallId: string;
		args: Record<string, unknown>;
	}) => Promise<"allow" | "deny" | "always">;
	onQuestionRequest?: (
		ctx: import("./types-messages.ts").AskUserContext,
	) => Promise<string>;
	maxTotalTokens?: number;
	/** Hierarchical hard limits for one agent run. */
	runBudget?: RunBudgetLimits;
	// Per-turn stream options managed by the harness.
	streamOptions?: AgentHarnessStreamOptions;
	eventLogPath?: string;
	steeringInterrupt?: boolean;
	acceptance?: AcceptanceConfig;
	// Inference mode (Ctrl+M)
	inferenceMode?: InferenceMode;
	/** Absolute paths allowed in addition to CWD for file tools. */
	allowedPaths?: string[];
	/** When true, skip CWD/allowedPaths enforcement for all file tools. */
	allowAllPaths?: boolean;
	/** Universal output/result truncation limits. Unset fields fall back to DEFAULT_TRUNCATION. */
	truncation?: TruncationConfig;
	/** When true, prefix all bash commands with `rtk` for token savings. */
	rtkProxyEnabled?: boolean;
}

export interface WebSearchConfig {
	baseUrl: string;
	maxResults?: number;
}

// ── Error types ───────────────────────────────────────────────────────────

export enum AgentErrorType {
	TURN_TIMEOUT = "turn_timeout",
	CONTEXT_FULL = "context_full",
	PROVIDER_ERROR = "provider_error",
	ABORTED = "aborted",
	TOOL_EXECUTION_FAILED = "tool_execution_failed",
	TOOL_ARGUMENT_ERROR = "tool_argument_error",
	TOOL_DUPLICATE_CALL = "tool_duplicate_call",
	TOOL_FAILURE_LOOP = "tool_failure_loop",
	HOOK_FAILED = "hook_failed",
	INVALID_CONFIG = "invalid_config",
}

export interface AgentErrorOptions {
	type: AgentErrorType;
	message: string;
	cause?: unknown;
	turnId?: string;
	toolName?: string;
	retryable?: boolean;
}

export class AgentError extends Error {
	readonly type: AgentErrorType;
	readonly cause?: unknown;
	readonly turnId?: string;
	readonly toolName?: string;
	readonly retryable: boolean;

	constructor(options: AgentErrorOptions) {
		super(options.message);
		if (options.cause) {
			Object.defineProperty(this, "cause", {
				value: options.cause,
				writable: true,
				enumerable: false,
			});
		}
		this.name = "AgentError";
		this.type = options.type;
		this.cause = options.cause;
		this.turnId = options.turnId;
		this.toolName = options.toolName;
		this.retryable = options.retryable ?? this.isDefaultRetryable(options.type);
	}

	private isDefaultRetryable(type: AgentErrorType): boolean {
		return (
			type === AgentErrorType.PROVIDER_ERROR ||
			type === AgentErrorType.CONTEXT_FULL
		);
	}
}

export function wrapError(
	type: AgentErrorType,
	original: Error,
	extra?: Partial<AgentErrorOptions>,
): AgentError {
	return new AgentError({
		type,
		message: original.message,
		cause: original,
		...extra,
	});
}

// ── Truncation config ────────────────────────────────────────────────────
// Single source of truth for every "cap this text/output at N" limit in the
// harness. Previously each of these was a separate hardcoded constant
// scattered across tool registry, compaction, subagents, and the TUI
// transcript view.

export interface TruncationConfig {
	/** Cap on tool result content appended to context. 0 disables. */
	toolResultMaxChars?: number;
	/** Cap on lines read by file/search tools (read, find, list-files, grep). */
	maxLines?: number;
	/** Max chars per matched line in grep-style output. */
	grepLineMaxChars?: number;
	/** Cap on subagent report text bubbled up to the parent context. */
	subagentResultMaxChars?: number;
	/** Cap on tool-result text folded into a compaction summary. */
	compactionSummaryMaxChars?: number;
	/** Per-role cap used by micro-compaction when trimming oversized message bodies. */
	microCompactMaxChars?: {
		tool?: number;
		assistant?: number;
		default?: number;
	};
	/** Cap on a single rendered message in the TUI transcript view. */
	transcriptMessageMaxChars?: number;
}

export const DEFAULT_TRUNCATION: Required<
	Omit<TruncationConfig, "microCompactMaxChars">
> & {
	microCompactMaxChars: Required<
		NonNullable<TruncationConfig["microCompactMaxChars"]>
	>;
} = {
	toolResultMaxChars: 100_000,
	maxLines: 2000,
	grepLineMaxChars: 500,
	subagentResultMaxChars: 16_000,
	compactionSummaryMaxChars: 2000,
	microCompactMaxChars: {
		tool: 4000,
		assistant: 10_000,
		default: 14_000,
	},
	transcriptMessageMaxChars: 4000,
};

/** Merge a partial override on top of the defaults, one level deep. */
export function resolveTruncationConfig(
	overrides?: TruncationConfig,
): typeof DEFAULT_TRUNCATION {
	if (!overrides) return DEFAULT_TRUNCATION;
	return {
		...DEFAULT_TRUNCATION,
		...overrides,
		microCompactMaxChars: {
			...DEFAULT_TRUNCATION.microCompactMaxChars,
			...overrides.microCompactMaxChars,
		},
	};
}
