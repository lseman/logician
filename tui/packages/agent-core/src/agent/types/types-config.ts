// ── Config types ──────────────────────────────────────────────────────────

import type { PermissionManager } from "../../tools/shared/permissions.ts";
import type { AgentStopPolicy, ExecutionProfile } from "../execution-policy.ts";
import type { AcceptanceConfig } from "../guards/acceptance-contract.ts";
import type { EventHandler } from "./types-events.ts";
import type { AgentHooks } from "./types-hooks.ts";
import type { Tool } from "./types-tools.ts";
import type { TruncationConfig } from "./types-truncation.ts";

export type QueueMode = "all" | "one-at-a-time";

/** Self-evaluation / reflection config for the agent loop. */
export interface ReflectionConfig {
	/** Whether to run a self-evaluation step before final conclusion. */
	enabled?: boolean;
	/** Maximum reflection turns allowed. */
	maxReflections?: number;
	/** Reflection prompt template. $task is replaced with the original task description. */
	prompt?: string;
}

export type ThinkingLevel =
	| "off"
	| "minimal"
	| "low"
	| "medium"
	| "high"
	| "xhigh";

export type InferenceMode =
	| "auto"
	| "thinking-general"
	| "thinking-coding"
	| "instruct-general"
	| "instruct-reasoning"
	| "instruct-coding"
	| "deterministic"
	| "creative"
	| "analytical";

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
	 * runner's built-in completion, reflection, acceptance, and repair policies.
	 */
	executionProfile?: ExecutionProfile;
	/** Policies evaluated externally when the agent loop naturally becomes idle. */
	stopPolicies?: AgentStopPolicy[];
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
	internalHooks?: AgentHooks;
	convertToLlm?: (
		messages: import("./types-messages.ts").AgentMessage[],
	) => import("./types-messages.ts").Message[];
	turnEndCallback?: (turnId: string) => void;
	guardsEnabled?: boolean;
	duplicateGuardEnabled?: boolean;
	failureGuardEnabled?: boolean;
	duplicateToolThreshold?: number;
	toolFailureLoopThreshold?: number;
	budgetStopEnabled?: boolean;
	proactiveCompactionEnabled?: boolean;
	proactiveCompactionFraction?: number;
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
		ctx: import("./types-tools.ts").AskUserContext,
	) => Promise<string>;
	maxTotalTokens?: number;
	// Per-turn stream options managed by the harness.
	streamOptions?: AgentHarnessStreamOptions;
	eventLogPath?: string;
	steeringInterrupt?: boolean;
	acceptance?: AcceptanceConfig;
	/** Self-evaluation / reflection config. */
	reflectionConfig?: ReflectionConfig;
	// Thinking loop detection
	thinkingLoopDetectionEnabled?: boolean;
	thinkingLoopMinThinkingLength?: number;
	thinkingLoopThinkingOnlyThreshold?: number;
	thinkingLoopEscalationRatio?: number;
	thinkingLoopMaxTotalThinkingTokens?: number;
	thinkingLoopMetaReasoningThreshold?: number;
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

// ── Acceptance Contract types ─────────────────────────────────────────────
// Re-exported so consumers that import from types.ts get everything in one place.

export type {
	AcceptanceCriterion,
	AcceptanceLedger,
	AcceptanceLevel,
	AcceptanceReport,
	AcceptanceReview,
	AcceptanceVerification,
	CriterionSeverity,
	EvidenceKind,
	ResolvedAcceptance,
} from "../guards/acceptance-contract.ts";
