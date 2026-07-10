// ── Config types ──────────────────────────────────────────────────────────

import type { AgentHooks } from "./types-hooks.ts";
import type { EventHandler } from "./types-events.ts";
import type { Tool } from "./types-tools.ts";
import type { PermissionManager } from "../../tools/shared/permissions.ts";

export type QueueMode = "all" | "one-at-a-time";

export type ThinkingLevel =
	| "off"
	| "minimal"
	| "low"
	| "medium"
	| "high"
	| "xhigh";

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
	duplicateToolThreshold?: number;
	toolFailureLoopThreshold?: number;
	budgetStopEnabled?: boolean;
	proactiveCompactionEnabled?: boolean;
	proactiveCompactionFraction?: number;
	continuationEnabled?: boolean;
	loopDetectionEnabled?: boolean;
	toolExecution?: "sequential" | "parallel";
	steeringQueueMode?: QueueMode;
	followUpQueueMode?: QueueMode;
	thinkingLevel?: ThinkingLevel;
	/** @deprecated Reasoner system removed. This field is ignored. */
	reasonerId?: string;
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
	// Thinking loop detection
	thinkingLoopDetectionEnabled?: boolean;
	thinkingLoopMinThinkingLength?: number;
	thinkingLoopThinkingOnlyThreshold?: number;
	thinkingLoopEscalationRatio?: number;
	thinkingLoopMaxTotalThinkingTokens?: number;
	thinkingLoopMetaReasoningThreshold?: number;
}

export interface WebSearchConfig {
	baseUrl: string;
	maxResults?: number;
}

// ── Acceptance Contract types ─────────────────────────────────────────────

export type EvidenceKind =
	| "changed-files"
	| "tests-added"
	| "commands-run"
	| "validation-output"
	| "residual-risks"
	| "no-staged-files"
	| "diff-summary"
	| "review-findings"
	| "manual-notes";

export interface AcceptanceCriterion {
	id?: string;
	must: string;
	evidence?: EvidenceKind[];
	severity?: "required" | "recommended";
}

export interface AcceptanceVerification {
	id: string;
	command: string;
	cwd?: string;
	timeoutMs?: number;
	allowFailure?: boolean;
}

export interface AcceptanceReview {
	agent: string;
	focus?: string;
	required?: boolean;
}

export interface AcceptanceConfig {
	criteria?: string[] | AcceptanceCriterion[];
	evidence?: EvidenceKind[];
	verify?: AcceptanceVerification[];
	review?: AcceptanceReview;
	stopRules?: string[];
	maxFinalizationTurns?: number;
}
