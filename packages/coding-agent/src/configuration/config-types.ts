// ── LogicianTuiConfig sub-config types ────────────────────────────────────────
// Domain-specific config chunks extracted from LogicianTuiConfig to improve
// type safety and reduce cognitive load for consumers.
//
// These can be used to type config sections, merge partials, or validate
// individual domains without importing the full 127-property interface.

import type {
	AgentModelConfig,
	InferenceMode,
	ThinkingLevel,
	TruncationConfig,
} from "@logician/agent-core";

// Re-export canonical types so consumers don't need two import paths.
export type InferenceModeValue = InferenceMode;
export type ThinkingLevelValue = ThinkingLevel;

// ── Model settings (subset of LogicianTuiConfig) ─────────────────────────────

// ── Model settings (subset of LogicianTuiConfig) ─────────────────────────────

export interface ModelSettings {
	baseUrl?: string;
	llmUrl?: string;
	model?: string;
	models?: AgentModelConfig[];
	systemPrompt?: string;
	chatTemplate?: string;
	temperature?: number;
	maxTokens?: number;
	maxIterations?: number;
	contextWindow?: number;
	contextWindowTokens?: number;
	maxTotalTokens?: number;
	inferenceMode?: InferenceModeValue;
	thinkingLevel?: ThinkingLevelValue;
	executionProfile?: "autonomous" | "minimal";
}

// ── Guard / Safeguard settings ────────────────────────────────────────────────

export interface GuardSettings {
	guardsEnabled?: boolean;
	duplicateGuardEnabled?: boolean;
	failureGuardEnabled?: boolean;
	duplicateToolThreshold?: number;
	toolFailureLoopThreshold?: number;
	budgetStopEnabled?: boolean;
	continuationEnabled?: boolean;
	postEditDiagnostics?: boolean;
}

// ── Runtime settings ──────────────────────────────────────────────────────────

export interface RuntimeSettings {
	hooks?: boolean;
	autoRetryEnabled?: boolean;
	maxRetries?: number;
	retryBaseDelayMs?: number;
	turnTimeoutMs?: number;
	cacheSize?: number;
	cacheTtlMs?: number;
	steeringInterrupt?: boolean;
}

// ── Tool / Permission settings ────────────────────────────────────────────────

export interface ToolSettings {
	toolExecution?: "sequential" | "parallel";
	permissionMode?: "acceptAll" | "acceptEdits" | "ask" | "plan";
	permissions?: { allow?: string[]; deny?: string[] };
	allowedPaths?: string[];
	allowAllPaths?: boolean;
	cwd?: string;
}

// ── Memory settings ───────────────────────────────────────────────────────────

export interface MemorySettings {
	memory?: boolean;
	memoryDbPath?: string;
	memoryExtractorModel?: string;
	memoryExtractor?: { baseUrl?: string; model?: string };
	memoryViewer?: boolean;
	memoryViewerPort?: number;
	memoryEmbeddings?: boolean;
	memoryEmbeddingModel?: string;
}

// ── Transcript settings ───────────────────────────────────────────────────────

export interface TranscriptSettings {
	transcriptMaxTurns?: number;
	transcriptMaxRenderedLines?: number;
}

// ── Reasoner settings ─────────────────────────────────────────────────────────

export interface ReasonerSettings {
	reasoner?: string;
	reasonerConfig?: Record<string, unknown>;
}

// ── LSP settings ──────────────────────────────────────────────────────────────

export interface LspSettings {
	enabled?: boolean;
	timeoutMs?: number;
	serverOverrides?: Record<
		string,
		{ command: string; args?: string[]; languageId: string }
	>;
}

// ── Compaction settings ───────────────────────────────────────────────────────

export interface CompactionSettings {
	enabled?: boolean;
	reserveTokens?: number;
	keepRecentTokens?: number;
}

// ── Web Search settings ───────────────────────────────────────────────────────

export interface WebSearchSettings {
	baseUrl?: string;
	maxResults?: number;
}

// ── Reflection settings ───────────────────────────────────────────────────────

export interface ReflectionSettings {
	enabled?: boolean;
	maxReflections?: number;
	prompt?: string;
}
