// ── Hook types ────────────────────────────────────────────────────────────

import type { ExplicitTaskState } from "../tasks/task-state-controller.ts";
import type { AgentHarnessStreamOptions } from "./types-config.ts";
import type { AgentMessage, Message, StopReason } from "./types-messages.ts";
import type { ToolCall } from "./types-tools.ts";

export interface BeforeToolCallContext {
	toolCall: ToolCall;
	args: Record<string, unknown>;
	iteration: number;
}

export interface BeforeToolCallResult {
	content?: string;
	isError?: boolean;
	args?: Record<string, unknown>;
}

export interface PreToolUseContext {
	toolCall: ToolCall;
	args: Record<string, unknown>;
	iteration: number;
}

export interface PreToolUseResult {
	additionalContext?: string;
	error?: string;
}

export interface AfterToolCallContext {
	toolCall: ToolCall;
	args: Record<string, unknown>;
	result: string;
	isError: boolean;
	iteration: number;
}

export interface AfterToolCallResult {
	content?: string;
	isError?: boolean;
	terminate?: boolean;
}

export interface PrepareNextTurnContext {
	messages: Message[];
	iteration: number;
	hadToolCalls: boolean;
}

export interface PrepareNextTurnResult {
	messages: Message[];
}

export interface ShouldStopAfterTurnContext {
	messages: Message[];
	iteration: number;
	hadToolCalls: boolean;
}

export interface GetSteeringMessagesContext {
	messages: Message[];
	iteration: number;
}

export interface TransformContext {
	messages: AgentMessage[];
	iteration: number;
	signal?: AbortSignal;
	/** Live structured task state for task-aware retrieval and context shaping. */
	taskState?: ExplicitTaskState;
}

export interface BeforeProviderRequestContext {
	model: string;
	sessionId: string;
	iteration: number;
	streamOptions: AgentHarnessStreamOptions;
}

export interface BeforeProviderRequestResult {
	/** Header patch. undefined values delete keys; explicit headers: undefined clears all. */
	headers?: Record<string, string | undefined>;
	/** Timeout override in milliseconds. */
	timeoutMs?: number;
	/** Max retry attempts override. */
	maxRetries?: number;
	/** Cache retention hint (e.g., "transient", "persistent"). */
	cacheRetention?: string;
	/** Additional request headers merged with auth and lifecycle headers. */
	metadata?: Record<string, unknown>;
	/** Provider metadata forwarded with requests. */
	transport?: string;
}

export interface BeforeProviderPayloadContext {
	model: string;
	payload: Record<string, unknown>;
}

export interface BeforeProviderPayloadResult {
	payload: Record<string, unknown>;
}

export interface AfterProviderResponseContext {
	model: string;
	content: string;
	toolCallCount: number;
	stopReason: StopReason;
	usageTokens?: number;
	iteration: number;
}

export interface TransformContextResult {
	messages: AgentMessage[];
}

export interface GetFollowUpMessagesContext {
	messages: Message[];
	iteration: number;
	assistantText: string;
	stopReason?: StopReason;
}

export interface BeforeCompactContext {
	/** Messages that will be summarized. */
	messages: Message[];
	/** Estimated token count before compaction. */
	tokensBefore: number;
	/** "manual" = explicit compact() call; "auto" = threshold-triggered. */
	reason: "manual" | "auto";
}

export interface BeforeCompactResult {
	/** Return true to skip compaction entirely. */
	cancel?: boolean;
	/** Provide a pre-built summary to use instead of generating one. */
	summary?: string;
}

export interface BeforeAgentStartContext {
	prompt: string;
	systemPrompt: string;
	messages: AgentMessage[];
}

export interface BeforeAgentStartResult {
	messages?: AgentMessage[];
	systemPrompt?: string;
}

export interface AgentHooks {
	beforeAgentStart?: (
		ctx: BeforeAgentStartContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeAgentStartResult | undefined>
		| BeforeAgentStartResult
		| undefined;
	beforeToolCall?: (
		ctx: BeforeToolCallContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeToolCallResult | undefined>
		| BeforeToolCallResult
		| undefined;
	afterToolCall?: (
		ctx: AfterToolCallContext,
		signal?: AbortSignal,
	) =>
		| Promise<AfterToolCallResult | undefined>
		| AfterToolCallResult
		| undefined;
	prepareNextTurn?: (
		ctx: PrepareNextTurnContext,
		signal?: AbortSignal,
	) =>
		| Promise<PrepareNextTurnResult | undefined>
		| PrepareNextTurnResult
		| undefined;
	transformContext?: (
		ctx: TransformContext,
		signal?: AbortSignal,
	) =>
		| Promise<TransformContextResult | undefined>
		| TransformContextResult
		| undefined;
	beforeProviderRequest?: (
		ctx: BeforeProviderRequestContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeProviderRequestResult | undefined>
		| BeforeProviderRequestResult
		| undefined;
	beforeProviderPayload?: (
		ctx: BeforeProviderPayloadContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeProviderPayloadResult | undefined>
		| BeforeProviderPayloadResult
		| undefined;
	afterProviderResponse?: (
		ctx: AfterProviderResponseContext,
		signal?: AbortSignal,
	) => Promise<void> | void;
	shouldStopAfterTurn?: (
		ctx: ShouldStopAfterTurnContext,
		signal?: AbortSignal,
	) => Promise<boolean | undefined> | boolean | undefined;
	getSteeringMessages?: (
		ctx: GetSteeringMessagesContext,
		signal?: AbortSignal,
	) => Promise<Message[] | undefined> | Message[] | undefined;
	getFollowUpMessages?: (
		ctx: GetFollowUpMessagesContext,
		signal?: AbortSignal,
	) => Promise<Message[] | undefined> | Message[] | undefined;
	beforeCompact?: (
		ctx: BeforeCompactContext,
		signal?: AbortSignal,
	) =>
		| Promise<BeforeCompactResult | undefined>
		| BeforeCompactResult
		| undefined;
}
