// ── Hook types ────────────────────────────────────────────────────────────

import type { AgentHarnessStreamOptions } from "./types-config.ts";
import type { Message, AgentMessage, StopReason } from "./types-messages.ts";
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

export interface AgentHooks {
	beforeToolCall?: (
		ctx: BeforeToolCallContext,
	) =>
		| Promise<BeforeToolCallResult | undefined>
		| BeforeToolCallResult
		| undefined;
	afterToolCall?: (
		ctx: AfterToolCallContext,
	) =>
		| Promise<AfterToolCallResult | undefined>
		| AfterToolCallResult
		| undefined;
	prepareNextTurn?: (
		ctx: PrepareNextTurnContext,
	) =>
		| Promise<PrepareNextTurnResult | undefined>
		| PrepareNextTurnResult
		| undefined;
	transformContext?: (
		ctx: TransformContext,
	) =>
		| Promise<TransformContextResult | undefined>
		| TransformContextResult
		| undefined;
	beforeProviderRequest?: (
		ctx: BeforeProviderRequestContext,
	) =>
		| Promise<BeforeProviderRequestResult | undefined>
		| BeforeProviderRequestResult
		| undefined;
	beforeProviderPayload?: (
		ctx: BeforeProviderPayloadContext,
	) =>
		| Promise<BeforeProviderPayloadResult | undefined>
		| BeforeProviderPayloadResult
		| undefined;
	afterProviderResponse?: (
		ctx: AfterProviderResponseContext,
	) => Promise<void> | void;
	shouldStopAfterTurn?: (
		ctx: ShouldStopAfterTurnContext,
	) => Promise<boolean | undefined> | boolean | undefined;
	getSteeringMessages?: (
		ctx: GetSteeringMessagesContext,
	) => Promise<Message[] | undefined> | Message[] | undefined;
	getFollowUpMessages?: (
		ctx: GetFollowUpMessagesContext,
	) => Promise<Message[] | undefined> | Message[] | undefined;
	beforeCompact?: (
		ctx: BeforeCompactContext,
	) =>
		| Promise<BeforeCompactResult | undefined>
		| BeforeCompactResult
		| undefined;
}
