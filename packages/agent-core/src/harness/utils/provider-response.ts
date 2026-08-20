/**
 * Post-LLM response processing.
 *
 * Handles everything after the backend.generate() call returns:
 * - Text tool call fallback parsing
 * - Stop reason mapping
 * - afterProviderResponse hook chain
 * - Assistant message creation (sanitized arguments)
 * - Event emission (message_start/update/end)
 *
 * Returns a result indicating whether processing succeeded or failed.
 * The runner decides how to handle failures (return finish vs continue).
 */

import type {
	AgentConfig,
	AgentEventSink,
	AgentMessage,
	Message,
	StopReason,
	ToolCall,
} from "../../types/index.ts";
import { stopReasonFor } from "../events.ts";
import {
	createAssistantMessage,
	sanitizeToolCallArguments,
} from "../messages.ts";
import type { ToolRegistry } from "../tools/registry.ts";
import {
	parseTextToolCalls,
	stripTextToolCalls,
} from "../tools/text-to-tool-calls.ts";
import type { LLMResponse } from "../utils/backend.ts";

export interface ProcessResponseResult {
	success: boolean;
	toolCalls: ToolCall[];
	assistantContent: string;
	stopReason: StopReason;
	assistant: Message;
	performedToolWork: boolean;
	errorMessage?: string;
}

export interface ProcessResponseContext {
	response: LLMResponse | undefined;
	registry: ToolRegistry;
	messages: AgentMessage[];
	newMessages: Message[];
	turnId: string;
	iteration: number;
	emit: AgentEventSink;
	config: AgentConfig;
}

/**
 * Process the LLM response: parse, validate, emit events, and return
 * structured data for tool execution.
 *
 * Returns success=false with an errorMessage on error stop reason or
 * output guard abort. The caller decides how to handle the failure.
 */
export function processProviderResponse(
	ctx: ProcessResponseContext,
): ProcessResponseResult {
	const {
		response,
		registry,
		messages,
		newMessages,
		turnId,
		iteration,
		emit,
		config,
	} = ctx;

	// Extract tool calls (with text fallback)
	let toolCalls: ToolCall[] = response?.toolCalls ?? [];
	let assistantContent = response?.content ?? "";

	if (toolCalls.length === 0 && response?.content) {
		const textCalls = parseTextToolCalls(response.content, (name: string) =>
			registry.has(name),
		);
		if (textCalls.length > 0) {
			toolCalls = textCalls;
			assistantContent = stripTextToolCalls(response.content);
		}
	}

	const performedToolWork = toolCalls.some(
		(call: ToolCall) => call.name !== "task_status",
	);

	const rawStopReason =
		(response?.stopReason as "stop" | "length" | "error") ?? "stop";
	const stopReason = stopReasonFor(rawStopReason, toolCalls);

	// afterProviderResponse hook
	config.hooks?.afterProviderResponse?.({
		model: config.model ?? "",
		content: assistantContent,
		toolCallCount: toolCalls.length,
		stopReason,
		usageTokens: response?.usage?.totalTokens,
		iteration,
	});

	// Create assistant message with sanitized arguments
	const assistant = createAssistantMessage(
		assistantContent,
		sanitizeToolCallArguments(toolCalls),
	);
	messages.push(assistant);
	newMessages.push(assistant);

	emit({ type: "message_start", turnId, role: "assistant" });
	emit({ type: "message_update", turnId, message: assistant });
	emit({ type: "message_end", turnId, message: assistant });

	if (rawStopReason === "error") {
		const errorMessage = response?.errorMessage ?? "Model request failed";
		emit({ type: "error", message: errorMessage });
		emit({
			type: "turn_end",
			turnId,
			stopReason,
			message: assistant,
			toolResults: [],
		});
		return {
			success: false,
			toolCalls,
			assistantContent,
			stopReason,
			assistant,
			performedToolWork,
			errorMessage,
		};
	}

	return {
		success: true,
		toolCalls,
		assistantContent,
		stopReason,
		assistant,
		performedToolWork,
	};
}
