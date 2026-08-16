/**
 * Post-LLM response processing.
 *
 * Handles everything after the backend.generate() call returns:
 * - Text tool call fallback parsing
 * - Stop reason mapping
 * - afterProviderResponse hook chain
 * - Output guard check for empty/degenerate responses
 * - Assistant message creation (sanitized arguments)
 * - Event emission (message_start/update/end across both buses)
 *
 * Returns a result indicating whether processing succeeded or failed.
 * The runner decides how to handle failures (return finish vs continue).
 */

import type { LLMResponse } from "../backend.ts";
import type { OutputGuard } from "../guards/output-guard.ts";
import type { ToolRegistry } from "../../tools/shared/registry.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentEventSink,
	AgentMessage,
	Message,
	StopReason,
	ToolCall,
} from "../types.ts";
import type { ExtensionEventBus } from "../../hooks/extensions/event-bus.ts";
import type { ExtensionEvent as TypedExtensionEvent } from "../../hooks/extensions/events.ts";

import {
	createAssistantMessage,
	sanitizeToolCallArguments,
} from "../messages.ts";
import {
	parseTextToolCalls,
	stripTextToolCalls,
} from "../../tools/shared/text-to-tool-calls.ts";
import { emitMessagePair, stopReasonFor } from "./callbacks.ts";

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
	outputGuard: OutputGuard | null;
	messages: AgentMessage[];
	newMessages: Message[];
	turnId: string;
	iteration: number;
	emit: AgentEventSink;
	config: AgentConfig;
	extensionBus?: ExtensionEventBus;
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
		outputGuard,
		messages,
		newMessages,
		turnId,
		iteration,
		emit,
		config,
		extensionBus,
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

	const performedToolWork = toolCalls.some((call: ToolCall) => call.name !== "task_status");

	const rawStopReason =
		(response?.stopReason as "stop" | "length" | "error") ?? "stop";
	const stopReason = stopReasonFor(rawStopReason, toolCalls);

	// afterProviderResponse hooks
	const responseHooks = [
		config.internalHooks?.afterProviderResponse,
		config.hooks?.afterProviderResponse,
	];
	for (const hook of responseHooks) {
		hook?.({
			model: config.model ?? "",
			content: assistantContent,
			toolCallCount: toolCalls.length,
			stopReason,
			usageTokens: response?.usage?.totalTokens,
			iteration,
		});
	}

	// Output guard: check for empty/degenerate responses
	if (outputGuard) {
		const guardCheck = outputGuard.checkResponse(
			assistantContent || null,
			toolCalls.length,
		);
		if (guardCheck.action === "abort") {
			const message = guardCheck.message ?? "Model returned empty response.";
			emit({ type: "error", message });
			return {
				success: false,
				toolCalls: [],
				assistantContent: "",
				stopReason: "error",
				assistant: { role: "assistant", content: message } as Message,
				performedToolWork: false,
				errorMessage: message,
			};
		}
	}

	// Create assistant message with sanitized arguments
	const assistant = createAssistantMessage(
		assistantContent,
		sanitizeToolCallArguments(toolCalls),
	);
	messages.push(assistant);
	newMessages.push(assistant);

	// Emit events (both typed and untyped paths)
	emitTyped(extensionBus, { type: "message_start", message: assistant });
	emit({ type: "message_start", turnId, role: "assistant" });
	emit({ type: "message_update", turnId, message: assistant });
	emitTyped(extensionBus, { type: "message_update", message: assistant });
	emit({ type: "message_end", turnId, message: assistant });
	emitTyped(extensionBus, { type: "message_end", message: assistant });

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

/**
 * Helper to emit a typed extension event if the bus is available.
 */
function emitTyped(
	bus: ExtensionEventBus | undefined,
	event: TypedExtensionEvent,
): void {
	if (!bus) return;
	void bus.emit(event);
}
