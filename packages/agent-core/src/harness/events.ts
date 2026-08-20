// ── Event helpers and message pair emission ───────────────────────────────
// Replaces loop/callbacks.ts — streaming event callbacks and message helpers.

import type { Message, StopReason, ToolCall } from "../types/index.ts";
import { createSystemMessage } from "./messages.ts";

export type EventSink = (
	event: import("../types/index.ts").AgentEvent,
) => Promise<void> | void;

export function withSystemPrompt(
	systemPrompt: string | undefined,
	messages: Message[],
): Message[] {
	return [
		createSystemMessage(systemPrompt ?? "You are a helpful assistant."),
		...messages.filter(
			(message): message is Message =>
				message != null && message.role !== "system",
		),
	];
}

export function assistantText(message: Message | undefined): string {
	return message?.role === "assistant" && typeof message.content === "string"
		? message.content
		: "";
}

export function stopReasonFor(
	responseStopReason: "stop" | "length" | "error",
	toolCalls: ToolCall[],
): StopReason {
	if (toolCalls.length > 0) return "tool_calls";
	if (responseStopReason === "length") return "length";
	if (responseStopReason === "error") return "error";
	return "stop";
}

export async function emitMessagePair(
	emit: EventSink,
	turnId: string,
	message: Message,
): Promise<void> {
	await emit({ type: "message_start", turnId, role: message.role });
	await emit({ type: "message_end", turnId, message });
}

export function applyHeaderPatch(
	current: Record<string, string> | undefined,
	patch: Record<string, string | undefined>,
): Record<string, string> {
	const next = { ...(current ?? {}) };
	for (const [name, value] of Object.entries(patch)) {
		if (value === undefined) delete next[name];
		else next[name] = value;
	}
	return next;
}
