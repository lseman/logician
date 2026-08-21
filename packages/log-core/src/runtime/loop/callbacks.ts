import { createSystemMessage } from "../../capabilities/provider/messages.ts";
import type {
	AgentEvent,
	Message,
	StopReason,
	ToolCall,
} from "../../system/types/types-messages.ts";

export type EventSink = (event: AgentEvent) => Promise<void> | void;

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

export function lastAssistantContent(messages: readonly Message[]): string {
	return assistantText(
		[...messages].reverse().find(message => message.role === "assistant"),
	);
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
