import { createSystemMessage } from "../core/messages.ts";
import type {
	AgentHooks,
	AgentMessage,
	Message,
	StopReason,
	ToolCall,
} from "../types/index.ts";

export type EventSink = (
	event: import("../types/index.ts").AgentEvent,
) => Promise<void> | void;

export interface LoopCallbacks {
	getSteeringMessages?: (ctx: {
		messages: Message[];
		iteration: number;
	}) => Promise<Message[] | undefined> | Message[] | undefined;
	getFollowUpMessages?: (ctx: {
		messages: Message[];
		iteration: number;
		assistantText: string;
		stopReason?: StopReason;
	}) => Promise<Message[] | undefined> | Message[] | undefined;
	prepareNextTurn?: (ctx: {
		messages: Message[];
		iteration: number;
		hadToolCalls: boolean;
	}) =>
		| Promise<{ messages?: Message[] } | undefined>
		| { messages?: Message[] }
		| undefined;
	shouldStopAfterTurn?: (ctx: {
		messages: Message[];
		iteration: number;
		hadToolCalls: boolean;
		message?: Message;
		toolResults: Message[];
	}) => Promise<boolean | undefined> | boolean | undefined;
	transformContext?: (ctx: {
		messages: AgentMessage[];
		iteration: number;
		signal?: AbortSignal;
	}) =>
		| Promise<{ messages?: AgentMessage[] } | undefined>
		| { messages?: AgentMessage[] }
		| undefined;
}

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

export async function firstMessages(
	callbacks: Array<
		(() => Promise<Message[] | undefined> | Message[] | undefined) | undefined
	>,
): Promise<Message[]> {
	for (const callback of callbacks) {
		if (!callback) continue;
		const messages = await callback();
		if (messages?.length) return messages;
	}
	return [];
}

export async function transformMessages(
	callbacks: Array<
		| AgentHooks["transformContext"]
		| LoopCallbacks["transformContext"]
		| undefined
	>,
	context: {
		messages: AgentMessage[];
		iteration: number;
		signal?: AbortSignal;
	},
): Promise<AgentMessage[] | undefined> {
	for (const callback of callbacks) {
		const result = await callback?.(context);
		if (result?.messages) return result.messages;
	}
	return undefined;
}

export async function prepareMessages(
	callbacks: Array<
		AgentHooks["prepareNextTurn"] | LoopCallbacks["prepareNextTurn"] | undefined
	>,
	context: {
		messages: Message[];
		iteration: number;
		hadToolCalls: boolean;
	},
): Promise<Message[] | undefined> {
	for (const callback of callbacks) {
		const result = await callback?.(context);
		if (result?.messages) return result.messages;
	}
	return undefined;
}

export async function shouldStop(
	callbacks: Array<
		| AgentHooks["shouldStopAfterTurn"]
		| LoopCallbacks["shouldStopAfterTurn"]
		| undefined
	>,
	context: {
		messages: Message[];
		iteration: number;
		hadToolCalls: boolean;
		message?: Message;
		toolResults: Message[];
	},
): Promise<boolean> {
	for (const callback of callbacks) {
		if ((await callback?.(context)) === true) return true;
	}
	return false;
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
