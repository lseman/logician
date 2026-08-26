import type { Message } from "@logician/log-core";
import {
	estimateChatPayloadTokens,
	estimateTokens,
} from "@logician/log-core/runtime";

type ToolDefinitions = Parameters<typeof estimateChatPayloadTokens>[1];

export interface ContextSource {
	name: string;
	tokens: number;
	detail: string;
}

export interface ContextInspection {
	text: string;
	tokens: number;
	sources: ContextSource[];
}

export interface ContextInspectionInput {
	messages: Message[];
	systemPrompt: string;
	memoryContext: string;
	toolDefinitions: ToolDefinitions;
}

/** Formats and measures the exact context zones shown by runtime inspection. */
export function inspectContext(
	input: ContextInspectionInput,
): ContextInspection {
	const { messages, systemPrompt, memoryContext, toolDefinitions } = input;
	const tokens =
		estimateChatPayloadTokens(messages, toolDefinitions) +
		estimateTokens(memoryContext);
	const sources = contextSources(input);
	const sourceLines = sources.map(
		zone =>
			`- ${zone.name}: ~${zone.tokens} tokens${zone.detail ? ` — ${zone.detail}` : ""}`,
	);
	const lines: string[] = ["## Prompt source map", "", ...sourceLines, ""];
	lines.push("## Effective context", "");

	if (!messages.length && !systemPrompt && !memoryContext) {
		lines.push("No messages yet.");
	}
	if (systemPrompt) {
		lines.push("[SYSTEM] system prompt", systemPrompt, "");
	}
	if (memoryContext) {
		lines.push("[SYSTEM] Memory Context", memoryContext, "");
	}

	for (const message of messages) {
		if (!message || message.role === "system") continue;
		const timestamp = message.timestamp
			? ` ${new Date(message.timestamp).toISOString()}`
			: "";
		const header = `[${message.role.toUpperCase()}]${timestamp}`;

		if (message.role === "tool" && message.content) {
			const toolName = findToolName(messages, message.tool_call_id || "");
			lines.push(`${header} (${toolName})\n${message.content}`);
		} else if (message.role === "assistant" && message.tool_calls?.length) {
			lines.push(
				`${header}\n${message.content || "(no content)"}\n\nTool calls:`,
			);
			for (const call of message.tool_calls) {
				lines.push(`  - ${call.name}(${call.arguments || ""})`);
			}
		} else {
			lines.push(`${header}\n${message.content || ""}`);
		}
		lines.push("");
	}

	return {
		text: `## Context (${messages.length} messages, ~${tokens} tokens)\n\n${lines.join("\n")}`,
		tokens,
		sources,
	};
}

export function contextSources(input: ContextInspectionInput): ContextSource[] {
	const { messages, systemPrompt, memoryContext, toolDefinitions } = input;
	const conversation = messages.filter(message => message.role !== "tool");
	const toolEvidence = messages.filter(message => message.role === "tool");
	return [
		{
			name: "Base instructions",
			tokens: estimateTokens(systemPrompt),
			detail: "system zone",
		},
		{
			name: "Plugin context",
			tokens: 0,
			detail: "startup hooks",
		},
		{
			name: "Tool definitions",
			tokens: estimateChatPayloadTokens([], toolDefinitions),
			detail: `${toolDefinitions?.length ?? 0} tools`,
		},
		{
			name: "Retrieved memory",
			tokens: estimateTokens(memoryContext),
			detail: memoryContext ? "request-time compact index" : "none retrieved",
		},
		{
			name: "Conversation",
			tokens: conversation.length ? estimateChatPayloadTokens(conversation) : 0,
			detail: `${conversation.length} messages`,
		},
		{
			name: "Tool evidence",
			tokens: toolEvidence.length ? estimateChatPayloadTokens(toolEvidence) : 0,
			detail: `${toolEvidence.length} results`,
		},
	].filter(zone => zone.tokens > 0 || zone.name === "Conversation");
}

function findToolName(messages: Message[], callId: string): string {
	return (
		messages
			.find(
				message =>
					message.role === "assistant" &&
					message.tool_calls?.some(call => call.id === callId),
			)
			?.tool_calls?.find(call => call.id === callId)?.name || "tool"
	);
}
