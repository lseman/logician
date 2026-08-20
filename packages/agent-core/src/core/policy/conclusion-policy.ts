import { awaitsUserInput, looksComplete } from "../guards/response-patterns.ts";
import type { AgentEventSink, Message } from "../types/types-messages.ts";

export function lastAssistantContent(messages: Message[]): string {
	const assistant = [...messages]
		.reverse()
		.find(message => message.role === "assistant");
	return typeof assistant?.content === "string" ? assistant.content : "";
}

export function lastHadToolCalls(messages: Message[]): boolean {
	const assistant = [...messages]
		.reverse()
		.find(message => message.role === "assistant");
	return Boolean(assistant?.tool_calls?.length);
}

export async function emitConclusion(
	emit: AgentEventSink,
	messages: Message[],
	iteration: number,
	maxIterations: number,
	_hadFollowUps: boolean,
): Promise<void> {
	const text = lastAssistantContent(messages);
	const hadTools = lastHadToolCalls(messages);
	if (hadTools || looksComplete(text) || awaitsUserInput(text)) return;
	if (iteration >= maxIterations) {
		await emit({
			type: "task_failed",
			reason: `Reached ${maxIterations} iteration limit without completing the task. Last response did not signal task completion.`,
			iteration,
			lastContent: text.slice(0, 300),
		});
	}
}
