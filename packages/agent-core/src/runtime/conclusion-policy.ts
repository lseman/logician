import {
	awaitsUserInput,
	looksComplete,
	looksNonCommittal,
} from "../agent/guards/response-patterns.ts";
import type { AgentEvent, Message } from "../agent/types.ts";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

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
	hadFollowUps: boolean,
): Promise<void> {
	const text = lastAssistantContent(messages);
	const hadTools = lastHadToolCalls(messages);
	if (hadTools || looksComplete(text) || awaitsUserInput(text)) return;
	if (looksNonCommittal(text) && !hadFollowUps) {
		await emit({
			type: "task_failed",
			reason: `Model stopped with non-committal text after ${iteration} iteration(s). It did not complete the task or produce actionable output.`,
			iteration,
			lastContent: text.slice(0, 300),
		});
		return;
	}
	if (iteration >= maxIterations) {
		await emit({
			type: "task_failed",
			reason: `Reached ${maxIterations} iteration limit without completing the task. ${looksNonCommittal(text) ? "Last response was non-committal." : "Last response did not signal task completion."}`,
			iteration,
			lastContent: text.slice(0, 300),
		});
	}
}
