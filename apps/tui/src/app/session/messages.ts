import type { Message } from "@logician/agent-core";
import type { Turn } from "@logician/coding-agent/sessions";

export function turnsToMessages(turns: Turn[]): Message[] {
	const messages: Message[] = [];
	for (const turn of turns) {
		if (turn.userMessage?.content) {
			messages.push({ role: "user", content: turn.userMessage.content });
		}
		const assistantText = (turn.assistantMessage?.chunks ?? [])
			.filter(chunk => chunk.type === "content" && chunk.contentText)
			.map(chunk => chunk.contentText)
			.join("");
		if (assistantText) {
			messages.push({ role: "assistant", content: assistantText });
		}
	}
	return messages;
}
