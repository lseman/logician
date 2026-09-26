import type {
	AssistantChunk,
	SessionState,
	ToolExecution,
	Turn,
} from "./model.ts";

export function selectCurrentTurn(state: SessionState): Turn | undefined {
	if (!state.currentTurnId) return undefined;
	return state.turns.find(turn => turn.id === state.currentTurnId);
}

export function selectThinkingChunks(turn: Turn | undefined): AssistantChunk[] {
	return (
		turn?.assistantMessage?.chunks.filter(chunk => chunk.type === "thinking") ??
		[]
	);
}

export function selectStreamingContent(turn: Turn | undefined): string | null {
	if (!turn?.assistantMessage) return null;
	const text = turn.assistantMessage.chunks
		.filter(chunk => chunk.type === "content" && !chunk.isComplete)
		.map(chunk => chunk.contentText)
		.join("");
	return text || null;
}

export function selectStreamingThinking(turn: Turn | undefined): string[] {
	return selectThinkingChunks(turn)
		.map(chunk => chunk.contentText || "")
		.filter(Boolean);
}

export function selectAssistantThinking(turn: Turn): string | null {
	const thinking = selectThinkingChunks(turn)
		.map(chunk => chunk.contentText || "")
		.filter(Boolean);
	return thinking.length > 0 ? thinking.join("\n\n") : null;
}

export function selectAssistantContent(turn: Turn): string | null {
	const text =
		turn.assistantMessage?.chunks
			.filter(chunk => chunk.type === "content")
			.map(chunk => chunk.contentText)
			.join("") ?? "";
	return text.length > 0 ? text : null;
}

export function selectAssistantTools(turn: Turn): ToolExecution[] {
	return (
		turn.assistantMessage?.chunks.flatMap(chunk =>
			chunk.type === "tool" && chunk.tool ? [chunk.tool] : [],
		) ?? []
	);
}

export function selectMessageCount(turns: readonly Turn[]): number {
	return turns.reduce(
		(count, turn) => count + 1 + (turn.assistantMessage ? 1 : 0),
		0,
	);
}
