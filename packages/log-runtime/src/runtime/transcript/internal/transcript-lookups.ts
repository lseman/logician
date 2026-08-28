import type { AssistantChunk, ChildChunk } from "../model.ts";

export function findLastToolChunk(
	chunks: AssistantChunk[],
	predicate: (chunk: AssistantChunk) => boolean,
): AssistantChunk | undefined {
	return chunks.findLast(
		chunk =>
			chunk.type === "tool" && chunk.tool !== undefined && predicate(chunk),
	);
}

export function findLastChildToolChunk(
	chunks: ChildChunk[],
	toolCallId: string,
): ChildChunk | undefined {
	return chunks.findLast(
		chunk => chunk.type === "tool" && chunk.tool?.toolCallId === toolCallId,
	);
}
