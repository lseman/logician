import type {
	ChildChunk,
	ChildToolCall,
	SpawnTaskStatus,
	ToolExecution,
} from "../model.ts";

function ensureDetails(tool: ToolExecution): Record<string, unknown> {
	if (!tool.details || Array.isArray(tool.details)) {
		tool.details = {};
	}
	return tool.details;
}

function ensureArray<T>(details: Record<string, unknown>, key: string): T[] {
	const current = details[key];
	if (Array.isArray(current)) return current as T[];
	const value: T[] = [];
	details[key] = value;
	return value;
}

export function getToolDetails(tool: ToolExecution): Record<string, unknown> {
	return ensureDetails(tool);
}

export function getChildChunks(tool: ToolExecution): ChildChunk[] {
	return ensureArray<ChildChunk>(ensureDetails(tool), "childChunks");
}

export function getChildToolCalls(tool: ToolExecution): ChildToolCall[] {
	return ensureArray<ChildToolCall>(ensureDetails(tool), "childToolCalls");
}

export function getSpawnTaskStatuses(
	tool: ToolExecution,
): Record<number, SpawnTaskStatus> {
	const details = ensureDetails(tool);
	const current = details.taskStatus;
	if (current && typeof current === "object" && !Array.isArray(current)) {
		return current as Record<number, SpawnTaskStatus>;
	}
	const value: Record<number, SpawnTaskStatus> = {};
	details.taskStatus = value;
	return value;
}
