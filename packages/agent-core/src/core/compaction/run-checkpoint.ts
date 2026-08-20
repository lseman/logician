import type { Message } from "../types/types-messages.ts";

export interface CheckpointTask {
	id: string | number;
	subject: string;
	status: string;
}

/** Replace a repeatedly compacted transcript with a small, structured handoff. */
export function resetToRunCheckpoint(
	messages: readonly Message[],
	tasks: readonly CheckpointTask[],
): Message[] {
	const system = messages.find(message => message.role === "system");
	const objective = messages.find(message => message.role === "user");
	const recentEvidence = messages
		.filter(message => message.role === "tool")
		.slice(-6)
		.map(message => String(message.content ?? "").slice(0, 800));
	const lastAssistant = [...messages]
		.reverse()
		.find(message => message.role === "assistant");
	const taskLines = tasks.map(
		task => `- #${task.id} [${task.status}] ${task.subject}`,
	);
	const content = [
		"[autonomous-checkpoint] Continue the same run from this structured handoff.",
		`Objective: ${String(objective?.content ?? "").slice(0, 2_000)}`,
		taskLines.length
			? `Tasks:\n${taskLines.join("\n")}`
			: "Tasks: none recorded",
		recentEvidence.length
			? `Recent tool evidence:\n${recentEvidence.map(item => `- ${item}`).join("\n")}`
			: "Recent tool evidence: none",
		`Last assistant state: ${String(lastAssistant?.content ?? "").slice(0, 1_500)}`,
		"Inspect the workspace as the source of truth. Do not assume unverified work succeeded.",
	].join("\n\n");
	return [
		...(system ? [system] : []),
		{ role: "user" as const, content, timestamp: Date.now() },
	];
}
