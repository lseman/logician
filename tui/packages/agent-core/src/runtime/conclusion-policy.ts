import type { AgentEvent, Message } from "../core/types.ts";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

const NON_COMMITTAL_PATTERNS = [
	/\b(i\s+(need|should|have|might|could|will)\s+(to\s+)?(?:check|look|think|consider|analyze|investigate|examine|review|verify))\b/i,
	/\b(let\s+me\s+(think|see|check|try|consider))\b/i,
	/\b(i'm\s+(going\s+to|thinking\s+about|not\s+sure|still\s+considering))\b/i,
	/\b(i'll\s+(try|check|look|see|think))\b/i,
	/\b(need\s+to\s+(check|think|verify|confirm))\b/i,
	/\b(however|but|although)\s+(i\s+(need|should|have|might))\b/i,
	/\b(this\s+(requires|needs|demands|warrants)\s+(further|more|additional))\b/i,
	/\b(i\s+(don't|do\s+not)\s+(know|think\s+|certain))\b/i,
	/\blet(?:'s|\s+me)\s+(?:step\s+back|circle\s+back|reconsider)\b/i,
	/\b(at\s+this\s+point|so\s+far)\s+(i\s+(have|can|see)|we\s+(need|should))\b/i,
];

const COMPLETE_PATTERNS = [
	/\b(task\s+complete|all\s+done|finished|completed\s+successfully|nothing\s+(else|more)\s+to\s+do|no\s+(further|more)\s+(steps?|action|work)|that('s|\s+is)\s+(all|done|complete))\b/i,
	/^done\s*$/i,
];

export function looksComplete(text: string): boolean {
	return Boolean(text) && COMPLETE_PATTERNS.some((pattern) => pattern.test(text));
}

export function looksNonCommittal(text: string): boolean {
	return text.trim().length >= 10 && NON_COMMITTAL_PATTERNS.some((pattern) => pattern.test(text));
}

export function lastAssistantContent(messages: Message[]): string {
	const assistant = [...messages].reverse().find((message) => message.role === "assistant");
	return typeof assistant?.content === "string" ? assistant.content : "";
}

export function lastHadToolCalls(messages: Message[]): boolean {
	const assistant = [...messages].reverse().find((message) => message.role === "assistant");
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
	if (hadTools || looksComplete(text)) return;
	if (looksNonCommittal(text) && !hadFollowUps) {
		await emit({ type: "task_failed", reason: `Model stopped with non-committal text after ${iteration} iteration(s). It did not complete the task or produce actionable output.`, iteration, lastContent: text.slice(0, 300) });
		return;
	}
	if (iteration >= maxIterations) {
		await emit({ type: "task_failed", reason: `Reached ${maxIterations} iteration limit without completing the task. ${looksNonCommittal(text) ? "Last response was non-committal." : "Last response did not signal task completion."}`, iteration, lastContent: text.slice(0, 300) });
	}
}
