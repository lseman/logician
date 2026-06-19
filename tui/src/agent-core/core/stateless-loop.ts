// ── Stateless loop facade ─────────────────────────────────────────────────
// Compatibility bridge toward a Pi-style loop contract. A run receives an
// immutable turn snapshot and returns the full transcript plus the messages
// produced by this invocation. The underlying AgentLoop remains available for
// older direct callers, but this facade prevents harness code from depending
// on loop instance reuse.

import type { LLMBackend } from "./backend.ts";
import { AgentLoop, type TurnMetrics } from "./loop.ts";
import type { AgentConfig, Message } from "./types.ts";

export interface StatelessAgentLoopOptions {
	config: AgentConfig;
	backend: LLMBackend;
	prompt: string;
	cwd?: string;
	maxIterations?: number;
	signal?: AbortSignal;
	initialMessages?: Message[];
	onLoopReady?: (loop: AgentLoop) => void;
}

export interface StatelessAgentLoopResult {
	messages: Message[];
	newMessages: Message[];
	loop: AgentLoop;
	turnMetrics: TurnMetrics;
}

function firstNewMessageIndex(initialMessages: Message[] | undefined, result: Message[]): number {
	const priorNonSystem = (initialMessages ?? []).filter((message) => message.role !== "system").length;
	return Math.min(result.length, 1 + priorNonSystem);
}

export async function runStatelessAgentLoop(
	options: StatelessAgentLoopOptions,
): Promise<StatelessAgentLoopResult> {
	const loop = new AgentLoop({
		config: options.config,
		backend: options.backend,
		cwd: options.cwd,
		maxIterations: options.maxIterations,
		signal: options.signal,
		initialMessages: options.initialMessages?.length
			? [...options.initialMessages]
			: undefined,
	});
	options.onLoopReady?.(loop);
	const messages = await loop.run(options.prompt);
	return {
		messages,
		newMessages: messages.slice(firstNewMessageIndex(options.initialMessages, messages)),
		loop,
		turnMetrics: loop.turnMetrics,
	};
}
