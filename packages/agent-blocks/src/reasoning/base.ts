// ── Reasoner Base ──────────────────────────────────────────────────────────────
// Abstract base class for algorithmic multi-step reasoners.
// Adapted from Python src/reasoners/base.py.

import type { LLMBackend } from "@logician/agent-core";

export interface ReasoningTrace {
	reasoning: string;
	answer: string;
	metadata: Record<string, unknown>;
}

export interface ReasonerConfig {
	temperature?: number;
	maxTokens?: number;
	[key: string]: unknown;
}

export interface Reasoner {
	llm: LLMBackend;
	config: ReasonerConfig;
	solve(
		query: string,
		initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace>;
}

export type ReasonerConstructor = new (
	llm: LLMBackend,
	config: ReasonerConfig,
) => Reasoner;

export abstract class BaseReasoner implements Reasoner {
	llm: LLMBackend;
	config: ReasonerConfig;

	constructor(llm: LLMBackend, config: ReasonerConfig = {}) {
		this.llm = llm;
		this.config = config;
	}

	protected _chat(
		messages: Record<string, unknown>[],
		overrides: { temperature?: number; maxTokens?: number } = {},
	): Promise<string> {
		const temperature = overrides.temperature ?? this.config.temperature ?? 0.7;
		const maxTokens = overrides.maxTokens ?? this.config.maxTokens ?? 2048;
		return this.llm
			.generate(messages, { temperature, maxTokens })
			.then(resp => resp.content?.trim() ?? "");
	}

	protected static _extractAnswer(text: string): string {
		const idx = text.lastIndexOf("Final answer:");
		if (idx !== -1) {
			return text.slice(idx + "Final answer:".length).trim();
		}
		const lines = text
			.split("\n")
			.map(l => l.trim())
			.filter(Boolean);
		return lines[lines.length - 1] ?? text.trim();
	}

	protected _split(text: string): [string, string] {
		const idx = text.lastIndexOf("Final answer:");
		if (idx !== -1) {
			const reasoning = text.slice(0, idx).trim();
			const answer = text.slice(idx + "Final answer:".length).trim();
			return [reasoning, answer];
		}
		return [text, BaseReasoner._extractAnswer(text)];
	}

	abstract solve(
		query: string,
		initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace>;
}
