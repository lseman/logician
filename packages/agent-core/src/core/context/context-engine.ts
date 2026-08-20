import type { Message } from "../types/types-messages.ts";

export interface ContextContribution {
	source: string;
	messages?: readonly Message[];
	systemPrompt?: string;
	priority?: number;
}

export interface ContextSourceUsage {
	source: string;
	messages: number;
	estimatedTokens: number;
	included: boolean;
}

export interface ContextSnapshot {
	systemPrompt?: string;
	messages: Message[];
	sources: readonly ContextSourceUsage[];
}

export interface ContextAssemblyRequest {
	history: readonly Message[];
	baseSystemPrompt?: string;
	contributions?: readonly ContextContribution[];
	maxInjectedTokens?: number;
}

function fingerprint(message: Message): string {
	return [
		message.role,
		message.name ?? "",
		message.tool_call_id ?? "",
		message.content ?? "",
		JSON.stringify(message.tool_calls ?? []),
	].join("\u0000");
}

/** Curates model-visible context and reports exactly which sources were included. */
export class ContextEngine {
	constructor(
		private readonly estimateTokens: (messages: readonly Message[]) => number,
	) {}

	assemble(request: ContextAssemblyRequest): ContextSnapshot {
		const messages = request.history.map(message => ({ ...message }));
		const seen = new Set(messages.map(fingerprint));
		const sources: ContextSourceUsage[] = [];
		let injectedTokens = 0;
		let systemPrompt = request.baseSystemPrompt;
		let hasSystemPromptOverride = false;

		const contributions = [...(request.contributions ?? [])].sort(
			(left, right) => (right.priority ?? 0) - (left.priority ?? 0),
		);
		for (const contribution of contributions) {
			const unique = (contribution.messages ?? [])
				.filter(message => !seen.has(fingerprint(message)))
				.map(message => ({ ...message }));
			const estimatedTokens = this.estimateTokens(unique);
			const included =
				request.maxInjectedTokens === undefined ||
				injectedTokens + estimatedTokens <= request.maxInjectedTokens;
			sources.push({
				source: contribution.source,
				messages: unique.length,
				estimatedTokens,
				included,
			});
			if (!included) continue;
			for (const message of unique) seen.add(fingerprint(message));
			messages.push(...unique);
			injectedTokens += estimatedTokens;
			if (contribution.systemPrompt && !hasSystemPromptOverride) {
				systemPrompt = contribution.systemPrompt;
				hasSystemPromptOverride = true;
			}
		}

		return { systemPrompt, messages, sources };
	}
}
