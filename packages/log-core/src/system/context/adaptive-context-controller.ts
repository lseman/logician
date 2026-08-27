import type { Message } from "../types/types-messages.ts";
import type {
	ContextAssemblyRequest,
	ContextContribution,
	ContextSnapshot,
	ContextSourceUsage,
} from "./context-engine.ts";

export interface AdaptiveContextRequest extends ContextAssemblyRequest {
	/** Current user objective, used only for deterministic lexical relevance. */
	objective?: string;
}

export interface AdaptiveContextPlan extends ContextSnapshot {
	id: string;
	budget: {
		limit: number;
		used: number;
	};
}

export interface ContextOutcome {
	success: boolean;
	/** When known, credit only sources supported by independent evidence. */
	usefulSources?: readonly string[];
}

export interface AdaptiveContextControllerOptions {
	/** Weight of prior outcomes relative to declared priority and relevance. */
	learningWeight?: number;
	/** EWMA update rate. Higher values adapt faster to recent outcomes. */
	learningRate?: number;
}

interface SourceState {
	utility: number;
	uses: number;
}

interface RankedContribution {
	contribution: ContextContribution;
	index: number;
	score: number;
}

interface RecordedPlan {
	sources: string[];
	recorded: boolean;
}

const TOKEN_PATTERN = /[\p{L}\p{N}_./-]{3,}/gu;

function words(value: string): Set<string> {
	return new Set(
		(value.toLowerCase().match(TOKEN_PATTERN) ?? []).slice(0, 512),
	);
}

function messageText(messages: readonly Message[]): string {
	return messages.map(message => String(message.content ?? "")).join("\n");
}

function relevance(
	objective: Set<string>,
	contribution: ContextContribution,
): number {
	if (objective.size === 0) return 0;
	const candidate = words(
		`${contribution.source}\n${contribution.systemPrompt ?? ""}\n${messageText(contribution.messages ?? [])}`,
	);
	let matches = 0;
	for (const term of objective) if (candidate.has(term)) matches++;
	return matches / Math.sqrt(objective.size * Math.max(1, candidate.size));
}

/**
 * Plans request-scoped context under a token budget and learns source utility
 * from independently measured run outcomes. Learning is deliberately in-memory:
 * persistence belongs to a host adapter once there are two useful adapters.
 */
export class AdaptiveContextController {
	private readonly sourceState = new Map<string, SourceState>();
	private readonly plans = new Map<string, RecordedPlan>();
	private nextPlan = 1;
	private readonly learningWeight: number;
	private readonly learningRate: number;

	constructor(
		private readonly estimateTokens: (messages: readonly Message[]) => number,
		options: AdaptiveContextControllerOptions = {},
	) {
		this.learningWeight = options.learningWeight ?? 2;
		this.learningRate = options.learningRate ?? 0.25;
	}

	buildContext(request: AdaptiveContextRequest): AdaptiveContextPlan {
		const messages = request.history.map(message => ({ ...message }));
		const seen = new Set(messages.map(fingerprint));
		const objective = words(request.objective ?? "");
		const candidates = this.rank(request.contributions ?? [], objective);
		const budgetLimit = Math.max(
			0,
			request.maxInjectedTokens ?? Number.MAX_SAFE_INTEGER,
		);
		const sources: ContextSourceUsage[] = [];
		const includedSources: string[] = [];
		let used = 0;
		let systemPrompt = request.baseSystemPrompt;
		let hasSystemPromptOverride = false;

		for (const candidate of candidates) {
			const contribution = candidate.contribution;
			const unique = (contribution.messages ?? [])
				.filter(message => !seen.has(fingerprint(message)))
				.map(message => ({ ...message }));
			const selected: Message[] = [];
			for (const message of unique) {
				const tokens = this.estimateTokens([message]);
				if (used + tokens > budgetLimit) continue;
				selected.push(message);
				used += tokens;
			}
			const included = selected.length > 0 || unique.length === 0;
			sources.push({
				source: contribution.source,
				messages: selected.length,
				estimatedTokens: this.estimateTokens(selected),
				included,
			});
			if (!included) continue;
			includedSources.push(contribution.source);
			for (const message of selected) seen.add(fingerprint(message));
			messages.push(...selected);
			if (contribution.systemPrompt && !hasSystemPromptOverride) {
				systemPrompt = contribution.systemPrompt;
				hasSystemPromptOverride = true;
			}
		}

		const id = `context-${this.nextPlan++}`;
		this.plans.set(id, { sources: includedSources, recorded: false });
		return {
			id,
			systemPrompt,
			messages,
			sources,
			budget: { limit: budgetLimit, used },
		};
	}

	recordOutcome(planId: string, outcome: ContextOutcome): boolean {
		const plan = this.plans.get(planId);
		if (!plan || plan.recorded) return false;
		plan.recorded = true;
		const useful = outcome.usefulSources
			? new Set(outcome.usefulSources)
			: null;
		for (const source of plan.sources) {
			const state = this.sourceState.get(source) ?? { utility: 0.5, uses: 0 };
			const reward = outcome.success && (!useful || useful.has(source)) ? 1 : 0;
			state.utility += this.learningRate * (reward - state.utility);
			state.uses++;
			this.sourceState.set(source, state);
		}
		this.prunePlans();
		return true;
	}

	private rank(
		contributions: readonly ContextContribution[],
		objective: Set<string>,
	): RankedContribution[] {
		return contributions
			.map((contribution, index) => {
				const state = this.sourceState.get(contribution.source);
				const learned = state?.utility ?? 0.5;
				const exploration = 1 / Math.sqrt(1 + (state?.uses ?? 0));
				return {
					contribution,
					index,
					score:
						(contribution.priority ?? 0) * 4 +
						relevance(objective, contribution) * 3 +
						learned * this.learningWeight +
						exploration * 0.25,
				};
			})
			.sort(
				(left, right) => right.score - left.score || left.index - right.index,
			);
	}

	private prunePlans(): void {
		if (this.plans.size <= 128) return;
		for (const [id, plan] of this.plans) {
			if (plan.recorded) this.plans.delete(id);
			if (this.plans.size <= 64) break;
		}
	}
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
