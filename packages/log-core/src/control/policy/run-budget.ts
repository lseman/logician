import type {
	RunBudgetConsumption,
	RunBudgetDecision,
	RunBudgetInitialState,
	RunBudgetLimits,
	RunBudgetSnapshot,
} from "../../system/types/run-budget.ts";

/** Hierarchical run accounting behind one small decision interface. */
export class RunBudgetController {
	private providerCalls = 0;
	private toolCalls = 0;
	private tokens = 0;
	private readonly startedAt: number;

	constructor(
		private readonly limits: RunBudgetLimits = {},
		private readonly now: () => number = Date.now,
		initial: RunBudgetInitialState = {},
		private readonly onConsumed?: (consumption: RunBudgetConsumption) => void,
	) {
		this.providerCalls = initial.providerCalls ?? 0;
		this.toolCalls = initial.toolCalls ?? 0;
		this.tokens = initial.tokens ?? 0;
		this.startedAt = initial.startedAt ?? now();
	}

	requestProviderCall(): RunBudgetDecision {
		const elapsed = this.checkElapsed();
		if (elapsed) return elapsed;
		const max = this.limits.maxProviderCalls;
		if (max !== undefined && this.providerCalls >= max) {
			return this.denied("provider-call budget exhausted");
		}
		this.providerCalls++;
		this.onConsumed?.({ resource: "provider_call", amount: 1 });
		return { allowed: true, snapshot: this.snapshot() };
	}

	requestToolBatch(count: number): RunBudgetDecision {
		const elapsed = this.checkElapsed();
		if (elapsed) return elapsed;
		const max = this.limits.maxToolCalls;
		if (max !== undefined && this.toolCalls + count > max) {
			return this.denied("tool-call budget exhausted");
		}
		this.toolCalls += count;
		if (count > 0) this.onConsumed?.({ resource: "tool_call", amount: count });
		return { allowed: true, snapshot: this.snapshot() };
	}

	recordTokens(count: number): RunBudgetDecision {
		const amount = Math.max(0, Math.floor(count));
		if (amount > 0) {
			this.tokens += amount;
			this.onConsumed?.({ resource: "token", amount });
		}
		const max = this.limits.maxTokens;
		if (max !== undefined && this.tokens > max)
			return this.denied("token budget exhausted");
		return { allowed: true, snapshot: this.snapshot() };
	}

	snapshot(): RunBudgetSnapshot {
		return {
			providerCalls: this.providerCalls,
			toolCalls: this.toolCalls,
			tokens: this.tokens,
			elapsedMs: Math.max(0, this.now() - this.startedAt),
			remainingProviderCalls:
				this.limits.maxProviderCalls === undefined
					? undefined
					: Math.max(0, this.limits.maxProviderCalls - this.providerCalls),
			remainingToolCalls:
				this.limits.maxToolCalls === undefined
					? undefined
					: Math.max(0, this.limits.maxToolCalls - this.toolCalls),
			remainingTokens:
				this.limits.maxTokens === undefined
					? undefined
					: Math.max(0, this.limits.maxTokens - this.tokens),
		};
	}

	private checkElapsed(): RunBudgetDecision | undefined {
		if (
			this.limits.maxElapsedMs !== undefined &&
			this.now() - this.startedAt >= this.limits.maxElapsedMs
		) {
			return this.denied("elapsed-time budget exhausted");
		}
		return undefined;
	}

	private denied(reason: string): RunBudgetDecision {
		return { allowed: false, reason, snapshot: this.snapshot() };
	}
}
