export interface RunBudgetLimits {
	maxProviderCalls?: number;
	maxToolCalls?: number;
	maxElapsedMs?: number;
	reserveFinalizationCalls?: number;
}

export interface RunBudgetSnapshot {
	providerCalls: number;
	toolCalls: number;
	elapsedMs: number;
	remainingProviderCalls?: number;
	remainingToolCalls?: number;
}

export interface RunBudgetDecision {
	allowed: boolean;
	reason?: string;
	snapshot: RunBudgetSnapshot;
}

/** Hierarchical run accounting behind one small decision interface. */
export class RunBudgetController {
	private providerCalls = 0;
	private toolCalls = 0;
	private readonly startedAt: number;

	constructor(
		private readonly limits: RunBudgetLimits = {},
		private readonly now: () => number = Date.now,
	) {
		this.startedAt = now();
	}

	requestProviderCall(finalization = false): RunBudgetDecision {
		const elapsed = this.checkElapsed();
		if (elapsed) return elapsed;
		const max = this.limits.maxProviderCalls;
		const reserve = finalization
			? 0
			: (this.limits.reserveFinalizationCalls ?? 0);
		if (max !== undefined && this.providerCalls >= Math.max(0, max - reserve)) {
			return this.denied("provider-call budget exhausted");
		}
		this.providerCalls++;
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
		return { allowed: true, snapshot: this.snapshot() };
	}

	snapshot(): RunBudgetSnapshot {
		return {
			providerCalls: this.providerCalls,
			toolCalls: this.toolCalls,
			elapsedMs: Math.max(0, this.now() - this.startedAt),
			remainingProviderCalls:
				this.limits.maxProviderCalls === undefined
					? undefined
					: Math.max(0, this.limits.maxProviderCalls - this.providerCalls),
			remainingToolCalls:
				this.limits.maxToolCalls === undefined
					? undefined
					: Math.max(0, this.limits.maxToolCalls - this.toolCalls),
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
