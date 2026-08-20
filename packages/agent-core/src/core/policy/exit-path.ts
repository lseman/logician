// ── Budget Decision ────────────────────────────────────────────────────────

import type { RunBudgetController } from "./run-budget.ts";

export interface BudgetCheckResult {
	allowed: boolean;
	reason?: string;
	snapshot: {
		providerCalls: number;
		toolCalls: number;
		tokens: number;
		elapsedMs: number;
	};
}

export function checkBudget(
	budget: RunBudgetController,
	request: "provider_call" | "tool_batch" | "tokens",
	count: number = 1,
): BudgetCheckResult {
	switch (request) {
		case "provider_call":
			return budget.requestProviderCall();
		case "tool_batch":
			return budget.requestToolBatch(count);
		case "tokens":
			return budget.recordTokens(count);
	}
}
