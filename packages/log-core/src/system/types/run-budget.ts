export interface RunBudgetLimits {
	maxProviderCalls?: number;
	maxToolCalls?: number;
	maxTokens?: number;
	maxElapsedMs?: number;
}

export interface RunBudgetSnapshot {
	providerCalls: number;
	toolCalls: number;
	tokens: number;
	elapsedMs: number;
	remainingProviderCalls?: number;
	remainingToolCalls?: number;
	remainingTokens?: number;
}

export interface RunBudgetDecision {
	allowed: boolean;
	reason?: string;
	snapshot: RunBudgetSnapshot;
}

export interface RunBudgetInitialState {
	providerCalls?: number;
	toolCalls?: number;
	tokens?: number;
	startedAt?: number;
}

export type RunBudgetConsumption =
	| { resource: "provider_call"; amount: 1 }
	| { resource: "tool_call"; amount: number }
	| { resource: "token"; amount: number };
