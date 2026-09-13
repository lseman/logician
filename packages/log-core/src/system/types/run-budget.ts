export interface RunBudgetLimits {
	maxProviderCalls?: number | undefined;
	maxToolCalls?: number | undefined;
	maxTokens?: number | undefined;
	maxElapsedMs?: number | undefined;
}

export interface RunBudgetSnapshot {
	providerCalls: number;
	toolCalls: number;
	tokens: number;
	elapsedMs: number;
	remainingProviderCalls?: number | undefined;
	remainingToolCalls?: number | undefined;
	remainingTokens?: number | undefined;
}

export interface RunBudgetDecision {
	allowed: boolean;
	reason?: string;
	snapshot: RunBudgetSnapshot;
}

export interface RunBudgetInitialState {
	providerCalls?: number | undefined;
	toolCalls?: number | undefined;
	tokens?: number | undefined;
	startedAt?: number | undefined;
}

export type RunBudgetConsumption =
	| { resource: "provider_call"; amount: 1 }
	| { resource: "tool_call"; amount: number }
	| { resource: "token"; amount: number };
