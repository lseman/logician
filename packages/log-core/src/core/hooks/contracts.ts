import type { AgentHooks } from "../types/types-messages.ts";

export type RunControlHooks = Pick<
	AgentHooks,
	| "prepareNextTurn"
	| "shouldStopAfterTurn"
	| "getSteeringMessages"
	| "getFollowUpMessages"
>;

export type ExtensionHooks = Omit<AgentHooks, keyof RunControlHooks>;

export function extensionHooks(hooks: AgentHooks | undefined): ExtensionHooks {
	if (!hooks) return {};
	return {
		beforeAgentStart: hooks.beforeAgentStart,
		beforeToolCall: hooks.beforeToolCall,
		afterToolCall: hooks.afterToolCall,
		transformContext: hooks.transformContext,
		beforeProviderRequest: hooks.beforeProviderRequest,
		beforeProviderPayload: hooks.beforeProviderPayload,
		afterProviderResponse: hooks.afterProviderResponse,
		beforeCompact: hooks.beforeCompact,
	};
}

export function runControlHooks(
	hooks: AgentHooks | undefined,
): RunControlHooks {
	if (!hooks) return {};
	return {
		prepareNextTurn: hooks.prepareNextTurn,
		shouldStopAfterTurn: hooks.shouldStopAfterTurn,
		getSteeringMessages: hooks.getSteeringMessages,
		getFollowUpMessages: hooks.getFollowUpMessages,
	};
}
