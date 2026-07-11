import type { MessageDeliveryManager } from "../message-queue/manager.ts";
import { createUserMessage } from "../core/messages.ts";
import type { AgentConfig, AgentEvent, AgentHooks, Message } from "../core/types.ts";

export interface HarnessQueueHookDependencies {
	messageDelivery: MessageDeliveryManager;
	nextTurnQueue: string[];
	onQueueChange(): void;
	onSavePoint?(): void;
	subscribers: ReadonlySet<(event: AgentEvent) => void>;
}

export function withHarnessQueueHooks(
	config: AgentConfig,
	deps: HarnessQueueHookDependencies,
): AgentConfig {
	const inject = (texts: string[]): Message[] | undefined => {
		if (!texts.length) return undefined;
		deps.onQueueChange();
		return texts.map(createUserMessage);
	};
	const internalHooks: AgentHooks = {
		transformContext: ({ messages }) => {
			const pending = deps.nextTurnQueue.splice(0);
			if (!pending.length) return undefined;
			deps.onQueueChange();
			const last = Math.max(0, messages.length - 1);
			return { messages: [...messages.slice(0, last), ...pending.map(createUserMessage), ...messages.slice(last)] };
		},
		getSteeringMessages: async () => inject(deps.messageDelivery.afterTurn().map((message) => message.content)),
		getFollowUpMessages: async () => inject(deps.messageDelivery.onIdle().map((message) => message.content)),
	};
	const originalOnEvent = config.onEvent;
	return {
		...config,
		internalHooks,
		onEvent: (event) => {
			originalOnEvent?.(event);
			if (event.type === "turn_end") deps.onSavePoint?.();
			for (const subscriber of deps.subscribers) subscriber(event);
		},
	};
}
