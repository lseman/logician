import { createUserMessage } from "../agent/messages.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentHooks,
	Message,
} from "../agent/types.ts";
import type { MessageDeliveryManager } from "../queue/manager.ts";

export interface HarnessQueueHookDependencies {
	messageDelivery: MessageDeliveryManager;
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
		getSteeringMessages: async () =>
			inject(deps.messageDelivery.afterTurn().map(message => message.content)),
		getFollowUpMessages: async () =>
			inject(deps.messageDelivery.onIdle().map(message => message.content)),
	};
	const originalOnEvent = config.onEvent;
	return {
		...config,
		internalHooks,
		onEvent: event => {
			originalOnEvent?.(event);
			if (event.type === "turn_end") deps.onSavePoint?.();
			for (const subscriber of deps.subscribers) subscriber(event);
		},
	};
}
