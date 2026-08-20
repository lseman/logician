import type { AgentConfig, AgentEvent } from "../../types/index.ts";

export interface HarnessQueueEventDependencies {
	subscribers: ReadonlySet<(event: AgentEvent) => void>;
}

/** Forward every loop event to harness subscribers. */
export function withQueueEventForwarding(
	config: AgentConfig,
	deps: HarnessQueueEventDependencies,
): AgentConfig {
	const originalOnEvent = config.onEvent;
	return {
		...config,
		onEvent: event => {
			originalOnEvent?.(event);
			for (const subscriber of deps.subscribers) subscriber(event);
		},
	};
}
