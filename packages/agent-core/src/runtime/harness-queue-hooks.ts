import type { AgentConfig, AgentEvent } from "../agent/types/index.ts";

export interface HarnessQueueEventDependencies {
	onSavePoint?(): void;
	subscribers: ReadonlySet<(event: AgentEvent) => void>;
}

/** Forward every loop event to harness subscribers and fire the save-point hook at turn end. */
export function withQueueEventForwarding(
	config: AgentConfig,
	deps: HarnessQueueEventDependencies,
): AgentConfig {
	const originalOnEvent = config.onEvent;
	return {
		...config,
		onEvent: event => {
			originalOnEvent?.(event);
			if (event.type === "turn_end") deps.onSavePoint?.();
			for (const subscriber of deps.subscribers) subscriber(event);
		},
	};
}
