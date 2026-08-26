import type { RuntimeEvent } from "./types-events.ts";

export const AGENT_PROTOCOL_VERSION = 1 as const;

/** Stable identifiers that correlate notifications across replay and clients. */
export interface AgentProtocolCorrelation {
	sessionId?: string;
	runId?: string;
	turnId?: string;
	toolCallId?: string;
}

export interface AgentProtocolNotification {
	protocolVersion: typeof AGENT_PROTOCOL_VERSION;
	sequence: number;
	timestamp: number;
	event: RuntimeEvent;
	correlation?: AgentProtocolCorrelation;
}

export function createNotification(
	event: RuntimeEvent,
	sequence: number,
	timestamp: number = Date.now(),
	correlation?: AgentProtocolCorrelation,
): AgentProtocolNotification {
	return {
		protocolVersion: AGENT_PROTOCOL_VERSION,
		sequence,
		timestamp,
		event,
		...(correlation && { correlation }),
	};
}

export function isAgentProtocolNotification(
	value: unknown,
): value is AgentProtocolNotification {
	if (!value || typeof value !== "object") return false;
	const candidate = value as Partial<AgentProtocolNotification>;
	return (
		candidate.protocolVersion === AGENT_PROTOCOL_VERSION &&
		Number.isSafeInteger(candidate.sequence) &&
		typeof candidate.timestamp === "number" &&
		(candidate.correlation === undefined ||
			(candidate.correlation !== null &&
				typeof candidate.correlation === "object")) &&
		Boolean(
			candidate.event &&
				typeof candidate.event === "object" &&
				typeof (candidate.event as { type?: unknown }).type === "string",
		)
	);
}
