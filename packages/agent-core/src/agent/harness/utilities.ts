// ── Utilities for AgentHarness ───────────────────────────────────────────
// Extracted from harness.ts to reduce its line count.

import { createHash } from "node:crypto";
import type { AgentEvent } from "../types/index.ts";

/**
 * Deterministic serialization for SHA-256 hashing.
 * Arrays and objects are sorted by key to ensure stable hashes.
 */
export function stableSerialize(value: unknown): string {
	if (Array.isArray(value)) return `[${value.map(stableSerialize).join(",")}]`;
	if (value && typeof value === "object") {
		return `{${Object.entries(value as Record<string, unknown>)
			.sort(([left], [right]) => left.localeCompare(right))
			.map(([key, item]) => `${JSON.stringify(key)}:${stableSerialize(item)}`)
			.join(",")}}`;
	}
	return JSON.stringify(value) ?? "null";
}

/** SHA-256 hash of a value's canonical JSON representation. */
export function digest(value: unknown): string {
	return createHash("sha256").update(stableSerialize(value)).digest("hex");
}

/**
 * Convert an AgentEvent to a trajectory payload with digests.
 * Removes large fields (delta, content, message, etc.) and adds size/digest metadata.
 */
export function trajectoryEventPayload(event: AgentEvent): Record<string, unknown> {
	const payload = structuredClone(event as unknown as Record<string, unknown>);
	for (const field of ["delta", "content", "message", "messages", "result"]) {
		const value = payload[field];
		if (value !== undefined) {
			payload[`${field}Digest`] = digest(value);
			payload[`${field}Size`] =
				typeof value === "string" ? value.length : JSON.stringify(value).length;
			payload[field] = undefined;
		}
	}
	if (event.type === "subagent_event") {
		payload.eventType = event.event.type;
		payload.event = undefined;
	}
	return payload;
}

// Streaming events are presentation updates, not durable state transitions.
// Persisting each token/partial tool result makes the append-only projection grow
// quadratically (snapshot cloning) and, more importantly, fsyncs on the UI thread.
// Boundary events retain everything needed for replay, diagnostics, and evals.
export const EPHEMERAL_AGENT_EVENTS = new Set<AgentEvent["type"]>([
	"text_delta",
	"thinking_delta",
	"tool_call_delta",
	"tool_execution_update",
	"message_update",
	"context_update",
	"phase",
]);

/**
 * Check if an event is a durable state transition (not an ephemeral update).
 * Durable events are persisted to the trajectory; ephemeral ones are not.
 */
export function isDurableAgentEvent(event: AgentEvent): boolean {
	return !EPHEMERAL_AGENT_EVENTS.has(event.type);
}
