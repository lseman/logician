import type { AgentEvent } from "../../system/types/types-messages.ts";

/** Events produced by the internal agent runtime. */
export type RuntimeEvent = AgentEvent;

/**
 * Stable lifecycle events exposed to extensions.
 *
 * Runtime-only details (for example token deltas and internal tool-call
 * bookkeeping) deliberately stay out of this protocol boundary.
 */
export const EXTENSION_LIFECYCLE_EVENT_TYPES = [
	"agent_start",
	"agent_end",
	"turn_start",
	"turn_end",
	"message_start",
	"message_update",
	"message_end",
	"tool_execution_start",
	"tool_execution_update",
	"tool_execution_end",
	"agent_retry_start",
	"agent_retry_end",
	"agent_error",
	"agent_settled",
	"session_delete",
	"model_select",
] as const;

export type ExtensionLifecycleEventType =
	(typeof EXTENSION_LIFECYCLE_EVENT_TYPES)[number];

export type ExtensionLifecycleEvent = Extract<
	AgentEvent,
	{ type: ExtensionLifecycleEventType }
>;

const extensionLifecycleEventTypes = new Set<string>(
	EXTENSION_LIFECYCLE_EVENT_TYPES,
);

export function isExtensionLifecycleEvent(
	event: RuntimeEvent,
): event is ExtensionLifecycleEvent {
	return extensionLifecycleEventTypes.has(event.type);
}
