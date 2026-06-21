// ── Extension system barrel export ────────────────────────────────────────────
// Public API for extensions: types, loader, state, runner, and typed events.

export * from "./types.ts";
export * from "./loader.ts";
export * from "./state.ts";
export * from "./runner.ts";

// Typed extension event system
export { ExtensionEventBus } from "../hooks/extension-event-bus.ts";
export { createExtensionContext } from "../hooks/extension-context.ts";
export type {
	ExtensionEvent,
	ExtensionEventName,
	ExtensionEventResult,
	ExtensionEventHandler,
	ExtensionErrorHandler,
} from "../hooks/extension-events.ts";
export type {
	BeforeAgentStartEvent,
	AgentEndEvent,
	TurnStartEvent,
	TurnEndEvent,
	MessageStartEvent,
	MessageUpdateEvent,
	MessageEndEvent,
	ToolExecutionStartEvent,
	ToolExecutionUpdateEvent,
	ToolExecutionEndEvent,
	ContextUpdateEvent,
	SessionBeforeSwitchEvent,
	SessionBeforeForkEvent,
	SessionBeforeCompactEvent,
	SessionCompactEvent,
	SessionShutdownEvent,
	BeforeProviderRequestEvent,
	AfterProviderResponseEvent,
} from "../hooks/extension-events.ts";
export type {
	ExtensionContext,
	ExtensionContextState,
	ExtensionContextActions,
} from "../hooks/extension-context.ts";
