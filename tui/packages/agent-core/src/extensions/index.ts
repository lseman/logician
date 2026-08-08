// ── Extension system barrel export ────────────────────────────────────────────
// Public API for extensions: types, loader, state, runner, and typed events.

export type {
	ExtensionContext,
	ExtensionContextActions,
	ExtensionContextState,
} from "../hooks/extensions/context.ts";
export { createExtensionContext } from "../hooks/extensions/context.ts";
// Typed extension event system
export { ExtensionEventBus } from "../hooks/extensions/event-bus.ts";
export type {
	AfterProviderResponseEvent,
	AgentEndEvent,
	BeforeAgentStartEvent,
	BeforeProviderRequestEvent,
	ContextEvent,
	ContextResult,
	ContextUpdateEvent,
	ExtensionErrorHandler,
	ExtensionEvent,
	ExtensionEventHandler,
	ExtensionEventName,
	ExtensionEventResult,
	MessageEndEvent,
	MessageStartEvent,
	MessageUpdateEvent,
	SessionBeforeCompactEvent,
	SessionBeforeForkEvent,
	SessionBeforeSwitchEvent,
	SessionCompactEvent,
	SessionShutdownEvent,
	ToolExecutionEndEvent,
	ToolExecutionStartEvent,
	ToolExecutionUpdateEvent,
	TurnEndEvent,
	TurnStartEvent,
} from "../hooks/extensions/events.ts";
export * from "./loader.ts";
export * from "./runner.ts";
export * from "./state.ts";
export * from "./types.ts";
// Pi adapter and type guard helpers (for Pi extensions running on Logician)
export { PiAdapter, isToolCallEventType, isBashToolResult } from "./pi-adapter.ts";
export type { PiAdapterOptions, PiExtensionAPI } from "./pi-adapter.ts";
