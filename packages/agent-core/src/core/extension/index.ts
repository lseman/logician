// ── Extension block ─────────────────────────────────────────────────────────
// TypeScript extension system: loader, runner, types, state, event-bus.

export type {
	ExtensionContext,
	ExtensionContextActions,
	ExtensionContextState,
} from "./context.ts";
export { createExtensionContext } from "./context.ts";
export * from "./loader.ts";
export type { PiAdapterOptions, PiExtensionAPI } from "./adapters/pi/index.ts";
export {
	isBashToolResult,
	isToolCallEventType,
	PiAdapter,
} from "./adapters/pi/index.ts";
export * from "./runner.ts";
export * from "./state.ts";
export * from "./types.ts";
export * from "./event-bus.ts";
export type { PluginCommandResult } from "./adapters/claude-code/plugin-manager.ts";
