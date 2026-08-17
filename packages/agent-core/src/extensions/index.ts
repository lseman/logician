// ── Extension system barrel export ────────────────────────────────────────────
// Public API for extensions: types, loader, state, runner.

export type {
	ExtensionContext,
	ExtensionContextActions,
	ExtensionContextState,
} from "../hooks/extensions/context.ts";
export { createExtensionContext } from "../hooks/extensions/context.ts";
export * from "./loader.ts";
export type { PiAdapterOptions, PiExtensionAPI } from "./pi-adapter.ts";
// Pi adapter and type guard helpers (for Pi extensions running on Logician)
export {
	isBashToolResult,
	isToolCallEventType,
	PiAdapter,
} from "./pi-adapter.ts";
export * from "./runner.ts";
export * from "./state.ts";
export * from "./types.ts";
