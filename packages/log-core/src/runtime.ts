/** Contracts used by hosts that embed the core engine. */

export { resolveAgentSettings } from "./control/configuration/agent-settings.ts";
export { loadExtensions } from "./system/extension/loader.ts";
export { ExtensionRunner } from "./system/extension/runner.ts";
export type { AbortResult } from "./runtime/harness/types.ts";
export * from "./capabilities/provider/messages.ts";
export * from "./capabilities/session/session.ts";
export type { HarnessPhase } from "./runtime/state/runtime-state.ts";
export { ToolRegistry } from "./capabilities/tools/registry.ts";
