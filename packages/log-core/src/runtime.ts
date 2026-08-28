/** Contracts used by hosts that embed the core engine. */

export {
	createAssistantMessage,
	createToolResultMessage,
	createUserMessage,
	estimateChatPayloadTokens,
	estimateTokens,
} from "./capabilities/provider/messages.ts";
export {
	type CustomSessionEntry,
	type SessionEntry,
	SessionRegistry,
	SessionStore,
} from "./capabilities/session/session-store.ts";
export { ToolRegistry } from "./capabilities/tools/registry.ts";
export { resolveAgentSettings } from "./control/configuration/agent-settings.ts";
export {
	EventJournal,
	type EventJournalEntry,
	type EventJournalOptions,
	type EventJournalQuery,
	type EventJournalSubscriptionOptions,
	type JournalEvent,
} from "./runtime/events/event-journal.ts";
export type { AbortResult } from "./runtime/harness/types.ts";
export type { HarnessPhase } from "./runtime/state/runtime-state.ts";
export { loadExtensions } from "./system/extension/loader.ts";
export { ExtensionRunner } from "./system/extension/runner.ts";
export {
	type CancellationCleanup,
	CancellationError,
	type CancellationKind,
	CancellationScope,
	type CancellationScopeOptions,
} from "./system/lifecycle/cancellation-scope.ts";
