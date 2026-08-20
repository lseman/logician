// ── Utils barrel ─────────────────────────────────────────────────────────

export { withQueueEventForwarding } from "../env/queue/hooks.ts";
export { type RunAgentLoopConfig, runAgentLoop } from "./agent-loop.ts";
export { createLLMBackend, type LLMBackend, OpenAIBackend } from "./backend.ts";
export { throwOnValidationErrors, validateConfig } from "./config-validator.ts";
export * from "./extension/index.ts";
export {
	beginFileFrame,
	clearFileFrames,
	currentFrameSize,
	recordBashMutations,
	recordFileBeforeWrite,
	restoreFileFrame,
	snapshotBeforeBash,
} from "./file-checkpoints.ts";
export * from "./guards/index.ts";
export {
	HookBus,
	type HookBusOptions,
	type HookEventName,
	type HookRegistration,
} from "./hook-bus.ts";
