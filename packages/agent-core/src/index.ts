// Deliberately small package facade. Internal subsystems are available through
// named package subpaths; only the engine contract shared by workspace
// consumers belongs at the package root.

export {
	type RunAgentLoopConfig,
	type RunAgentLoopContext,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "./core/execution/agent-loop-runner.ts";
export type {
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "./core/provider/backend.ts";
export { stripTextToolCalls } from "./core/provider/text-tool-calls.ts";
export {
	type AgentConfig,
	DEFAULT_TRUNCATION,
	INFERENCE_MODE_ORDER,
	type InferenceMode,
} from "./core/types/types-config.ts";
export type {
	AgentEvent,
	Message,
	Tool,
	ToolContext,
	ToolResult,
} from "./core/types/types-messages.ts";
export type {
	AcceptanceConfig,
	AcceptanceLedger,
} from "./core/guards/acceptance-contract.ts";
export { stripAcceptanceReport } from "./core/guards/acceptance-contract.ts";
export { PermissionManager } from "./infrastructure/tools/permissions.ts";
export { parseFrontmatter } from "./infrastructure/tools/utils/frontmatter.ts";
export {
	highlight,
	highlightAuto,
} from "./infrastructure/tools/utils/syntax-highlighter.ts";
export { formatContextSize } from "./tui-utils.ts";
