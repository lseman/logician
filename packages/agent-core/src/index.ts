// Deliberately small package facade. Internal subsystems are available through
// named package subpaths; only the engine contract shared by workspace
// consumers belongs at the package root.

export {
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
	type RunAgentLoopConfig,
	type RunAgentLoopContext,
} from "./core/execution/agent-loop-runner.ts";
export type {
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "./core/provider/backend.ts";
export {
	DEFAULT_TRUNCATION,
	INFERENCE_MODE_ORDER,
	type AgentConfig,
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
} from "./infrastructure/guards/acceptance-contract.ts";
export { stripAcceptanceReport } from "./infrastructure/guards/acceptance-contract.ts";
export { PermissionManager } from "./infrastructure/tools/permissions.ts";
export { parseFrontmatter } from "./infrastructure/tools/utils/frontmatter.ts";
export {
	highlight,
	highlightAuto,
} from "./infrastructure/tools/utils/syntax-highlighter.ts";
export { stripTextToolCalls } from "./infrastructure/tools/utils/text-to-tool-calls.ts";
export type { Task, TaskStatus } from "./runtime/todo-state.ts";
export {
	allocateTaskId,
	getTasks,
	getTasksMutable,
	notifyTodosChanged,
	onTodosChanged,
	replaceTasks,
} from "./runtime/todo-state.ts";
export { formatContextSize } from "./tui-utils.ts";
