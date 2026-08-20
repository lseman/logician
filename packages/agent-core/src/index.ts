/** Public contracts required to embed and extend the agent loop. */

export {
	type RunAgentLoopConfig,
	type RunAgentLoopContext,
	runAgentLoop,
} from "./core/execution/agent-loop-runner.ts";
export type {
	AcceptanceConfig,
	AcceptanceLedger,
} from "./core/guards/acceptance-contract.ts";
export { stripAcceptanceReport } from "./core/guards/acceptance-contract.ts";
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
