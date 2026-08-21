/** Public contracts required to embed and extend the agent loop. */

export {
	type RunAgentLoopConfig,
	type RunAgentLoopContext,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "./core/execution/agent-loop-runner.ts";
export type {
	AcceptanceConfig,
	AcceptanceLedger,
} from "./core/guards/acceptance-contract.ts";
export { stripAcceptanceReport } from "./core/guards/acceptance-contract.ts";
export type {
	BackendErrorCategory,
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "./core/provider/backend.ts";
export {
	BackendError,
	classifyHttpError,
	classifyNetworkError,
	createLLMBackend,
	normalizeProviderMessages,
	OpenAIBackend,
	parseProviderUsage,
} from "./core/provider/backend.ts";
export {
	parseTextToolCalls,
	stripTextToolCalls,
} from "./core/provider/text-tool-calls.ts";
export {
	type AgentConfig,
	type AgentModelConfig,
	cycleInferenceMode,
	DEFAULT_INFERENCE_MODE,
	DEFAULT_TRUNCATION,
	getInferenceMode,
	INFERENCE_MODE_ORDER,
	INFERENCE_MODES,
	type InferenceMode,
	isValidInferenceMode,
	type QueueMode,
	THINKING_LEVELS,
	type ThinkingLevel,
	type TruncationConfig,
	type WebSearchConfig,
} from "./core/types/types-config.ts";
export type {
	AgentEvent,
	AgentHooks,
	AskUserContext,
	CompactableMessage,
	Message,
	Tool,
	ToolCall,
	ToolContext,
	ToolResult,
} from "./core/types/types-messages.ts";
