import type {
	InferenceMode,
	ThinkingLevel,
} from "../../system/types/types-config.ts";
import type { ExecutionProfile } from "../policy/execution-policy.ts";

const DEFAULT_MAX_ITERATIONS = 30;

export interface AgentSettings {
	executionProfile: ExecutionProfile;
	inferenceMode: InferenceMode;
	maxIterations: number;
	thinkingLevel: ThinkingLevel;
	toolExecution: "parallel" | "sequential";
}

export interface AgentSettingsInput {
	executionProfile?: ExecutionProfile | undefined;
	inferenceMode?: InferenceMode | undefined;
	maxIterations?: number | undefined;
	thinkingLevel?: ThinkingLevel | undefined;
	toolExecution?: "parallel" | "sequential" | undefined;
}

export function resolveAgentSettings(
	config: AgentSettingsInput,
): AgentSettings {
	return {
		executionProfile: config.executionProfile ?? "minimal",
		inferenceMode: config.inferenceMode ?? "none",
		maxIterations: config.maxIterations ?? DEFAULT_MAX_ITERATIONS,
		thinkingLevel: config.thinkingLevel ?? "off",
		toolExecution: config.toolExecution ?? "parallel",
	};
}
