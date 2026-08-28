import type {
	AgentConfig,
	AgentEvent,
	Tool,
	WebSearchConfig,
} from "@logician/log-core";
import { envNumber, eventLogPathFor } from "../environment.ts";
import type { AgentBridgeOptions } from "../types.ts";

export interface AgentConfigFactoryOptions {
	bridge: AgentBridgeOptions;
	cwd: string;
	sessionId: string;
	transcriptPath: string;
	systemPrompt: string;
	tools: Tool[];
	webSearch: WebSearchConfig;
	permissions: AgentConfig["permissions"];
	hooks: AgentConfig["hooks"];
	onPermissionRequest: NonNullable<AgentConfig["onPermissionRequest"]>;
	onQuestionRequest: NonNullable<AgentConfig["onQuestionRequest"]>;
	onTurnEnd: (turnId: string) => void;
	onEvent: (event: AgentEvent) => void;
}

/** Map bridge-facing options into the core runtime contract in one place. */
export function createAgentConfig(
	options: AgentConfigFactoryOptions,
): AgentConfig {
	const { bridge } = options;
	return {
		baseUrl: bridge.baseUrl,
		model: bridge.model,
		models: bridge.models,
		systemPrompt: options.systemPrompt,
		tools: options.tools,
		webSearch: options.webSearch,
		cwd: options.cwd,
		maxIterations: bridge.maxIterations || 30,
		executionProfile: bridge.executionProfile,
		temperature: bridge.temperature,
		maxTokens: bridge.maxTokens,
		thinkingLevel: bridge.thinkingLevel ?? "off",
		inferenceMode: bridge.inferenceMode ?? "none",
		toolExecution: bridge.toolExecution ?? "parallel",
		contextWindowTokens:
			envNumber("LOGICIAN_CONTEXT_WINDOW") ||
			envNumber("LOGICIAN_CTX_SIZE") ||
			bridge.contextWindowTokens,
		runtimeHooksEnabled:
			bridge.runtimeHooksEnabled ?? process.env.LOGICIAN_HOOKS !== "0",
		hookSessionId: options.sessionId,
		hookTranscriptPath: options.transcriptPath,
		eventLogPath: eventLogPathFor(options.transcriptPath),
		steeringInterrupt: bridge.steeringInterrupt,
		maxTotalTokens: bridge.maxTotalTokens,
		permissions: options.permissions,
		guardsEnabled: bridge.guardsEnabled,
		duplicateGuardEnabled: bridge.duplicateGuardEnabled,
		failureGuardEnabled: bridge.failureGuardEnabled,
		duplicateToolThreshold: bridge.duplicateToolThreshold,
		toolFailureLoopThreshold: bridge.toolFailureLoopThreshold,
		progressStopEnabled: bridge.progressStopEnabled,
		proactiveCompactionEnabled: bridge.proactiveCompactionEnabled,
		continuationEnabled: bridge.continuationEnabled,
		rtkProxyEnabled: bridge.rtkProxyEnabled,
		graphicianEnabled: bridge.graphicianEnabled ?? true,
		fffgrepEnabled: bridge.fffgrepEnabled ?? true,
		autoRetryEnabled: bridge.autoRetryEnabled,
		maxRetries: bridge.maxRetries,
		retryBaseDelayMs: bridge.retryBaseDelayMs,
		turnTimeoutMs: bridge.turnTimeoutMs,
		cacheSize: bridge.cacheSize,
		cacheTtlMs: bridge.cacheTtlMs,
		streamOptions: bridge.streamOptions,
		allowedPaths: bridge.allowedPaths,
		allowAllPaths: bridge.allowAllPaths,
		truncation: bridge.truncation,
		onPermissionRequest: options.onPermissionRequest,
		onQuestionRequest: options.onQuestionRequest,
		hooks: options.hooks,
		turnEndCallback: options.onTurnEnd,
		onEvent: options.onEvent,
	};
}
