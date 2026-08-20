import type { ReasonerConfig } from "@logician/agent-blocks/reasoning";
import type {
	AgentConfig,
	AgentModelConfig,
	Tool,
	TruncationConfig,
} from "../../core/types/index.ts";
import type { RuntimeEvent } from "../../core/types/runtime-events.ts";
import type { WebSearchConfig } from "../../core/types/types-config.ts";
import type {
	PermissionMode,
	PermissionRules,
} from "../../infrastructure/tools/permissions.ts";

export type EventCallback = (event: RuntimeEvent) => void;
export type ErrorCallback = (error: Error) => void;

export type RuntimeSettingsPatch = Partial<
	Pick<
		AgentConfig,
		| "thinkingLevel"
		| "temperature"
		| "inferenceMode"
		| "maxTokens"
		| "maxIterations"
		| "executionProfile"
		| "guardsEnabled"
		| "duplicateGuardEnabled"
		| "failureGuardEnabled"
		| "budgetStopEnabled"
		| "continuationEnabled"
		| "autoRetryEnabled"
		| "proactiveCompactionEnabled"
		| "rtkProxyEnabled"
		| "ariadneEnabled"
		| "fffgrepEnabled"
	>
> & {
	reasonerId?: string;
	steeringInterrupt?: boolean;
	postEditDiagnostics?: boolean;
	memoryEnabled?: boolean;
	guardMode?: "auto" | "on" | "off";
};

export interface AgentBridgeOptions {
	configPath?: string;
	baseUrl: string;
	model: string;
	models?: AgentModelConfig[];
	chatTemplate?: string;
	temperature?: number;
	maxTokens?: number;
	maxIterations?: number;
	thinkingLevel?: AgentConfig["thinkingLevel"];
	inferenceMode?: AgentConfig["inferenceMode"];
	executionProfile?: AgentConfig["executionProfile"];
	contextWindowTokens?: number;
	toolExecution?: AgentConfig["toolExecution"];
	runtimeHooksEnabled?: boolean;
	permissionMode?: PermissionMode;
	permissionRules?: PermissionRules;
	steeringInterrupt?: boolean;
	maxTotalTokens?: number;
	autoStartMcp?: boolean;
	tools?: Tool[];
	extraTools?: Tool[];
	cwd?: string;
	systemPrompt?: string;
	webSearch?: Partial<WebSearchConfig>;
	guardsEnabled?: boolean;
	duplicateGuardEnabled?: boolean;
	failureGuardEnabled?: boolean;
	duplicateToolThreshold?: number;
	toolFailureLoopThreshold?: number;
	budgetStopEnabled?: boolean;
	proactiveCompactionEnabled?: boolean;
	compaction?: {
		enabled?: boolean;
		reserveTokens?: number;
		keepRecentTokens?: number;
	};
	maxParallelAgents?: number;
	lsp?: {
		enabled?: boolean;
		timeoutMs?: number;
		serverOverrides?: Record<
			string,
			{ command: string; args?: string[]; languageId: string }
		>;
	};
	continuationEnabled?: boolean;
	postEditDiagnostics?: boolean;
	rtkProxyEnabled?: boolean;
	ariadneEnabled?: boolean;
	fffgrepEnabled?: boolean;
	autoRetryEnabled?: boolean;
	maxRetries?: number;
	retryBaseDelayMs?: number;
	turnTimeoutMs?: number;
	cacheSize?: number;
	cacheTtlMs?: number;
	streamOptions?: AgentConfig["streamOptions"];
	allowedPaths?: string[];
	allowAllPaths?: boolean;
	truncation?: TruncationConfig;
	projectTrusted?: boolean;
	autoResumeSession?: boolean;
	extensionDirs?: { user?: string; paths?: string[] };
	memoryEnabled?: boolean;
	memoryDbPath?: string;
	memoryExtractorModel?: string;
	memoryExtractorBaseUrl?: string;
	memoryCaptureTools?: boolean;
	memoryInjectContext?: boolean;
	memoryContextBudget?: number;
	memoryViewerEnabled?: boolean;
	memoryViewerPort?: number;
	memoryViewerHost?: string;
	memoryEmbeddingsEnabled?: boolean;
	memoryEmbeddingModel?: string;
	reasoner?: string;
	reasonerConfig?: ReasonerConfig;
	repositoryMapEnabled?: boolean;
	repositoryMapMaxTokens?: number;
}
