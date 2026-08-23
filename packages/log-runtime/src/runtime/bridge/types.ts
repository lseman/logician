import type {
	AgentConfig,
	AgentModelConfig,
	Tool,
	TruncationConfig,
	WebSearchConfig,
} from "@logician/log-core";
import type {
	PermissionMode,
	PermissionRules,
} from "@logician/log-core/permissions";
import type { AgentProtocolNotification } from "@logician/log-protocol";
import type { ReasonerConfig } from "../../capabilities/reasoning/index.ts";

export type ProtocolCallback = (
	notification: AgentProtocolNotification,
) => void;
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
		| "progressStopEnabled"
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

// ── Per-capability config namespaces ────────────────────────────────────────
// One namespace per RuntimeContext slot (see bridge/capability-context.ts):
// each manager's constructor options nest under the matching key instead of
// living flush with AgentBridgeOptions' ~65 AgentConfig-forwarded fields.
// Fields with no manager behind them (temperature, guardsEnabled, ...) stay
// flat — namespacing those would rename every AgentConfig-forwarded field
// for no seam benefit, since there's nothing to swap behind them.

export interface LspCapabilityConfig {
	enabled?: boolean;
	timeoutMs?: number;
	serverOverrides?: Record<
		string,
		{ command: string; args?: string[]; languageId: string }
	>;
}

export interface MemoryCapabilityConfig {
	enabled?: boolean;
	dbPath?: string;
	extractorModel?: string;
	extractorBaseUrl?: string;
	captureTools?: boolean;
	injectContext?: boolean;
	contextBudget?: number;
	viewerEnabled?: boolean;
	viewerPort?: number;
	viewerHost?: string;
	embeddingsEnabled?: boolean;
	embeddingModel?: string;
}

export interface RepositoryMapCapabilityConfig {
	enabled?: boolean;
	maxTokens?: number;
}

export interface PermissionsCapabilityConfig {
	mode?: PermissionMode;
	rules?: PermissionRules;
}

export interface ExtensionsCapabilityConfig {
	dirs?: { user?: string; paths?: string[] };
}

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
	progressStopEnabled?: boolean;
	proactiveCompactionEnabled?: boolean;
	compaction?: {
		enabled?: boolean;
		reserveTokens?: number;
		keepRecentTokens?: number;
	};
	maxParallelAgents?: number;
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
	reasoner?: string;
	reasonerConfig?: ReasonerConfig;

	lsp?: LspCapabilityConfig;
	memory?: MemoryCapabilityConfig;
	repositoryMap?: RepositoryMapCapabilityConfig;
	permissions?: PermissionsCapabilityConfig;
	extensions?: ExtensionsCapabilityConfig;
}
