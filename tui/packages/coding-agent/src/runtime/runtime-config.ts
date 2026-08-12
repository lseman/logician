import type { AgentBridgeOptions } from "../application/agent-bridge.ts";
import {
	configBool,
	configNumber,
	configString,
	type LogicianTuiConfig,
	loadGlobalLogicianConfig,
	loadLogicianConfig,
} from "../configuration/config.ts";

export interface ResolvedRuntimeConfig {
	configPath?: string;
	warnings: string[];
	source: LogicianTuiConfig;
	bridge: AgentBridgeOptions;
}

export function resolveRuntimeConfig(
	cwd: string,
	environment: NodeJS.ProcessEnv = process.env,
	options: { loadProjectConfig?: boolean } = {},
): ResolvedRuntimeConfig {
	const global = loadGlobalLogicianConfig(environment.HOME);
	const project =
		options.loadProjectConfig === false ? undefined : loadLogicianConfig(cwd);
	const loaded =
		!project || project.path === global.path
			? global
			: {
					path: project.path ?? global.path,
					config: { ...global.config, ...project.config },
					warnings: [...global.warnings, ...project.warnings],
				};
	const config = loaded.config;

	return {
		configPath: loaded.path,
		warnings: loaded.warnings,
		source: config,
		bridge: {
			baseUrl:
				environment.LOGICIAN_LLM_URL ||
				configString(config.baseUrl) ||
				configString(config.llmUrl) ||
				"http://127.0.0.1:8080",
			model: environment.LOGICIAN_MODEL || configString(config.model) || "",
			models: config.models,
			systemPrompt:
				environment.LOGICIAN_SYSTEM_PROMPT || configString(config.systemPrompt),
			chatTemplate: configString(config.chatTemplate),
			temperature: configNumber(config.temperature),
			maxTokens: configNumber(config.maxTokens),
			maxIterations: configNumber(config.maxIterations),
			thinkingLevel: config.thinkingLevel,
			inferenceMode: config.inferenceMode,
			executionProfile: config.executionProfile,
			toolExecution:
				configString(config.toolExecution) === "sequential"
					? "sequential"
					: "parallel",
			contextWindowTokens:
				configNumber(environment.LOGICIAN_CONTEXT_WINDOW) ||
				configNumber(environment.LOGICIAN_CTX_SIZE) ||
				configNumber(config.contextWindowTokens) ||
				configNumber(config.contextWindow),
			runtimeHooksEnabled:
				environment.LOGICIAN_HOOKS !== undefined
					? environment.LOGICIAN_HOOKS !== "0"
					: configBool(config.hooks),
			mcpEager:
				environment.LOGICIAN_MCP_EAGER !== undefined
					? environment.LOGICIAN_MCP_EAGER !== "0"
					: configBool(config.mcpEager),
			webSearch: config.webSearch
				? {
						baseUrl: configString(config.webSearch.baseUrl),
						maxResults: configNumber(config.webSearch.maxResults),
					}
				: undefined,
			permissionMode: config.permissionMode,
			permissionRules: config.permissions,
			steeringInterrupt: configBool(config.steeringInterrupt),
			maxTotalTokens: configNumber(config.maxTotalTokens),
			guardsEnabled: configBool(config.guardsEnabled),
			duplicateGuardEnabled: configBool(config.duplicateGuardEnabled, true),
			failureGuardEnabled: configBool(config.failureGuardEnabled),
			duplicateToolThreshold: configNumber(config.duplicateToolThreshold),
			toolFailureLoopThreshold: configNumber(config.toolFailureLoopThreshold),
			budgetStopEnabled: configBool(config.budgetStopEnabled),
			thinkingLoopDetectionEnabled: configBool(
				config.thinkingLoopDetectionEnabled,
				true,
			),
			proactiveCompactionEnabled: configBool(config.compaction?.enabled),
			continuationEnabled: configBool(config.continuationEnabled, true),
			reflectionConfig: config.reflectionConfig,
			postEditDiagnostics: configBool(config.postEditDiagnostics, true),
			rtkProxyEnabled: configBool(config.rtkProxyEnabled),
			ariadneEnabled: configBool(config.ariadneEnabled, true),
			fffgrepEnabled: configBool(config.fffgrepEnabled, true),
			autoRetryEnabled: configBool(config.autoRetryEnabled, true),
			maxRetries: configNumber(config.maxRetries),
			retryBaseDelayMs: configNumber(config.retryBaseDelayMs),
			turnTimeoutMs: configNumber(config.turnTimeoutMs),
			cacheSize: configNumber(config.cacheSize),
			cacheTtlMs: configNumber(config.cacheTtlMs),
			memoryEnabled: configBool(config.memory, false),
			memoryDbPath: configString(config.memoryDbPath),
			memoryExtractorModel:
				environment.LOGICIAN_MEMORY_EXTRACTOR_MODEL ||
				configString(config.memoryExtractor?.model) ||
				configString(config.memoryExtractorModel),
			memoryExtractorBaseUrl:
				environment.LOGICIAN_MEMORY_EXTRACTOR_URL ||
				configString(config.memoryExtractor?.baseUrl),
			memoryViewerEnabled: configBool(config.memoryViewer, true),
			memoryViewerPort: configNumber(config.memoryViewerPort),
			memoryEmbeddingsEnabled:
				environment.LOGICIAN_MEMORY_EMBEDDINGS !== undefined
					? configBool(environment.LOGICIAN_MEMORY_EMBEDDINGS, false)
					: configBool(config.memoryEmbeddings, false),
			memoryEmbeddingModel:
				environment.LOGICIAN_MEMORY_EMBEDDING_MODEL ||
				configString(config.memoryEmbeddingModel),
			reasoner:
				environment.LOGICIAN_REASONER ||
				configString(config.reasoner) ||
				"none",
			reasonerConfig: config.reasonerConfig,
			cwd: config.cwd ?? cwd,
			allowedPaths: config.allowedPaths,
			allowAllPaths: configBool(config.allowAllPaths),
			truncation: config.truncation,
			autoResumeSession: configBool(config.autoResumeSession, true),
			projectTrusted: options.loadProjectConfig !== false,
		},
	};
}
