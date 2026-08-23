import type { AgentBridgeOptions } from "../bridge/agent-bridge.ts";
import {
	configBool,
	configNumber,
	configString,
	type LogicianTuiConfig,
} from "./config.ts";
import {
	loadGlobalLogicianConfig,
	loadLogicianConfig,
} from "./config-store.ts";

export interface ResolvedRuntimeConfig {
	configPath?: string;
	warnings: string[];
	source: LogicianTuiConfig;
	bridge: AgentBridgeOptions;
}

function mergeObject<T extends Record<string, unknown>>(
	base: T | undefined,
	override: T | undefined,
): T | undefined {
	if (!base) return override;
	if (!override) return base;
	return { ...base, ...override };
}

/** Merge validated config layers without dropping sibling settings in sections. */
export function mergeRuntimeConfigLayers(
	global: LogicianTuiConfig,
	project: LogicianTuiConfig,
): LogicianTuiConfig {
	const merged: LogicianTuiConfig = { ...global, ...project };
	for (const key of [
		"webSearch",
		"permissions",
		"compaction",
		"memoryExtractor",
		"reasonerConfig",
		"mcp",
		"mcpServers",
		"plugins",
	] as const) {
		const value = mergeObject(
			global[key] as Record<string, unknown> | undefined,
			project[key] as Record<string, unknown> | undefined,
		);
		if (value) (merged as Record<string, unknown>)[key] = value;
	}
	if (global.lsp || project.lsp) {
		const serverOverrides = mergeObject(
			global.lsp?.serverOverrides,
			project.lsp?.serverOverrides,
		);
		merged.lsp = {
			...global.lsp,
			...project.lsp,
			...(serverOverrides ? { serverOverrides } : {}),
		};
	}
	if (global.truncation || project.truncation) {
		const microCompactMaxChars = mergeObject(
			global.truncation?.microCompactMaxChars,
			project.truncation?.microCompactMaxChars,
		);
		merged.truncation = {
			...global.truncation,
			...project.truncation,
			...(microCompactMaxChars ? { microCompactMaxChars } : {}),
		};
	}
	return merged;
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
					config: mergeRuntimeConfigLayers(global.config, project.config),
					warnings: [...global.warnings, ...project.warnings],
				};
	const config = loaded.config;

	return {
		configPath: loaded.path,
		warnings: loaded.warnings,
		source: config,
		bridge: {
			configPath: loaded.path,
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
			webSearch: config.webSearch
				? {
						baseUrl: configString(config.webSearch.baseUrl),
						maxResults: configNumber(config.webSearch.maxResults),
					}
				: undefined,
			permissions: {
				mode: config.permissionMode,
				rules: config.permissions,
			},
			steeringInterrupt: configBool(config.steeringInterrupt),
			maxTotalTokens: configNumber(config.maxTotalTokens),
			guardsEnabled: configBool(config.guardsEnabled),
			duplicateGuardEnabled: configBool(config.duplicateGuardEnabled, true),
			failureGuardEnabled: configBool(config.failureGuardEnabled),
			duplicateToolThreshold: configNumber(config.duplicateToolThreshold),
			toolFailureLoopThreshold: configNumber(config.toolFailureLoopThreshold),
			progressStopEnabled: configBool(config.progressStopEnabled),
			proactiveCompactionEnabled: configBool(config.compaction?.enabled),
			compaction: config.compaction,
			maxParallelAgents: configNumber(config.maxParallelAgents),
			lsp: config.lsp,
			continuationEnabled: configBool(config.continuationEnabled, true),
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
			memory: {
				enabled: configBool(config.memory, false),
				dbPath: configString(config.memoryDbPath),
				extractorModel:
					environment.LOGICIAN_MEMORY_EXTRACTOR_MODEL ||
					configString(config.memoryExtractor?.model) ||
					configString(config.memoryExtractorModel),
				extractorBaseUrl:
					environment.LOGICIAN_MEMORY_EXTRACTOR_URL ||
					configString(config.memoryExtractor?.baseUrl),
				viewerEnabled: configBool(config.memoryViewer, true),
				viewerPort: configNumber(config.memoryViewerPort),
				embeddingsEnabled:
					environment.LOGICIAN_MEMORY_EMBEDDINGS !== undefined
						? configBool(environment.LOGICIAN_MEMORY_EMBEDDINGS, false)
						: configBool(config.memoryEmbeddings, false),
				embeddingModel:
					environment.LOGICIAN_MEMORY_EMBEDDING_MODEL ||
					configString(config.memoryEmbeddingModel),
			},
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
