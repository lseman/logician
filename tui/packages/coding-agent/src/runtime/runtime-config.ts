import type { AgentBridgeOptions } from "./bridge.ts";
import {
	configBool,
	configNumber,
	configString,
	loadGlobalLogicianConfig,
	loadLogicianConfig,
	type LogicianTuiConfig,
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
	const project = options.loadProjectConfig === false
		? undefined
		: loadLogicianConfig(cwd);
	const loaded = !project || project.path === global.path
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
			model:
				environment.LOGICIAN_MODEL ||
				configString(config.model) ||
				"",
			models: config.models,
			systemPrompt:
				environment.LOGICIAN_SYSTEM_PROMPT ||
				configString(config.systemPrompt),
			chatTemplate: configString(config.chatTemplate),
			temperature: configNumber(config.temperature),
			maxTokens: configNumber(config.maxTokens),
			maxIterations: configNumber(config.maxIterations),
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
			continuationEnabled: configBool(config.continuationEnabled, true),
			postEditDiagnostics: configBool(config.postEditDiagnostics, true),
			autoRetryEnabled: configBool(config.autoRetryEnabled, true),
			maxRetries: configNumber(config.maxRetries),
			retryBaseDelayMs: configNumber(config.retryBaseDelayMs),
			turnTimeoutMs: configNumber(config.turnTimeoutMs),
			cacheSize: configNumber(config.cacheSize),
			cacheTtlMs: configNumber(config.cacheTtlMs),
			cwd: config.cwd ?? cwd,
			allowedPaths: config.allowedPaths,
			allowAllPaths: configBool(config.allowAllPaths),
			truncation: config.truncation,
			projectTrusted: options.loadProjectConfig !== false,
		},
	};
}
