import type { AgentBridgeOptions } from "../bridge/agent-bridge.ts";
import {
	configBool,
	configNumber,
	configString,
	type LogicianTuiConfig,
} from "./config.ts";
import {
	buildConfigProvenance,
	type ConfigProvenance,
} from "./config-provenance.ts";
import {
	loadGlobalLogicianConfig,
	loadLogicianConfig,
} from "./config-store.ts";
import { applyEnvOverrides } from "./env-overrides.ts";

export interface ResolvedRuntimeConfig {
	configPath?: string;
	warnings: string[];
	source: LogicianTuiConfig;
	/** Precedence table: which layer (global/project/env) set each key. */
	provenance: ConfigProvenance;
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
		"reasonerConfig",
		"mcp",
		"mcpServers",
		"plugins",
		"legroom",
		"memoriam",
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
	// Resolution chain: defaults < global < project < env. Env values are
	// coerced against the settings schema; bad values warn and leave the
	// config value standing.
	const envApplied = applyEnvOverrides(
		loaded.config as unknown as Record<string, unknown>,
		environment,
	);
	const config = envApplied.config as unknown as LogicianTuiConfig;
	// Precedence table from the raw layer inputs (not the validated
	// config, which carries injected defaults).
	const provenance = buildConfigProvenance({
		global: global.raw,
		project: project?.raw,
		env: envApplied.applied,
	});

	return {
		configPath: loaded.path,
		warnings: [...loaded.warnings, ...envApplied.warnings],
		source: config,
		provenance,
		bridge: {
			configPath: loaded.path,
			baseUrl:
				configString(config.baseUrl) ||
				configString(config.llmUrl) ||
				"http://127.0.0.1:8080",
			model: configString(config.model) ?? "",
			legroom: config.legroom,
			memoriam: config.memoriam,
			models: config.models,
			systemPrompt: configString(config.systemPrompt),
			chatTemplate: configString(config.chatTemplate),
			temperature: configNumber(config.temperature),
			maxTokens: configNumber(config.maxTokens),
			maxIterations: configNumber(config.maxIterations),
			thinkingLevel: config.thinkingLevel,
			thinkingFormat: config.thinkingFormat,
			inferenceMode: config.inferenceMode,
			executionProfile: config.executionProfile,
			toolExecution:
				configString(config.toolExecution) === "sequential"
					? "sequential"
					: "parallel",
			contextWindowTokens:
				configNumber(config.contextWindowTokens) ||
				configNumber(config.contextWindow),
			runtimeHooksEnabled: configBool(config.hooks),
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
			verifiedStopEnabled: configBool(config.verifiedStopEnabled),
			proactiveCompactionEnabled: configBool(config.compaction?.enabled),
			compaction: config.compaction,
			maxParallelAgents: configNumber(config.maxParallelAgents),
			lsp: config.lsp,
			continuationEnabled: configBool(config.continuationEnabled, true),
			postEditDiagnostics: configBool(config.postEditDiagnostics, true),
			rtkProxyEnabled: configBool(config.rtkProxyEnabled),
			graphicianEnabled: configBool(config.graphicianEnabled, true),
			fffgrepEnabled: configBool(config.fffgrepEnabled, true),
			autoRetryEnabled: configBool(config.autoRetryEnabled, true),
			xdevEnabled: configBool(config.tools?.xdev, true),
			todoEnabled: configBool(config.todoEnabled, false),
			maxRetries: configNumber(config.maxRetries),
			retryBaseDelayMs: configNumber(config.retryBaseDelayMs),
			turnTimeoutMs: configNumber(config.turnTimeoutMs),
			cacheSize: configNumber(config.cacheSize),
			cacheTtlMs: configNumber(config.cacheTtlMs),
			reasoner: configString(config.reasoner) || "none",
			reasonerConfig: config.reasonerConfig,
			cwd: config.cwd ?? cwd,
			allowedPaths: config.allowedPaths,
			allowAllPaths: configBool(config.allowAllPaths),
			truncation: config.truncation,
			projectTrusted: options.loadProjectConfig !== false,
		},
	};
}
