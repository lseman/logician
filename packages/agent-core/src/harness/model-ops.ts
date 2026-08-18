// ── Model / thinking-level operations for AgentHarness ────────────────────
// Pure(ish) helpers operating on explicit state passed in by the harness.
// The harness owns the mutable fields (config, backend, session) and
// supplies them here via ModelOpsDeps, mirroring queue-ops.ts's pattern.

import type { LLMBackend } from "../core/backend.ts";
import type {
	AgentEvent,
	AgentModelConfig,
	ThinkingLevel,
} from "../types/index.ts";
import { cycleModel as cycleModelPure, resolveModelUrl } from "./model.ts";

export interface ModelOpsConfig {
	model: string;
	baseUrl: string;
	models?: AgentModelConfig[];
	thinkingLevel?: ThinkingLevel;
}

export interface ModelOpsDeps {
	getConfig: () => ModelOpsConfig;
	setModel: (model: string) => void;
	setBaseUrl: (baseUrl: string) => void;
	setModels: (models: AgentModelConfig[]) => void;
	setThinkingLevel: (level: ThinkingLevel) => void;
	getBackend: () => LLMBackend;
	setBackend: (backend: LLMBackend) => void;
	appendModelChange: (model: string) => void;
	appendThinkingLevelChange: (level: ThinkingLevel) => void;
	emit: (event: AgentEvent) => void;
}

export function getModel(deps: ModelOpsDeps): string {
	return deps.getConfig().model;
}

export function getBaseUrl(deps: ModelOpsDeps): string {
	return deps.getConfig().baseUrl;
}

export function getModels(deps: ModelOpsDeps): string[] {
	const config = deps.getConfig();
	const configured = config.models ?? [];
	return [
		...(configured.some(option => option.model === config.model)
			? []
			: [config.model]),
		...configured.map(option => option.model),
	];
}

export function setModelEndpoint(
	deps: ModelOpsDeps,
	model: string,
	baseUrl: string,
): void {
	deps.setModel(model);
	deps.setBaseUrl(baseUrl);
	const backend = deps.getBackend();
	deps.setBackend(
		backend.withEndpoint?.(model, baseUrl) ?? backend.withModel(model),
	);
}

/** Resolve the baseUrl for a given model identifier. */
export function getModelUrl(deps: ModelOpsDeps, modelName: string): string {
	const config = deps.getConfig();
	return resolveModelUrl(config.models, modelName, config.baseUrl);
}

export function setModels(
	deps: ModelOpsDeps,
	models: AgentModelConfig[],
): void {
	deps.setModels(models);
}

export function cycleModel(
	deps: ModelOpsDeps,
	direction: "forward" | "backward" = "forward",
): string {
	const config = deps.getConfig();
	const currentLevel = config.thinkingLevel ?? "off";
	const result = cycleModelPure(
		config.model,
		config.baseUrl,
		config.thinkingLevel,
		config.models ?? [],
		direction,
	);
	if (!result.didCycle) {
		return result.model;
	}

	if (result.baseUrl !== config.baseUrl) {
		deps.setBaseUrl(result.baseUrl);
	}

	if (result.thinkingLevelClamped) {
		deps.setThinkingLevel(result.thinkingLevel);
		deps.emit({
			type: "thinking_level_clamped",
			level: result.thinkingLevel,
			reason: `Model ${result.model} does not support ${currentLevel} thinking level`,
		});
	}

	deps.setModel(result.model);
	deps.appendModelChange(result.model);
	deps.emit({
		type: "model_cycle",
		model: result.model,
		fromModel: result.fromModel,
		thinkingLevel: result.thinkingLevel,
	});
	return result.model;
}

export function getThinkingLevel(deps: ModelOpsDeps): string {
	return deps.getConfig().thinkingLevel ?? "off";
}

export function setThinkingLevel(deps: ModelOpsDeps, level: string): void {
	const config = deps.getConfig();
	const currentLevel = config.thinkingLevel;
	const nextLevel = level as ThinkingLevel;
	deps.setThinkingLevel(nextLevel);
	deps.appendThinkingLevelChange(nextLevel);
	deps.emit({ type: "thinking_level_changed", level });
	// If level changed, emit model_cycle with updated thinking level
	if (level !== currentLevel) {
		deps.emit({
			type: "model_cycle",
			model: config.model,
			fromModel: config.model,
			thinkingLevel: level,
		});
	}
}

export function setModel(deps: ModelOpsDeps, model: string): void {
	const config = deps.getConfig();
	const oldModel = config.model;
	const targetUrl = getModelUrl(deps, model);
	if (targetUrl !== config.baseUrl) {
		deps.setBaseUrl(targetUrl);
	}
	deps.setModel(model);
	deps.appendModelChange(model);
	deps.emit({
		type: "model_cycle",
		model,
		fromModel: oldModel,
		thinkingLevel: config.thinkingLevel,
	});
}
