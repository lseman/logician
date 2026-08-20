// ── Model cycling + thinking-level clamping for AgentHarness ──────────────
// Pure helpers; the harness still owns config mutation and event emission.

import type {
	AgentModelConfig,
	ThinkingLevel,
} from "../../types/types-config.ts";
import { THINKING_LEVELS } from "../../types/types-config.ts";

/** Clamp an arbitrary string to a known ThinkingLevel, defaulting to "off". */
export function clampThinkingLevel(level: string): ThinkingLevel {
	const idx = THINKING_LEVELS.indexOf(level as ThinkingLevel);
	if (idx >= 0) return level as ThinkingLevel;
	return "off";
}

/** Resolve the baseUrl configured for a given model name, falling back to `defaultBaseUrl`. */
export function resolveModelUrl(
	models: AgentModelConfig[] | undefined,
	modelName: string,
	defaultBaseUrl: string,
): string {
	const found = models?.find(m => m.model === modelName);
	return found?.url ?? defaultBaseUrl;
}

export interface CycleModelResult {
	/** False when there's nothing to cycle to (no configured models, or ring collapses to one entry) — harness should return the current model as-is without emitting events. */
	didCycle: boolean;
	model: string;
	fromModel: string;
	baseUrl: string;
	thinkingLevel: ThinkingLevel;
	thinkingLevelClamped: boolean;
}

/**
 * Compute the next model in the cycling ring: [currentModel, ...configuredModels],
 * wrapping around. Returns the resolved model/baseUrl/thinkingLevel for the
 * harness to apply; does not mutate anything itself.
 */
export function cycleModel(
	currentModel: string,
	currentBaseUrl: string,
	currentThinkingLevel: string | undefined,
	configuredModels: AgentModelConfig[],
	direction: "forward" | "backward" = "forward",
): CycleModelResult {
	const noop: CycleModelResult = {
		didCycle: false,
		model: currentModel,
		fromModel: currentModel,
		baseUrl: currentBaseUrl,
		thinkingLevel: clampThinkingLevel(currentThinkingLevel ?? "off"),
		thinkingLevelClamped: false,
	};

	if (configuredModels.length === 0) {
		return noop;
	}

	const cycleModels: Array<{ name: string; model: string }> =
		configuredModels.map(m => ({
			name: m.name,
			model: m.model,
		}));

	const currentInList = cycleModels.some(m => m.model === currentModel);
	if (!currentInList) {
		cycleModels.unshift({ name: currentModel, model: currentModel });
	}

	if (cycleModels.length <= 1) {
		return noop;
	}

	const currentIndex = cycleModels.findIndex(m => m.model === currentModel);
	const nextIndex =
		direction === "forward"
			? (currentIndex + 1) % cycleModels.length
			: (currentIndex - 1 + cycleModels.length) % cycleModels.length;
	const next = cycleModels[nextIndex];
	const model = next?.model ?? currentModel;

	const targetUrl = resolveModelUrl(configuredModels, model, currentBaseUrl);

	const currentLevel = currentThinkingLevel ?? "off";
	const clampedLevel = clampThinkingLevel(currentLevel);

	return {
		didCycle: true,
		model,
		fromModel: currentModel,
		baseUrl: targetUrl,
		thinkingLevel: clampedLevel,
		thinkingLevelClamped: clampedLevel !== currentLevel,
	};
}
