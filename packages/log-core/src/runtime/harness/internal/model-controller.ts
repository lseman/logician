import type { ConfigurationStore } from "../../../control/configuration/configuration-store.ts";
import type { LLMBackend } from "../../../capabilities/provider/backend.ts";
import type {
	AgentConfig,
	AgentModelConfig,
	ThinkingLevel,
} from "../../../system/types/types-config.ts";
import type { AgentEvent } from "../../../system/types/types-messages.ts";
import { cycleModel, resolveModelUrl } from "../live/model.ts";

export interface HarnessModelControllerOptions {
	backend: LLMBackend;
	configuration: ConfigurationStore<AgentConfig>;
	emit(event: AgentEvent): void;
	persistModel(model: string): void;
	persistThinking(level: ThinkingLevel): void;
}

/** Owns model endpoint selection, cycling, and thinking-level compatibility. */
export class HarnessModelController {
	private activeBackend: LLMBackend;

	constructor(private readonly options: HarnessModelControllerOptions) {
		this.activeBackend = options.backend;
	}

	get backend(): LLMBackend {
		return this.activeBackend;
	}

	get model(): string {
		return this.options.configuration.current.model;
	}

	get baseUrl(): string {
		return this.options.configuration.current.baseUrl;
	}

	models(): string[] {
		const config = this.options.configuration.current;
		const configured = config.models ?? [];
		return [
			...(configured.some(option => option.model === config.model)
				? []
				: [config.model]),
			...configured.map(option => option.model),
		];
	}

	setEndpoint(model: string, baseUrl: string): void {
		this.options.configuration.update({ model, baseUrl });
		this.activeBackend =
			this.activeBackend.withEndpoint?.(model, baseUrl) ??
			this.activeBackend.withModel(model);
	}

	setModels(models: AgentModelConfig[]): void {
		this.options.configuration.update({ models });
	}

	cycle(direction: "forward" | "backward" = "forward"): string {
		const config = this.options.configuration.current;
		const currentLevel = config.thinkingLevel ?? "off";
		const result = cycleModel(
			config.model,
			config.baseUrl,
			config.thinkingLevel,
			config.models ?? [],
			direction,
		);
		if (!result.didCycle) return result.model;
		this.options.configuration.update({
			baseUrl: result.baseUrl,
			model: result.model,
			...(result.thinkingLevelClamped
				? { thinkingLevel: result.thinkingLevel }
				: {}),
		});
		if (result.thinkingLevelClamped) {
			this.options.emit({
				type: "thinking_level_clamped",
				level: result.thinkingLevel,
				reason: `Model ${result.model} does not support ${currentLevel} thinking level`,
			});
		}
		this.activeBackend =
			this.activeBackend.withEndpoint?.(result.model, result.baseUrl) ??
			this.activeBackend.withModel(result.model);
		this.options.persistModel(result.model);
		this.options.emit({
			type: "model_cycle",
			model: result.model,
			fromModel: result.fromModel,
			thinkingLevel: result.thinkingLevel,
		});
		return result.model;
	}

	get thinkingLevel(): string {
		return this.options.configuration.current.thinkingLevel ?? "off";
	}

	setThinkingLevel(level: string): void {
		const config = this.options.configuration.current;
		const previous = config.thinkingLevel;
		const thinkingLevel = level as ThinkingLevel;
		this.options.configuration.update({ thinkingLevel });
		this.options.persistThinking(thinkingLevel);
		this.options.emit({ type: "thinking_level_changed", level });
		if (level !== previous) {
			this.options.emit({
				type: "model_cycle",
				model: config.model,
				fromModel: config.model,
				thinkingLevel: level,
			});
		}
	}

	setModel(model: string): void {
		const config = this.options.configuration.current;
		const oldModel = config.model;
		const baseUrl = resolveModelUrl(config.models, model, config.baseUrl);
		this.options.configuration.update({ model, baseUrl });
		this.activeBackend =
			this.activeBackend.withEndpoint?.(model, baseUrl) ??
			this.activeBackend.withModel(model);
		this.options.persistModel(model);
		this.options.emit({
			type: "model_cycle",
			model,
			fromModel: oldModel,
			thinkingLevel: config.thinkingLevel,
		});
	}
}
