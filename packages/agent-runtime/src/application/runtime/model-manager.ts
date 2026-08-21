import type { AgentConfig, AgentModelConfig } from "@logician/agent-core";
import type { AgentHarness } from "@logician/agent-core/harness";

export interface ModelOption {
	key: string;
	name: string;
	model: string;
	url: string;
	active: boolean;
}

/** Model selection and endpoint changes for a live runtime. */
export class RuntimeModelManager {
	constructor(
		private readonly config: () => AgentConfig,
		private readonly harness: () => AgentHarness | null,
	) {}

	current(): string {
		return this.harness()?.models.model ?? this.config().model ?? "";
	}

	baseUrl(): string {
		return this.config().baseUrl;
	}

	list(): string[] {
		const configured = this.config().models;
		return configured?.length
			? configured.map(model => model.model)
			: [this.current()];
	}

	options(): ModelOption[] {
		const configured = this.config().models ?? [];
		if (configured.length === 0) {
			return [
				{
					key: this.current(),
					name: this.current(),
					model: this.current(),
					url: this.baseUrl(),
					active: true,
				},
			];
		}
		return configured.map((option, index) => {
			const url = option.url || this.config().baseUrl;
			return {
				key: `${index}:${option.name}`,
				name: option.name,
				model: option.model,
				url,
				active: option.model === this.current() && url === this.baseUrl(),
			};
		});
	}

	selectOption(key: string): { model: string; url: string } | null {
		const option = this.options().find(candidate => candidate.key === key);
		if (!option) return null;
		this.config().model = option.model;
		this.config().baseUrl = option.url;
		this.harness()?.models.setEndpoint(option.model, option.url);
		return { model: option.model, url: option.url };
	}

	cycle(direction: "forward" | "backward" = "forward"): string | null {
		return this.harness()?.models.cycle(direction) ?? null;
	}

	setAvailable(models: AgentModelConfig[]): void {
		this.config().models = models;
		this.harness()?.models.setModels(models);
	}

	select(model: string): void {
		this.config().model = model;
		this.harness()?.models.setModel(model);
	}
}
