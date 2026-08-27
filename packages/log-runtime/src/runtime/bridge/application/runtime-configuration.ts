import type { AgentConfig } from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";
import {
	type RuntimeSettingsView,
	type RuntimeToggleKey,
	SettingsGateway,
} from "../support/settings-gateway.ts";
import type { RuntimeSettingsPatch } from "../types.ts";

type ThinkingLevel = NonNullable<AgentConfig["thinkingLevel"]>;

export interface RuntimeConfigurationDependencies {
	config: AgentConfig;
	backend: { setDefaultThinkingLevel(level: ThinkingLevel): void };
	session: () => RuntimeSettingsSession | null;
	sessionId: () => string;
	tools: RuntimeSettingsTools;
	interactions: { readonly mode: string };
	legroom: {
		isEnabled(): boolean;
		setEnabled(enabled: boolean): void;
	};
	memoriam: {
		isEnabled(): boolean;
		setEnabled(enabled: boolean): void;
	};
	defaultTools: () => AgentConfig["tools"];
	setReasoner: (id: string) => void;
	emit: (event: RuntimeEvent) => void;
	postEditDiagnostics: boolean;
}

export interface RuntimeSettingsSession {
	configure(patch: Partial<AgentConfig>): void;
	models: { setThinkingLevel(level: ThinkingLevel): void };
	enableAutoCompaction(enabled: boolean): void;
}

export interface RuntimeSettingsTools {
	setGraphicianEnabled(enabled: boolean): void;
	setFffgrepEnabled(enabled: boolean): void;
}

/** Owns mutable runtime settings and propagates each change to live capabilities. */
export class RuntimeConfiguration {
	private readonly dependencies: RuntimeConfigurationDependencies;
	private readonly settings: SettingsGateway;
	private postEditDiagnosticsEnabled: boolean;

	constructor(dependencies: RuntimeConfigurationDependencies) {
		this.dependencies = dependencies;
		this.postEditDiagnosticsEnabled = dependencies.postEditDiagnostics;
		this.settings = new SettingsGateway({
			config: () => this.dependencies.config,
			patchCore: patch => this.patchCore(patch),
			setThinkingLevel: level => {
				if (level !== undefined) this.setThinkingLevel(level);
			},
			setTemperature: value => this.setTemperature(value),
			setReasoner: id => this.dependencies.setReasoner(id),
			setSteeringInterrupt: enabled =>
				this.dependencies.session()?.configure({ steeringInterrupt: enabled }),
			setToggle: (key, enabled) => this.setToggle(key, enabled),
			permissionMode: () => this.dependencies.interactions.mode,
			postEditDiagnostics: () => this.postEditDiagnosticsEnabled,
			legroomEnabled: () => this.dependencies.legroom.isEnabled(),
			memoriamEnabled: () => this.dependencies.memoriam.isEnabled(),
		});
	}

	update(patch: RuntimeSettingsPatch): void {
		this.settings.update(patch);
	}

	read(): RuntimeSettingsView {
		return this.settings.read();
	}

	get postEditDiagnostics(): boolean {
		return this.postEditDiagnosticsEnabled;
	}

	get legroomEnabled(): boolean {
		return this.dependencies.legroom.isEnabled();
	}

	private patchCore(patch: Partial<AgentConfig>): void {
		Object.assign(this.dependencies.config, patch);
		this.dependencies.session()?.configure(patch);
	}

	private setThinkingLevel(level: ThinkingLevel): void {
		this.dependencies.config.thinkingLevel = level;
		this.dependencies.session()?.models.setThinkingLevel(level);
		this.dependencies.backend.setDefaultThinkingLevel(level);
	}

	private setTemperature(temperature: number): void {
		this.dependencies.config.temperature = temperature;
		this.dependencies.session()?.configure({ temperature });
	}

	private setToggle(key: RuntimeToggleKey, enabled: boolean): void {
		if (key === "postEditDiagnostics") {
			this.postEditDiagnosticsEnabled = enabled;
			return;
		}
		if (key === "legroomEnabled") {
			this.dependencies.legroom.setEnabled(enabled);
			this.notice(
				"Legroom",
				enabled ? "Legroom SDK enabled" : "Legroom SDK disabled",
			);
			return;
		}
		if (key === "memoriamEnabled") {
			this.dependencies.memoriam.setEnabled(enabled);
			this.notice(
				"Memoriam",
				enabled ? "Memoriam SDK enabled" : "Memoriam SDK disabled",
			);
			return;
		}
		if (key === "graphicianEnabled" || key === "fffgrepEnabled") {
			this.setToolToggle(key, enabled);
			return;
		}

		Object.assign(this.dependencies.config, { [key]: enabled });
		if (
			key === "guardsEnabled" ||
			key === "duplicateGuardEnabled" ||
			key === "failureGuardEnabled" ||
			key === "progressStopEnabled" ||
			key === "continuationEnabled" ||
			key === "autoRetryEnabled"
		) {
			this.dependencies.session()?.configure({ [key]: enabled });
		}
		if (key === "proactiveCompactionEnabled") {
			this.dependencies.session()?.enableAutoCompaction(enabled);
		}
	}

	private setToolToggle(
		key: "graphicianEnabled" | "fffgrepEnabled",
		enabled: boolean,
	): void {
		this.dependencies.config[key] = enabled;
		if (key === "graphicianEnabled") {
			this.dependencies.tools.setGraphicianEnabled(enabled);
		} else {
			this.dependencies.tools.setFffgrepEnabled(enabled);
		}
		this.dependencies.config.tools = this.dependencies.defaultTools();
		this.dependencies.session()?.configure({
			tools: this.dependencies.config.tools,
		});
	}

	private notice(label: string, text: string): void {
		this.dependencies.emit({ type: "notice", level: "info", label, text });
	}
}
