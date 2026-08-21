import type { AgentConfig } from "@logician/agent-core";
import { resolveAgentSettings } from "@logician/agent-core/runtime";
import type { RuntimeSettingsPatch } from "../bridge/types.ts";

export type RuntimeToggleKey =
	| "guardsEnabled"
	| "duplicateGuardEnabled"
	| "failureGuardEnabled"
	| "progressStopEnabled"
	| "continuationEnabled"
	| "autoRetryEnabled"
	| "proactiveCompactionEnabled"
	| "postEditDiagnostics"
	| "rtkProxyEnabled"
	| "ariadneEnabled"
	| "fffgrepEnabled"
	| "memoryEnabled";

export interface RuntimeSettingsHost {
	config(): Readonly<AgentConfig>;
	patchCore(patch: Partial<AgentConfig>): void;
	setThinkingLevel(level: AgentConfig["thinkingLevel"]): void;
	setTemperature(value: number): void;
	setReasoner(id: string): void;
	setSteeringInterrupt(enabled: boolean): void;
	setToggle(key: RuntimeToggleKey, enabled: boolean): void;
	permissionMode(): string;
	postEditDiagnostics(): boolean;
	memoryEnabled(): boolean;
}

export interface RuntimeSettingsView {
	model: string;
	temperature: number;
	maxTokens: number;
	maxIterations: number;
	thinkingLevel: string;
	inferenceMode: string;
	permissionMode: string;
	executionProfile: string;
	guardsEnabled: boolean;
	proactiveCompactionEnabled: boolean;
	postEditDiagnostics: boolean;
	rtkProxyEnabled: boolean;
	ariadneEnabled: boolean;
	fffgrepEnabled: boolean;
	memoryEnabled: boolean;
	duplicateGuardEnabled: boolean;
	failureGuardEnabled: boolean;
	continuationEnabled: boolean;
	autoRetryEnabled: boolean;
	progressStopEnabled: boolean;
	guardMode: "auto" | "on" | "off";
}

const TOGGLE_KEYS: readonly RuntimeToggleKey[] = [
	"guardsEnabled",
	"duplicateGuardEnabled",
	"failureGuardEnabled",
	"progressStopEnabled",
	"continuationEnabled",
	"autoRetryEnabled",
	"proactiveCompactionEnabled",
	"postEditDiagnostics",
	"rtkProxyEnabled",
	"ariadneEnabled",
	"fffgrepEnabled",
	"memoryEnabled",
];

/** Normalizes settings mutations and projects the single settings view used by clients. */
export class RuntimeSettingsManager {
	constructor(private readonly host: RuntimeSettingsHost) {}

	update(patch: RuntimeSettingsPatch): void {
		if (patch.thinkingLevel !== undefined)
			this.host.setThinkingLevel(patch.thinkingLevel);
		if (patch.temperature !== undefined)
			this.host.setTemperature(patch.temperature);
		if (patch.reasonerId !== undefined) this.host.setReasoner(patch.reasonerId);
		if (patch.steeringInterrupt !== undefined)
			this.host.setSteeringInterrupt(patch.steeringInterrupt);
		if (patch.guardMode !== undefined) {
			this.host.patchCore({
				guardsEnabled:
					patch.guardMode === "auto" ? undefined : patch.guardMode === "on",
			});
		}

		const corePatch: Partial<AgentConfig> = {};
		for (const key of [
			"inferenceMode",
			"maxTokens",
			"maxIterations",
			"executionProfile",
		] as const) {
			if (patch[key] !== undefined)
				Object.assign(corePatch, { [key]: patch[key] });
		}
		if (Object.keys(corePatch).length > 0) this.host.patchCore(corePatch);

		for (const key of TOGGLE_KEYS) {
			const enabled = patch[key];
			if (enabled !== undefined) this.host.setToggle(key, enabled);
		}
	}

	read(): RuntimeSettingsView {
		const config = this.host.config();
		const resolved = resolveAgentSettings(config as AgentConfig);
		return {
			model: config.model,
			temperature: config.temperature ?? 0.5,
			maxTokens: config.maxTokens ?? 4096,
			maxIterations: resolved.maxIterations,
			thinkingLevel: resolved.thinkingLevel,
			inferenceMode: resolved.inferenceMode,
			permissionMode: this.host.permissionMode(),
			executionProfile: resolved.executionProfile,
			guardsEnabled: config.guardsEnabled ?? false,
			proactiveCompactionEnabled: config.proactiveCompactionEnabled ?? true,
			postEditDiagnostics: this.host.postEditDiagnostics(),
			rtkProxyEnabled: config.rtkProxyEnabled ?? false,
			ariadneEnabled: config.ariadneEnabled ?? true,
			fffgrepEnabled: config.fffgrepEnabled ?? true,
			memoryEnabled: this.host.memoryEnabled(),
			duplicateGuardEnabled: config.duplicateGuardEnabled ?? true,
			failureGuardEnabled: config.failureGuardEnabled ?? false,
			continuationEnabled: config.continuationEnabled ?? true,
			autoRetryEnabled: config.autoRetryEnabled ?? true,
			progressStopEnabled: config.progressStopEnabled ?? false,
			guardMode:
				config.guardsEnabled === undefined
					? "auto"
					: config.guardsEnabled
						? "on"
						: "off",
		};
	}
}
