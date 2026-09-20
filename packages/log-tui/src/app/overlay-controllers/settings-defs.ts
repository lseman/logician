// ── Schema-driven settings definitions ────────────────────────────────────
// Generates the settings-overlay definition list from the log-runtime
// settings schema registry, replacing the previous hand-maintained list.
// Adding a setting to the UI = add `ui` metadata to the registry entry.

import {
	getEnumValues,
	getUiSettings,
	type SettingSpec,
} from "@logician/log-runtime/configuration";
import type {
	SettingDef,
	SettingOption,
} from "../../overlays/settings-overlay.ts";

/** Runtime settings view as returned by `bridge.getSettingsData()`. */
export interface SettingsData {
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
	graphicianEnabled: boolean;
	fffgrepEnabled: boolean;
	legroomEnabled: boolean;
	memoriamEnabled: boolean;
	duplicateGuardEnabled: boolean;
	failureGuardEnabled: boolean;
	continuationEnabled: boolean;
	autoRetryEnabled: boolean;
	progressStopEnabled: boolean;
	guardMode: "auto" | "on" | "off";
}

function toggleDef(
	name: string,
	tab: string,
	section: string,
	description: string,
	enabled: boolean,
): SettingDef {
	return {
		name,
		tab,
		section,
		currentValue: enabled ? "on" : "off",
		description,
		displayType: "toggle",
		options: [
			{ label: "on", value: "true", current: enabled, toggleOn: true },
			{ label: "off", value: "false", current: !enabled, toggleOn: false },
		],
	};
}

function enumDef(
	name: string,
	tab: string,
	section: string,
	description: string,
	values: readonly string[],
	current: string,
	labels?: Readonly<Record<string, string>>,
	extra?: Partial<SettingOption>,
): SettingDef {
	const options = values.map(value => ({
		label: labels?.[value] ?? value,
		value,
		current: value === current,
		...(value === "on" ? { toggleOn: true as const } : {}),
		...(value === "off" ? { toggleOn: false as const } : {}),
		...extra,
	}));
	return {
		name,
		tab,
		section,
		currentValue: current,
		description,
		displayType: "select",
		options,
	};
}

function numberDef(
	name: string,
	tab: string,
	section: string,
	description: string,
	value: number,
	presets: readonly number[] | undefined,
): SettingDef {
	const options: SettingOption[] | undefined = presets?.map(preset => ({
		label: String(preset),
		value: String(preset),
		current: Math.abs(value - preset) < 0.001,
	}));
	return {
		name,
		tab,
		section,
		currentValue: String(value),
		description,
		displayType: "number",
		...(options ? { options } : {}),
	};
}

function specMeta(spec: SettingSpec): {
	tab: string;
	section: string;
	description: string;
} {
	const meta = spec.ui;
	if (!meta) throw new Error("spec has no ui metadata");
	return { tab: meta.tab, section: meta.group, description: meta.description };
}

/**
 * Build the settings-definition list from the schema registry.
 *
 * @param data — live runtime settings view (`bridge.getSettingsData()`)
 * @param workflowMode — current act/plan workflow mode
 */
export function buildSettingsDefs(
	data: SettingsData,
	workflowMode: string,
): SettingDef[] {
	const defs: SettingDef[] = [];
	for (const { key, spec } of getUiSettings()) {
		const meta = spec.ui;
		if (!meta) continue;
		const name = meta.label;
		const { tab, section, description } = specMeta(spec);

		switch (name) {
			case "Model":
				// Opens the model selector rather than editing inline.
				defs.push({
					name,
					tab,
					section,
					currentValue: data.model,
					description,
					displayType: "text",
				});
				break;
			case "Guards":
				defs.push(
					enumDef(
						name,
						tab,
						section,
						description,
						["auto", "on", "off"],
						data.guardMode,
						{ auto: "Auto" },
					),
				);
				break;
			case "Compaction":
				defs.push(
					toggleDef(
						name,
						tab,
						section,
						description,
						data.proactiveCompactionEnabled,
					),
				);
				break;
			case "Legroom SDK":
				defs.push(
					toggleDef(name, tab, section, description, data.legroomEnabled),
				);
				break;
			case "Memoriam SDK":
				defs.push(
					toggleDef(name, tab, section, description, data.memoriamEnabled),
				);
				break;
			case "Workflow mode":
				defs.push(
					enumDef(
						name,
						tab,
						section,
						description,
						getEnumValues(key) ?? [],
						workflowMode,
						meta.labels,
					),
				);
				break;
			case "Thinking level":
			case "Inference mode":
				defs.push(
					enumDef(
						name,
						tab,
						section,
						description,
						getEnumValues(key) ?? [],
						key === "thinkingLevel" ? data.thinkingLevel : data.inferenceMode,
						meta.labels,
					),
				);
				break;
			case "Execution policy":
				defs.push(
					enumDef(
						name,
						tab,
						section,
						description,
						getEnumValues(key) ?? [],
						data.executionProfile,
						meta.labels,
					),
				);
				break;
			case "Temperature":
				defs.push(
					numberDef(
						name,
						tab,
						section,
						description,
						data.temperature,
						meta.presets,
					),
				);
				break;
			case "Max tokens":
				defs.push(
					numberDef(
						name,
						tab,
						section,
						description,
						data.maxTokens,
						meta.presets,
					),
				);
				break;
			case "Max iterations":
				defs.push(
					numberDef(
						name,
						tab,
						section,
						description,
						data.maxIterations,
						meta.presets,
					),
				);
				break;
			default: {
				// Plain boolean toggles: the runtime view carries the same
				// camelCase field as the config key.
				const enabled = data[key as keyof SettingsData] as boolean | undefined;
				defs.push(toggleDef(name, tab, section, description, enabled === true));
				break;
			}
		}
	}
	return defs;
}
