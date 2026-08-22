import { describe, expect, test } from "bun:test";
import type { AgentConfig } from "@logician/log-core";
import {
	SettingsGateway,
	type RuntimeToggleKey,
} from "../../runtime/settings-gateway.ts";

describe("SettingsGateway", () => {
	test("normalizes a client patch into core and feature mutations", () => {
		const config: AgentConfig = { baseUrl: "local", model: "model" };
		const toggles: Array<[RuntimeToggleKey, boolean]> = [];
		const settings = new SettingsGateway({
			config: () => config,
			patchCore: patch => Object.assign(config, patch),
			setThinkingLevel: level => {
				config.thinkingLevel = level;
			},
			setTemperature: value => {
				config.temperature = value;
			},
			setReasoner: () => {},
			setSteeringInterrupt: value => {
				config.steeringInterrupt = value;
			},
			setToggle: (key, enabled) => toggles.push([key, enabled]),
			permissionMode: () => "ask",
			postEditDiagnostics: () => true,
			memoryEnabled: () => false,
		});

		settings.update({
			maxIterations: 12,
			guardMode: "on",
			ariadneEnabled: false,
		});
		expect(config.maxIterations).toBe(12);
		expect(config.guardsEnabled).toBe(true);
		expect(toggles).toEqual([["ariadneEnabled", false]]);
		expect(settings.read().permissionMode).toBe("ask");
	});
});
