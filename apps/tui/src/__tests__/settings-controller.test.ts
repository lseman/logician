import { test } from "bun:test";
import assert from "node:assert/strict";
import { openSettingsSelector } from "../app/overlay-controllers/settings.ts";
import type { SettingDef } from "../overlays/settings-overlay.ts";

void test("settings exposes tri-state guards and every inference provider mode", async () => {
	let settings: SettingDef[] = [];
	const ctx = {
		bridge: {
			getSettingsData: () => ({
				model: "test",
				temperature: 0.5,
				maxTokens: 4096,
				maxIterations: 30,
				thinkingLevel: "off",
				inferenceMode: "none",
				permissionMode: "ask",
				executionProfile: "autonomous",
				guardsEnabled: false,
				guardMode: "auto",
				proactiveCompactionEnabled: true,
				postEditDiagnostics: true,
				rtkProxyEnabled: false,
				legroomEnabled: true,
				graphicianEnabled: true,
				fffgrepEnabled: true,
				memoryEnabled: true,
				duplicateGuardEnabled: true,
				failureGuardEnabled: false,
				continuationEnabled: true,
				autoRetryEnabled: true,
				progressStopEnabled: false,
			}),
		},
		settingsSelector: {
			setSettings: (value: SettingDef[]) => {
				settings = value;
			},
			setMessage: () => {},
			show: () => {},
		},
		tui: {
			showOverlay: () => ({ focus: () => {} }),
		},
	} as unknown as Parameters<typeof openSettingsSelector>[0];

	await openSettingsSelector(ctx);

	const guards = settings.find(setting => setting.name === "Guards");
	assert.deepEqual(
		guards?.options.map(option => option.value),
		["auto", "on", "off"],
	);
	assert.equal(guards?.currentValue, "auto");
	const inference = settings.find(setting => setting.name === "Inference mode");
	assert.ok(inference?.options.some(option => option.value === "auto"));
	assert.ok(inference?.options.some(option => option.value === "none"));
	assert.equal(
		settings.find(setting => setting.name === "Legroom SDK")?.currentValue,
		"on",
	);
});
