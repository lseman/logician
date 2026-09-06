import { spyOn, test } from "bun:test";
import assert from "node:assert/strict";
import * as configuration from "@logician/log-runtime/configuration";
import { handleSettingsSelectorAction, openSettingsSelector } from "../app/overlay-controllers/settings.ts";
import type { SettingDef } from "../overlays/settings-overlay.ts";

void test("settings exposes tri-state guards and every inference provider mode", async () => {
	let settings: SettingDef[] = [];
	const updates: Record<string, unknown>[] = [];
	const notifications: string[] = [];
	const ctx = {
		bridge: {
			updateSettings: (update: Record<string, unknown>) => updates.push(update),
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
				memoriamEnabled: true,
				graphicianEnabled: true,
				fffgrepEnabled: true,
				duplicateGuardEnabled: true,
				failureGuardEnabled: false,
				continuationEnabled: true,
				autoRetryEnabled: true,
				progressStopEnabled: false,
				workflowMode: "act",
			}),
		},
		workflowMode: "act",
		settingsSelector: {
			setSettings: (value: SettingDef[]) => {
				settings = value;
			},
			setMessage: () => {},
			show: () => {},
		},
		tui: {
			setShowHardwareCursor: () => {},
			showOverlay: () => ({ focus: () => {} }),
			requestRender: () => {},
			removeOverlay: () => {},
		},
		statusPanel: { update: () => {} },
		transcript: {
			getTurns: () => [],
			addSystemMessage: () => {},
		},
		transcriptDisplay: { setTurns: () => {} },
		notify: (message: string) => notifications.push(message),
	} as unknown as Parameters<typeof openSettingsSelector>[0];

	await openSettingsSelector(ctx);

	const guards = settings.find(setting => setting.name === "Guards");
	assert.deepEqual(
		guards?.options?.map(option => option.value) ?? [],
		["auto", "on", "off"],
	);
	assert.equal(guards?.currentValue, "auto");
	const inference = settings.find(setting => setting.name === "Inference mode");
	assert.ok(inference?.options?.some(option => option.value === "auto"));
	assert.ok(inference?.options?.some(option => option.value === "none"));
	assert.equal(
		settings.find(setting => setting.name === "Legroom SDK")?.currentValue,
		"on",
	);
	const budgetStop = settings.find(setting => setting.name === "Budget early-stop");
	assert.ok(budgetStop);
	const save = spyOn(configuration, "saveConfigField").mockReturnValue(true);
	try {
		for (const option of budgetStop.options ?? []) {
			handleSettingsSelectorAction(ctx, { type: "change", settingName: budgetStop.name, value: option.value });
			assert.deepEqual(updates.at(-1), { progressStopEnabled: option.value === "true" });
			assert.deepEqual(save.mock.calls.at(-1), ["progressStopEnabled", option.value === "true"]);
		}
		assert.equal(notifications.some(message => message.includes("Unknown setting")), false);
	} finally {
		save.mockRestore();
	}

});
