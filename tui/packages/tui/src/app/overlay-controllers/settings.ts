// ── Settings selector controller ───────────────────────────────────────────

import { saveConfigField } from "@logician/coding-agent/configuration";
import type { SettingDef, SettingsSelectorAction } from "../../overlays/settings-overlay.ts";
import { applyThinkingLevel, setExecutionProfile, setInferenceMode } from "../inference-settings.ts";
import type { OverlayHandlersCtx } from "./context.ts";
import { openModelSelector } from "./selectors.ts";


// ── Settings selector ───────────────────────────────────────────────────

export async function openSettingsSelector(ctx: OverlayHandlersCtx): Promise<void> {
	try {
		const data = ctx.bridge.getSettingsData();
		const thinkingLevels = [
			"off",
			"minimal",
			"low",
			"medium",
			"high",
			"xhigh",
		];
		const permissionModes = ["acceptAll", "acceptEdits", "ask", "plan"];
		const settings: SettingDef[] = [
			{
				name: "Model",
				currentValue: data.model,
				description: "LLM model to use",
				options: [{ label: data.model, value: data.model, current: true }],
			},
			{
				name: "Temperature",
				currentValue: String(data.temperature),
				description: "Sampling temperature (0–2)",
				options: [0.0, 0.3, 0.5, 0.7, 1.0].map((v) => ({
					label: String(v),
					value: String(v),
					current: Math.abs(data.temperature - v) < 0.001,
				})),
			},
			{
				name: "Max tokens",
				currentValue: String(data.maxTokens),
				description: "Maximum response tokens",
				options: [1024, 2048, 4096, 8192, 16384].map((v) => ({
					label: String(v),
					value: String(v),
					current: data.maxTokens === v,
				})),
			},
			{
				name: "Max iterations",
				currentValue: String(data.maxIterations),
				description: "Maximum tool-use iterations per turn",
				options: [10, 20, 30, 50, 100].map((v) => ({
					label: String(v),
					value: String(v),
					current: data.maxIterations === v,
				})),
			},
			{
				name: "Thinking level",
				currentValue: data.thinkingLevel,
				description: "Depth of reasoning before responding",
				options: thinkingLevels.map((v) => ({
					label: v.charAt(0).toUpperCase() + v.slice(1),
					value: v,
					current: data.thinkingLevel === v,
				})),
			},
			{
				name: "Permission mode",
				currentValue: data.permissionMode,
				description: "How the agent handles tool permissions",
				options: permissionModes.map((v) => ({
					label: v,
					value: v,
					current: data.permissionMode === v,
				})),
			},
			{
				name: "Guards",
				currentValue: data.guardsEnabled ? "on" : "off",
				description: "Safety guards against harmful tool use",
				options: [
					{
						label: "on",
						value: "true",
						current: data.guardsEnabled,
						toggleOn: true,
					},
					{
						label: "off",
						value: "false",
						current: !data.guardsEnabled,
						toggleOn: false,
					},
				],
			},
			{
				name: "Compaction",
				currentValue: data.proactiveCompactionEnabled ? "on" : "off",
				description: "Auto-compact context to save tokens",
				options: [
					{
						label: "on",
						value: "true",
						current: data.proactiveCompactionEnabled,
						toggleOn: true,
					},
					{
						label: "off",
						value: "false",
						current: !data.proactiveCompactionEnabled,
						toggleOn: false,
					},
				],
			},
			{
				name: "Inference mode",
				currentValue: data.inferenceMode,
				description: "Pre-defined sampling parameter set (Alt+M to cycle)",
				options: [
					{
						label: "Think General",
						value: "thinking-general",
						current: data.inferenceMode === "thinking-general",
					},
					{
						label: "Think Code",
						value: "thinking-coding",
						current: data.inferenceMode === "thinking-coding",
					},
					{
						label: "Instruct",
						value: "instruct-general",
						current: data.inferenceMode === "instruct-general",
					},
					{
						label: "Reason",
						value: "instruct-reasoning",
						current: data.inferenceMode === "instruct-reasoning",
					},
				],
			},
			{
				name: "Post-edit diagnostics",
				currentValue: data.postEditDiagnostics ? "on" : "off",
				description: "Check edited files against the project",
				options: [
					{
						label: "on",
						value: "true",
						current: data.postEditDiagnostics,
						toggleOn: true,
					},
					{
						label: "off",
						value: "false",
						current: !data.postEditDiagnostics,
						toggleOn: false,
					},
				],
			},
			{
				name: "RTK CLI proxy",
				currentValue: data.rtkProxyEnabled ? "on" : "off",
				description:
					"Prefix all bash commands with `rtk` for 60-90% output compression",
				options: [
					{
						label: "on",
						value: "true",
						current: data.rtkProxyEnabled,
						toggleOn: true,
					},
					{
						label: "off",
						value: "false",
						current: !data.rtkProxyEnabled,
						toggleOn: false,
					},
				],
			},
			{
				name: "Execution policy",
				currentValue: data.executionProfile,
				description:
					"Agent policy ownership — autonomous uses built-in policies, minimal leaves stop policy to the caller",
				options: [
					{
						label: "autonomous",
						value: "autonomous",
						current: data.executionProfile === "autonomous",
					},
					{
						label: "minimal",
						value: "minimal",
						current: data.executionProfile === "minimal",
					},
				],
			},
		];
		ctx.settingsSelector.setSettings(settings);
		ctx.settingsSelector.setMessage(
			"Enter selects a setting · Enter in detail applies",
		);
		ctx.settingsSelector.show();
		const overlay = ctx.tui.showOverlay(ctx.settingsSelector, {
			anchor: "aboveInput",
			align: "left",
			maxHeight: 18,
		});
		overlay.focus();
	} catch (e: unknown) {
		ctx.transcript.addSystemMessage(
			`Settings error: ${e instanceof Error ? e.message : String(e)}`,
		);
	}
}

export function handleSettingsSelectorAction(
	ctx: OverlayHandlersCtx,
	action: SettingsSelectorAction,
): void {
	if (action.type === "close") {
		ctx.tui.removeOverlay(ctx.settingsSelector);
		ctx.statusPanel.update({ phase: "ready" });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
		return;
	}
	if (
		action.type === "open" &&
		action.settingName.toLowerCase() === "model"
	) {
		ctx.tui.removeOverlay(ctx.settingsSelector);
		openModelSelector(ctx);
		return;
	}
	if (action.type !== "change") return;
	// action.type === "change"
	const { settingName, value } = action;
	ctx.settingsSelector.setMessage(`Applying ${settingName}...`);
	ctx.tui.requestRender();

	// Apply the setting via the bridge
	switch (settingName.toLowerCase()) {
		case "model":
			ctx.bridge.setModel(value);
			ctx.notify(`Model: ${value}`, "success");
			break;
		case "temperature": {
			const num = Number(value);
			if (Number.isFinite(num) && num >= 0 && num <= 2) {
				ctx.bridge.setTemperature(num);
				ctx.notify(`Temperature: ${num}`, "success");
			} else {
				ctx.notify("Temperature must be between 0 and 2.", "error");
			}
			break;
		}
		case "max tokens": {
			const num = Number.parseInt(value, 10);
			if (Number.isFinite(num) && num >= 1) {
				ctx.bridge.setMaxTokens(num);
				ctx.notify(`Max tokens: ${num}`, "success");
			} else {
				ctx.notify("Max tokens must be a positive integer.", "error");
			}
			break;
		}
		case "max iterations": {
			const num = Number.parseInt(value, 10);
			if (Number.isFinite(num) && num >= 1) {
				ctx.bridge.setMaxIterations(num);
				ctx.notify(`Max iterations: ${num}`, "success");
			} else {
				ctx.notify("Max iterations must be a positive integer.", "error");
			}
			break;
		}
		case "thinking level":
			applyThinkingLevel(ctx, value);
			ctx.notify(`Thinking level: ${value}`, "success");
			break;
		case "permission mode":
			ctx.bridge.setPermissionMode(
				value as "acceptAll" | "acceptEdits" | "ask" | "plan",
			);
			ctx.notify(`Permission mode: ${value}`, "success");
			break;
		case "guards": {
			const on = value === "true";
			ctx.bridge.setRuntimeToggle("guardsEnabled", on);
			ctx.notify(`Guards: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "compaction": {
			const on = value === "true";
			ctx.bridge.setRuntimeToggle("proactiveCompactionEnabled", on);
			ctx.notify(`Compaction: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "post-edit diagnostics": {
			const on = value === "true";
			ctx.bridge.setRuntimeToggle("postEditDiagnostics", on);
			saveConfigField("postEditDiagnostics", on);
			ctx.notify(`Post-edit diagnostics: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "rtk cli proxy": {
			const on = value === "true";
			ctx.bridge.setRuntimeToggle("rtkProxyEnabled", on);
			saveConfigField("rtkProxyEnabled", on);
			ctx.statusPanel.update({ rtkProxyEnabled: on });
			ctx.notify(`RTK proxy: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "inference mode": {
			const valid = [
				"thinking-general",
				"thinking-coding",
				"instruct-general",
				"instruct-reasoning",
			];
			if (!valid.includes(value)) {
				ctx.notify(
					`Invalid inference mode: ${value}. Valid: ${valid.join(", ")}`,
					"error",
				);
			} else {
				setInferenceMode(ctx, value);
			}
			break;
		}
		case "execution policy": {
			const valid: Array<"autonomous" | "minimal"> = [
				"autonomous",
				"minimal",
			];
			if (!valid.includes(value as (typeof valid)[number])) {
				ctx.notify(
					`Invalid execution policy: ${value}. Valid: ${valid.join(", ")}`,
					"error",
				);
			} else {
				setExecutionProfile(ctx, value as "autonomous" | "minimal");
				ctx.notify(`Execution policy: ${value}`, "success");
			}
			break;
		}
		default:
			ctx.notify(`Unknown setting: ${settingName}`, "error");
	}

	ctx.tui.removeOverlay(ctx.settingsSelector);
	ctx.statusPanel.update({ phase: "ready" });
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.tui.requestRender();
}
