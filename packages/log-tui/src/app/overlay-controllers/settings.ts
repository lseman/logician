// ── Settings selector controller ───────────────────────────────────────────
import {
	getEnumValues,
	saveConfigField,
	saveConfigNestedField,
} from "@logician/log-runtime/configuration";
import type { SettingsSelectorAction } from "../../overlays/settings-overlay.ts";
import {
	applyThinkingLevel,
	setExecutionProfile,
	setInferenceMode,
	setPlanMode,
} from "../inference-settings.ts";
import type { OverlayHandlersCtx } from "./context.ts";
import { openModelSelector } from "./selectors.ts";
import { buildSettingsDefs } from "./settings-defs.ts";

// ── Settings selector ───────────────────────────────────────────────────

export async function openSettingsSelector(
	ctx: OverlayHandlersCtx,
): Promise<void> {
	try {
		const data = ctx.bridge.getSettingsData();
		const settings = buildSettingsDefs(data, ctx.workflowMode);
		ctx.tui.setShowHardwareCursor(false);
		ctx.settingsSelector.setSettings(settings);
		ctx.settingsSelector.show();
		ctx.settingsSelector.setMessage(
			"Changes apply to your current configuration.",
		);
		const overlay = ctx.tui.showOverlay(ctx.settingsSelector, {
			anchor: "center",
			width: "100%",
			maxHeight: "100%",
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
		ctx.tui.setShowHardwareCursor(true);
		ctx.tui.removeOverlay(ctx.settingsSelector);
		ctx.statusPanel.update({ phase: "ready" });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
		return;
	}
	if (action.type === "open" && action.settingName.toLowerCase() === "model") {
		ctx.tui.setShowHardwareCursor(true);
		ctx.tui.removeOverlay(ctx.settingsSelector);
		openModelSelector(ctx);
		return;
	}
	if (action.type !== "change" && action.type !== "confirm") return;
	// action.type === "change" | "confirm"
	const { settingName, value } = action;
	ctx.settingsSelector.setMessage(`Applying ${settingName}...`);
	ctx.tui.requestRender();

	// Apply the setting via the bridge
	switch (settingName.toLowerCase()) {
		case "model":
			ctx.bridge.models.select(value);
			ctx.notify(`Model: ${value}`, "success");
			break;
		case "temperature": {
			const num = Number(value);
			if (Number.isFinite(num) && num >= 0 && num <= 2) {
				ctx.bridge.updateSettings({ temperature: num });
				saveConfigField("temperature", num);
				ctx.notify(`Temperature: ${num}`, "success");
			} else {
				ctx.notify("Temperature must be between 0 and 2.", "error");
			}
			break;
		}
		case "max tokens": {
			const num = Number.parseInt(value, 10);
			if (Number.isFinite(num) && num >= 1) {
				ctx.bridge.updateSettings({ maxTokens: num });
				saveConfigField("maxTokens", num);
				ctx.notify(`Max tokens: ${num}`, "success");
			} else {
				ctx.notify("Max tokens must be a positive integer.", "error");
			}
			break;
		}
		case "max iterations": {
			const num = Number.parseInt(value, 10);
			if (Number.isFinite(num) && num >= 1) {
				ctx.bridge.updateSettings({ maxIterations: num });
				saveConfigField("maxIterations", num);
				ctx.notify(`Max iterations: ${num}`, "success");
			} else {
				ctx.notify("Max iterations must be a positive integer.", "error");
			}
			break;
		}
		case "thinking level":
			applyThinkingLevel(ctx, value, { persist: true });
			ctx.notify(`Thinking level: ${value}`, "success");
			break;
		case "workflow mode":
			setPlanMode(ctx, value === "plan");
			ctx.notify(`Permission mode: ${value}`, "success");
			break;
		case "guards": {
			const mode = value as "auto" | "on" | "off";
			ctx.bridge.updateSettings({ guardMode: mode });
			saveConfigField(
				"guardsEnabled",
				mode === "auto" ? undefined : mode === "on",
			);
			ctx.notify(`Guards: ${mode}`, "success");
			break;
		}
		case "compaction": {
			const on = value === "true";
			ctx.bridge.updateSettings({ proactiveCompactionEnabled: on });
			saveConfigNestedField("compaction", "enabled", on);
			ctx.notify(`Compaction: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "post-edit diagnostics": {
			const on = value === "true";
			ctx.bridge.updateSettings({ postEditDiagnostics: on });
			saveConfigField("postEditDiagnostics", on);
			ctx.notify(`Post-edit diagnostics: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "rtk cli proxy": {
			const on = value === "true";
			ctx.bridge.updateSettings({ rtkProxyEnabled: on });
			saveConfigField("rtkProxyEnabled", on);
			ctx.statusPanel.update({ rtkProxyEnabled: on });
			ctx.notify(`RTK proxy: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "legroom sdk": {
			const on = value === "true";
			ctx.bridge.updateSettings({ legroomEnabled: on });
			saveConfigNestedField("legroom", "mode", on ? "sdk" : "off");
			ctx.statusPanel.update({ legroomEnabled: on });
			ctx.notify(`Legroom SDK: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "memoriam sdk": {
			const on = value === "true";
			ctx.bridge.updateSettings({ memoriamEnabled: on });
			saveConfigNestedField("memoriam", "mode", on ? "sdk" : "off");
			ctx.statusPanel.update({ memoriamEnabled: on });
			ctx.notify(`Memoriam SDK: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "graphician": {
			const on = value === "true";
			ctx.bridge.updateSettings({ graphicianEnabled: on });
			saveConfigField("graphicianEnabled", on);
			ctx.statusPanel.update({ graphicianEnabled: on });
			ctx.notify(`Graphician: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "fffgrep": {
			const on = value === "true";
			ctx.bridge.updateSettings({ fffgrepEnabled: on });
			saveConfigField("fffgrepEnabled", on);
			ctx.statusPanel.update({ fffgrepEnabled: on });
			ctx.notify(`fffgrep: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "inference mode": {
			const valid = getEnumValues("inferenceMode") ?? [];
			if (!valid.includes(value)) {
				ctx.notify(
					`Invalid inference mode: ${value}. Valid: ${valid.join(", ")}`,
					"error",
				);
			} else {
				setInferenceMode(ctx, value, { persist: true });
			}
			break;
		}
		case "execution policy": {
			const normalized = value === "auto" ? "autonomous" : value;
			const valid: Array<"autonomous" | "minimal"> = ["autonomous", "minimal"];
			if (!valid.includes(normalized as (typeof valid)[number])) {
				ctx.notify(
					`Invalid execution policy: ${value}. Valid: ${valid.join(", ")}`,
					"error",
				);
			} else {
				setExecutionProfile(ctx, normalized as "autonomous" | "minimal");
				ctx.notify(
					`Execution mode: ${normalized === "autonomous" ? "auto" : "minimal"}`,
					"success",
				);
			}
			break;
		}
		case "duplicate-call guard":
		case "failure-loop guard":
		case "continuation":
		case "auto-compact on full context":
		case "budget early-stop":
		case "progress early-stop": {
			const on = value === "true";
			const runtimeKeys = {
				"duplicate-call guard": "duplicateGuardEnabled",
				"failure-loop guard": "failureGuardEnabled",
				continuation: "continuationEnabled",
				"auto-compact on full context": "autoRetryEnabled",
				"progress early-stop": "progressStopEnabled",
				"budget early-stop": "progressStopEnabled",
			} as const;
			const configKey =
				runtimeKeys[settingName.toLowerCase() as keyof typeof runtimeKeys];
			ctx.bridge.updateSettings({ [configKey]: on });
			saveConfigField(configKey, on);
			ctx.notify(`${settingName}: ${on ? "on" : "off"}`, "success");
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
