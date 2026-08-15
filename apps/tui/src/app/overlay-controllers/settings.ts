// ── Settings selector controller ───────────────────────────────────────────

import {
	saveConfigField,
	saveConfigNestedField,
} from "@logician/coding-agent/configuration";
import type {
	SettingDef,
	SettingsSelectorAction,
} from "../../overlays/settings-overlay.ts";
import {
	applyThinkingLevel,
	setExecutionProfile,
	setInferenceMode,
} from "../inference-settings.ts";
import type { OverlayHandlersCtx } from "./context.ts";
import { openModelSelector } from "./selectors.ts";

// ── Settings selector ───────────────────────────────────────────────────

export async function openSettingsSelector(
	ctx: OverlayHandlersCtx,
): Promise<void> {
	try {
		const data = ctx.bridge.getSettingsData();
		const thinkingLevels = ["off", "minimal", "low", "medium", "high", "xhigh"];
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
				options: [0.0, 0.3, 0.5, 0.7, 1.0].map(v => ({
					label: String(v),
					value: String(v),
					current: Math.abs(data.temperature - v) < 0.001,
				})),
			},
			{
				name: "Max tokens",
				currentValue: String(data.maxTokens),
				description: "Maximum response tokens",
				options: [1024, 2048, 4096, 8192, 16384].map(v => ({
					label: String(v),
					value: String(v),
					current: data.maxTokens === v,
				})),
			},
			{
				name: "Max iterations",
				currentValue: String(data.maxIterations),
				description: "Maximum tool-use iterations per turn",
				options: [10, 20, 30, 50, 100].map(v => ({
					label: String(v),
					value: String(v),
					current: data.maxIterations === v,
				})),
			},
			{
				name: "Thinking level",
				currentValue: data.thinkingLevel,
				description: "Depth of reasoning before responding",
				options: thinkingLevels.map(v => ({
					label: v.charAt(0).toUpperCase() + v.slice(1),
					value: v,
					current: data.thinkingLevel === v,
				})),
			},
			{
				name: "Permission mode",
				currentValue: data.permissionMode,
				description: "How the agent handles tool permissions",
				options: permissionModes.map(v => ({
					label: v,
					value: v,
					current: data.permissionMode === v,
				})),
			},
			{
				name: "Guards",
				currentValue: data.guardMode,
				description: "Loop guards: auto uses safe defaults; off disables all",
				options: [
					{
						label: "Auto",
						value: "auto",
						current: data.guardMode === "auto",
					},
					{
						label: "on",
						value: "on",
						current: data.guardMode === "on",
						toggleOn: true,
					},
					{
						label: "off",
						value: "off",
						current: data.guardMode === "off",
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
						label: "Auto",
						value: "auto",
						current: data.inferenceMode === "auto",
					},
					{
						label: "Provider defaults",
						value: "none",
						current: data.inferenceMode === "none",
					},
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
					{
						label: "Code",
						value: "instruct-coding",
						current: data.inferenceMode === "instruct-coding",
					},
					{
						label: "Exact",
						value: "deterministic",
						current: data.inferenceMode === "deterministic",
					},
					{
						label: "Creative",
						value: "creative",
						current: data.inferenceMode === "creative",
					},
					{
						label: "Analyze",
						value: "analytical",
						current: data.inferenceMode === "analytical",
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
				name: "Ariadne",
				currentValue: data.ariadneEnabled ? "on" : "off",
				description:
					"Expose the Ariadne code-graph tool for semantic repository analysis",
				options: [
					{
						label: "on",
						value: "true",
						current: data.ariadneEnabled,
						toggleOn: true,
					},
					{
						label: "off",
						value: "false",
						current: !data.ariadneEnabled,
						toggleOn: false,
					},
				],
			},
			{
				name: "fffgrep",
				currentValue: data.fffgrepEnabled ? "on" : "off",
				description: "Prefer the fff indexed MCP grep tool over local grep",
				options: [
					{
						label: "on",
						value: "true",
						current: data.fffgrepEnabled,
						toggleOn: true,
					},
					{
						label: "off",
						value: "false",
						current: !data.fffgrepEnabled,
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
			...[
				[
					"Duplicate-call guard",
					data.duplicateGuardEnabled,
					"Block repeated identical tool calls",
				],
				[
					"Failure-loop guard",
					data.failureGuardEnabled,
					"Block repeated equivalent tool failures",
				],
				[
					"Thinking-loop guard",
					data.thinkingLoopDetectionEnabled,
					"Detect reasoning loops without action",
				],
				[
					"Continuation",
					data.continuationEnabled,
					"Continue bounded unfinished autonomous work",
				],
				[
					"Automatic retries",
					data.autoRetryEnabled,
					"Retry transient provider failures",
				],
				[
					"Reflection",
					data.reflectionEnabled,
					"Run bounded self-review before completion",
				],
				[
					"Budget early-stop",
					data.budgetStopEnabled,
					"Stop when useful token growth flattens",
				],
				[
					"Memory",
					data.memoryEnabled,
					"Persist and retrieve cross-session memories",
				],
			].map(([name, enabled, description]) => ({
				name: String(name),
				currentValue: enabled ? "on" : "off",
				description: String(description),
				options: [
					{
						label: "on",
						value: "true",
						current: Boolean(enabled),
						toggleOn: true,
					},
					{ label: "off", value: "false", current: !enabled, toggleOn: false },
				],
			})),
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
	if (action.type === "open" && action.settingName.toLowerCase() === "model") {
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
				ctx.bridge.setMaxTokens(num);
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
				ctx.bridge.setMaxIterations(num);
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
		case "permission mode":
			ctx.bridge.setPermissionMode(
				value as "acceptAll" | "acceptEdits" | "ask" | "plan",
			);
			saveConfigField("permissionMode", value);
			ctx.notify(`Permission mode: ${value}`, "success");
			break;
		case "guards": {
			const mode = value as "auto" | "on" | "off";
			ctx.bridge.setGuardMode(mode);
			saveConfigField(
				"guardsEnabled",
				mode === "auto" ? undefined : mode === "on",
			);
			ctx.notify(`Guards: ${mode}`, "success");
			break;
		}
		case "compaction": {
			const on = value === "true";
			ctx.bridge.setRuntimeToggle("proactiveCompactionEnabled", on);
			saveConfigNestedField("compaction", "enabled", on);
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
		case "ariadne": {
			const on = value === "true";
			ctx.bridge.setRuntimeToggle("ariadneEnabled", on);
			saveConfigField("ariadneEnabled", on);
			ctx.statusPanel.update({ ariadneEnabled: on });
			ctx.notify(`Ariadne: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "fffgrep": {
			const on = value === "true";
			ctx.bridge.setRuntimeToggle("fffgrepEnabled", on);
			saveConfigField("fffgrepEnabled", on);
			ctx.statusPanel.update({ fffgrepEnabled: on });
			ctx.notify(`fffgrep: ${on ? "on" : "off"}`, "success");
			break;
		}
		case "inference mode": {
			const valid = [
				"auto",
				"none",
				"thinking-general",
				"thinking-coding",
				"instruct-general",
				"instruct-reasoning",
				"instruct-coding",
				"deterministic",
				"creative",
				"analytical",
			];
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
			const valid: Array<"autonomous" | "minimal"> = ["autonomous", "minimal"];
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
		case "duplicate-call guard":
		case "failure-loop guard":
		case "thinking-loop guard":
		case "continuation":
		case "automatic retries":
		case "reflection":
		case "budget early-stop":
		case "memory": {
			const on = value === "true";
			const keys = {
				"duplicate-call guard": ["duplicateGuardEnabled", "guardrails"],
				"failure-loop guard": ["failureGuardEnabled", "guardrails"],
				"thinking-loop guard": ["thinkingLoopDetectionEnabled", "guardrails"],
				continuation: ["continuationEnabled", "continuationEnabled"],
				"automatic retries": ["autoRetryEnabled", "autoRetryEnabled"],
				reflection: ["reflectionEnabled", "reflectionConfig"],
				"budget early-stop": ["budgetStopEnabled", "guardrails"],
				memory: ["memoryEnabled", "memory"],
			} as const;
			const [runtimeKey, configKey] =
				keys[settingName.toLowerCase() as keyof typeof keys];
			ctx.bridge.setRuntimeToggle(runtimeKey, on);
			if (configKey === "reflectionConfig")
				saveConfigNestedField("reflectionConfig", "enabled", on);
			else if (configKey === "guardrails")
				saveConfigNestedField("guardrails", runtimeKey, on);
			else saveConfigField(configKey, on);
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
