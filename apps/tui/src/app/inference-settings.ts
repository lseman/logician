// ── Inference mode & execution profile helpers ──────────────────────────────
// Mixes bridge control + status display + disk config persistence for the
// inference-mode and execution-profile settings.

import {
	INFERENCE_MODE_ORDER,
	type InferenceMode,
} from "@logician/agent-core";
import type { AgentCoreBridge } from "@logician/coding-agent/application";
import { saveConfigField } from "@logician/coding-agent/configuration";
import type { StatusBar } from "../footer/layout.ts";
import type { TuiHandle } from "../terminal/core.ts";

export { INFERENCE_MODE_ORDER, type InferenceMode };

export interface InferenceSettingsCtx {
	bridge: AgentCoreBridge;
	statusPanel: StatusBar;
	tui: TuiHandle;
	inferenceMode: InferenceMode;
	thinkingLevel: string;
	notify: (
		message: string,
		level?: "info" | "success" | "warning" | "error",
	) => void;
}

// Inference mode helper — used by the keyboard shortcut and /settings.
export function setInferenceMode(
	ctx: InferenceSettingsCtx,
	mode: string,
	options: { persist?: boolean; notify?: boolean } = {},
): void {
	if (!INFERENCE_MODE_ORDER.includes(mode as InferenceMode)) return;
	const oldMode = ctx.inferenceMode;
	ctx.inferenceMode = mode as InferenceMode;
	ctx.bridge.setInferenceMode(mode);
	ctx.statusPanel.update({ inferenceMode: mode });
	if (oldMode !== mode && options.notify !== false) {
		const labels: Record<string, string> = {
			auto: "Auto",
			none: "Provider",
			"thinking-general": "Thinking (General)",
			"thinking-coding": "Thinking (Precise Code)",
			"instruct-general": "Instruct (General)",
			"instruct-reasoning": "Instruct (Reasoning)",
			"instruct-coding": "Instruct (Code)",
			deterministic: "Exact (Deterministic)",
			creative: "Creative",
			analytical: "Analytical",
		};
		ctx.notify(`Inference mode: ${labels[mode] ?? mode}`, "success");
	}
	if (options.persist === true && oldMode !== mode)
		saveConfigField("inferenceMode", mode);
}

export function cycleInferenceMode(ctx: InferenceSettingsCtx): void {
	const currentIndex = INFERENCE_MODE_ORDER.indexOf(ctx.inferenceMode);
	setInferenceMode(
		ctx,
		INFERENCE_MODE_ORDER[(currentIndex + 1) % INFERENCE_MODE_ORDER.length],
		{ persist: true },
	);
	ctx.tui.requestRender();
}

export function applyThinkingLevel(
	ctx: InferenceSettingsCtx,
	level: string,
	options: { persist?: boolean } = {},
): void {
	ctx.thinkingLevel = level;
	ctx.bridge.setThinkingLevel(level);
	ctx.statusPanel.update({ thinkingLevel: level });
	if (options.persist === true) saveConfigField("thinkingLevel", level);
}

export function setExecutionProfile(
	ctx: InferenceSettingsCtx,
	profile: "autonomous" | "minimal",
): void {
	ctx.bridge.setExecutionProfile(profile);
	ctx.statusPanel.update({ executionProfile: profile });
	saveConfigField("executionProfile", profile);
}

export function cycleExecutionProfile(
	ctx: InferenceSettingsCtx,
): "autonomous" | "minimal" {
	const current = ctx.bridge.getSettingsData().executionProfile;
	const next = current === "autonomous" ? "minimal" : "autonomous";
	setExecutionProfile(ctx, next);
	return next;
}
