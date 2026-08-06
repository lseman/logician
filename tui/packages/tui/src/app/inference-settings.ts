// ── Inference mode & execution profile helpers ──────────────────────────────
// Mixes bridge control + status display + disk config persistence for the
// inference-mode and execution-profile settings.

import type { AgentCoreBridge } from "@logician/coding-agent/application";
import { saveConfigField } from "@logician/coding-agent/configuration";
import type { StatusBar } from "../status/status-bar.ts";
import type { TUI } from "../terminal/core.ts";

export type InferenceMode =
	| "auto"
	| "thinking-general"
	| "thinking-coding"
	| "instruct-general"
	| "instruct-reasoning"
	| "instruct-coding"
	| "deterministic"
	| "creative"
	| "analytical";

export const INFERENCE_MODE_ORDER: readonly InferenceMode[] = [
	"auto",
	"thinking-general",
	"thinking-coding",
	"instruct-general",
	"instruct-reasoning",
	"instruct-coding",
	"deterministic",
	"creative",
	"analytical",
];

export interface InferenceSettingsCtx {
	bridge: AgentCoreBridge;
	statusPanel: StatusBar;
	tui: TUI;
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
): void {
	if (!INFERENCE_MODE_ORDER.includes(mode as InferenceMode)) return;
	const oldMode = ctx.inferenceMode;
	ctx.inferenceMode = mode as InferenceMode;
	ctx.bridge.setInferenceMode(mode);
	ctx.statusPanel.update({ inferenceMode: mode });
	if (oldMode !== mode) {
		const labels: Record<string, string> = {
			auto: "Auto",
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
		saveConfigField("inferenceMode", mode);
	}
}

export function cycleInferenceMode(ctx: InferenceSettingsCtx): void {
	const currentIndex = INFERENCE_MODE_ORDER.indexOf(ctx.inferenceMode);
	setInferenceMode(
		ctx,
		INFERENCE_MODE_ORDER[(currentIndex + 1) % INFERENCE_MODE_ORDER.length],
	);
	ctx.tui.requestRender();
}

export function applyThinkingLevel(
	ctx: InferenceSettingsCtx,
	level: string,
): void {
	ctx.thinkingLevel = level;
	ctx.bridge.setThinkingLevel(level);
	ctx.statusPanel.update({ thinkingLevel: level });
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
