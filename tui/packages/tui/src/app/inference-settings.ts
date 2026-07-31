// ── Inference mode & execution profile helpers ──────────────────────────────
// Mixes bridge control + status display + disk config persistence for the
// inference-mode and execution-profile settings.

import { AgentCoreBridge } from "@logician/coding-agent/application";
import { saveConfigField } from "@logician/coding-agent/configuration";
import { StatusBar } from "../status/status-bar.ts";
import { TUI } from "../terminal/core.ts";

export type InferenceMode =
	| "thinking-general"
	| "thinking-coding"
	| "instruct-general"
	| "instruct-reasoning";

export interface InferenceSettingsCtx {
	bridge: AgentCoreBridge;
	statusPanel: StatusBar;
	tui: TUI;
	inferenceMode: InferenceMode;
	thinkingLevel: string;
	notify: (message: string, level?: "info" | "success" | "warning" | "error") => void;
}

// Inference mode helper — used by the keyboard shortcut and /settings.
export function setInferenceMode(ctx: InferenceSettingsCtx, mode: string): void {
	const valid = [
		"thinking-general",
		"thinking-coding",
		"instruct-general",
		"instruct-reasoning",
	];
	if (!valid.includes(mode)) return;
	const oldMode = ctx.inferenceMode;
	ctx.inferenceMode = mode as InferenceMode;
	ctx.bridge.setInferenceMode(mode);
	ctx.statusPanel.update({ inferenceMode: mode });
	if (oldMode !== mode) {
		const labels: Record<string, string> = {
			"thinking-general": "Thinking (General)",
			"thinking-coding": "Thinking (Precise Code)",
			"instruct-general": "Instruct (General)",
			"instruct-reasoning": "Instruct (Reasoning)",
		};
		ctx.notify(`Inference mode: ${labels[mode] ?? mode}`, "success");
		saveConfigField("inferenceMode", mode);
	}
}

export function cycleInferenceMode(ctx: InferenceSettingsCtx): void {
	const modes: InferenceMode[] = [
		"thinking-general",
		"thinking-coding",
		"instruct-general",
		"instruct-reasoning",
	];
	const currentIndex = modes.indexOf(ctx.inferenceMode);
	setInferenceMode(ctx, modes[(currentIndex + 1) % modes.length]);
	ctx.tui.requestRender();
}

export function applyThinkingLevel(ctx: InferenceSettingsCtx, level: string): void {
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
