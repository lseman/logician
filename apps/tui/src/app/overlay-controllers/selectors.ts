// ── Selector and file-mention controllers ──────────────────────────────────

import {
	getReasonerIds,
	getReasonerMeta,
	type ReasonerMeta,
} from "@logician/log-runtime/reasoning";
import type { ThinkingLevel } from "@logician/log-core";
import { saveConfigField } from "@logician/log-runtime/configuration";
import { listProjectFiles } from "@logician/log-runtime/context";
import type {
	InferenceModeInfo,
	InferenceModeSelectorAction,
} from "../../overlays/inference-mode-selector.ts";
import type {
	ModelInfo,
	ModelSelectorAction,
} from "../../overlays/model-selector.ts";
import type { QueueManagerAction } from "../../overlays/queue-manager.ts";
import type {
	ReasonerInfo,
	ReasonerSelectorAction,
} from "../../overlays/reasoner-selector.ts";
import type {
	ThemeInfo,
	ThemeSelectorAction,
} from "../../overlays/theme-selector.ts";
import type {
	ThinkingLevelInfo,
	ThinkingLevelSelectorAction,
} from "../../overlays/thinking-level-selector.ts";
import {
	getAvailableThemes,
	getCurrentThemeName,
	setTheme,
} from "../../terminal/theme.ts";
import type { InferenceMode } from "../inference-settings.ts";
import type { OverlayHandlersCtx } from "./context.ts";

// ── Reasoner selector ───────────────────────────────────────────────────

export async function openReasonerSelector(
	ctx: OverlayHandlersCtx,
): Promise<void> {
	ctx.statusPanel.update({ phase: "reasoner" });
	const currentId = ctx.bridge.getReasonerStatus();
	const reasoners: ReasonerInfo[] = getReasonerIds().map(id => {
		const meta = getReasonerMeta(id) as ReasonerMeta;
		return {
			id,
			name: meta.name,
			description: meta.description,
			active: id === currentId,
		};
	});
	ctx.reasonerSelector.setReasoners(reasoners);
	ctx.reasonerSelector.setMessage(
		"Enter selects reasoning mode for the next turn.",
	);
	ctx.reasonerSelector.show();
	const overlay = ctx.tui.showOverlay(ctx.reasonerSelector, {
		anchor: "aboveInput",
		align: "left",
		maxHeight: 18,
	});
	overlay.focus();
}

export function handleReasonerSelectorAction(
	ctx: OverlayHandlersCtx,
	action: ReasonerSelectorAction,
): void {
	if (action.type === "close") {
		ctx.tui.removeOverlay(ctx.reasonerSelector);
		ctx.statusPanel.update({ phase: "ready" });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		return;
	}
	const reasoner = action.reasoner;
	ctx.reasonerSelector.setMessage(`Setting: ${reasoner.name}...`);
	ctx.tui.requestRender();
	ctx.bridge.updateSettings({ reasonerId: reasoner.id });
	saveConfigField("reasoner", reasoner.id);
	ctx.tui.removeOverlay(ctx.reasonerSelector);
	ctx.statusPanel.update({ phase: "ready" });
	ctx.statusPanel.update({ reasoner: reasoner.id });
	ctx.notify(`Reasoning mode: ${reasoner.name}`, "success");
	ctx.tui.requestRender();
}

// ── File mention autocomplete ────────────────────────────────────────

export async function updateFileMentionPopup(
	ctx: OverlayHandlersCtx,
	query: string,
): Promise<void> {
	const cwd = process.cwd();
	if (ctx.fileMentionListedCwd !== cwd || !ctx.fileMentionListing) {
		ctx.fileMentionListedCwd = cwd;
		ctx.fileMentionListing = listProjectFiles(cwd);
	}
	const files = await ctx.fileMentionListing;

	// The user may have kept typing (or dismissed the mention) while the
	// listing was in flight; only apply this result if still relevant.
	if (ctx.inputBar.getActiveMentionQuery() !== query) return;

	ctx.fileMentionPopup.setFiles(files);
	ctx.fileMentionPopup.setQuery(query);
	if (ctx.fileMentionPopup.hasMatches()) {
		if (!ctx.fileMentionPopup.isVisibleOverlay()) ctx.fileMentionPopup.show();
	} else {
		ctx.fileMentionPopup.hide();
	}
	ctx.tui.requestRender();
}

// ── Model selector ───────────────────────────────────────────────────

export function openModelSelector(ctx: OverlayHandlersCtx): void {
	ctx.statusPanel.update({ phase: "model" });
	const modelInfos: ModelInfo[] = ctx.bridge.models.options().map(option => ({
		id: option.key,
		name: option.name,
		active: option.active,
		url: `${option.model} · ${option.url}`,
	}));
	ctx.modelSelector.setModels(modelInfos);
	ctx.modelSelector.setMessage("Enter selects model for the current session.");
	ctx.modelSelector.show();
	const overlay = ctx.tui.showOverlay(ctx.modelSelector, {
		anchor: "aboveInput",
		align: "left",
		maxHeight: 18,
	});
	overlay.focus();
}

export function handleModelSelectorAction(
	ctx: OverlayHandlersCtx,
	action: ModelSelectorAction,
): void {
	if (action.type === "close") {
		ctx.tui.removeOverlay(ctx.modelSelector);
		ctx.statusPanel.update({ phase: "ready" });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		return;
	}
	const selected = action.model;
	ctx.modelSelector.setMessage(`Switching to ${selected.name}...`);
	ctx.tui.requestRender();
	// Switch the model via the bridge (handles url switching too)
	const applied = ctx.bridge.models.selectOption(selected.id);
	if (!applied) return;
	// Save to global settings
	saveConfigField("model", applied.model);
	saveConfigField("baseUrl", applied.url);
	// Update status
	ctx.tui.removeOverlay(ctx.modelSelector);
	ctx.statusPanel.update({ phase: "ready", model: applied.model });
	ctx.notify(`Model: ${selected.name}`, "success");
	ctx.tui.requestRender();
}

// ── Theme selector ───────────────────────────────────────────────────

export async function openThemeSelector(
	ctx: OverlayHandlersCtx,
): Promise<void> {
	const available = getAvailableThemes();
	const current = getCurrentThemeName();
	const themes: ThemeInfo[] = available.map(name => ({
		name,
		description: `${name.charAt(0).toUpperCase() + name.slice(1)} theme`,
		active: name === current,
	}));
	ctx.themeSelector.setThemes(themes);
	ctx.themeSelector.setMessage("Enter selects a color theme.");
	ctx.themeSelector.show();
	const overlay = ctx.tui.showOverlay(ctx.themeSelector, {
		anchor: "aboveInput",
		align: "left",
		maxHeight: 18,
	});
	overlay.focus();
}

export function handleThemeSelectorAction(
	ctx: OverlayHandlersCtx,
	action: ThemeSelectorAction,
): void {
	if (action.type === "close") {
		ctx.tui.removeOverlay(ctx.themeSelector);
		ctx.statusPanel.update({ phase: "ready" });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		return;
	}
	const themeInfo = action.theme;
	ctx.themeSelector.setMessage(`Setting: ${themeInfo.name}...`);
	ctx.tui.requestRender();
	const ok = setThemeByName(themeInfo.name);
	ctx.tui.removeOverlay(ctx.themeSelector);
	ctx.statusPanel.update({ phase: "ready" });
	if (ok) {
		ctx.notify(`Theme: ${themeInfo.name}`, "success");
	} else {
		ctx.notify(`Unknown theme: ${themeInfo.name}`, "error");
	}
	ctx.tui.requestRender();
}

export function setThemeByName(name: string): boolean {
	const available = getAvailableThemes();
	if (!available.includes(name)) return false;
	setTheme(name);
	saveConfigField("theme", name);
	return true;
}

// ── Inference mode selector ──────────────────────────────────────────

export function openInferenceModeSelector(ctx: OverlayHandlersCtx): void {
	const inferenceModes: InferenceModeInfo[] = [
		{
			id: "auto",
			label: "Auto",
			description: "Auto-select from task phase",
			thinking: true,
			useProviderDefaults: false,
			params: {
				temperature: 0.7,
				top_p: 0.8,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 1.0,
				repetition_penalty: 1.0,
			},
		},
		{
			id: "none",
			label: "Provider",
			description: "Let the provider use its own defaults",
			thinking: false,
			useProviderDefaults: true,
			params: {
				temperature: 0.7,
				top_p: 0.8,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 0.0,
				repetition_penalty: 1.0,
			},
		},
		{
			id: "thinking-general",
			label: "Think Gen",
			description: "General thinking — high creativity",
			thinking: true,
			useProviderDefaults: false,
			params: {
				temperature: 1.0,
				top_p: 0.95,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 1.5,
				repetition_penalty: 1.0,
			},
		},
		{
			id: "thinking-coding",
			label: "Think Code",
			description: "Precise coding — lower temp",
			thinking: true,
			useProviderDefaults: false,
			params: {
				temperature: 0.6,
				top_p: 0.95,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 0.0,
				repetition_penalty: 1.0,
			},
		},
		{
			id: "instruct-general",
			label: "Instruct",
			description: "Non-thinking — balanced",
			thinking: false,
			useProviderDefaults: false,
			params: {
				temperature: 0.7,
				top_p: 0.8,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 1.5,
				repetition_penalty: 1.0,
			},
		},
		{
			id: "instruct-reasoning",
			label: "Reason",
			description: "Non-thinking — high temp",
			thinking: false,
			useProviderDefaults: false,
			params: {
				temperature: 1.0,
				top_p: 0.95,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 1.5,
				repetition_penalty: 1.0,
			},
		},
		{
			id: "instruct-coding",
			label: "Code",
			description: "Non-thinking — precise output",
			thinking: false,
			useProviderDefaults: false,
			params: {
				temperature: 0.3,
				top_p: 0.9,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 0.0,
				repetition_penalty: 1.0,
			},
		},
		{
			id: "deterministic",
			label: "Exact",
			description: "Near-zero temp — reproducible",
			thinking: false,
			useProviderDefaults: false,
			params: {
				temperature: 0.0,
				top_p: 0.0,
				top_k: 1,
				min_p: 0.0,
				presence_penalty: 0.0,
				repetition_penalty: 1.0,
			},
		},
		{
			id: "creative",
			label: "Creative",
			description: "Ultra-high temp — brainstorm",
			thinking: false,
			useProviderDefaults: false,
			params: {
				temperature: 1.3,
				top_p: 0.99,
				top_k: 40,
				min_p: 0.0,
				presence_penalty: 2.0,
				repetition_penalty: 0.9,
			},
		},
		{
			id: "analytical",
			label: "Analyze",
			description: "Low temp — code review",
			thinking: false,
			useProviderDefaults: false,
			params: {
				temperature: 0.2,
				top_p: 0.7,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 0.5,
				repetition_penalty: 1.1,
			},
		},
	];

	ctx.inferenceModeSelector.setModes(inferenceModes, ctx.inferenceMode);
	ctx.inferenceModeSelector.setMessage(
		"Enter selects inference mode for this session.",
	);
	ctx.inferenceModeSelector.show();
	const overlay = ctx.tui.showOverlay(ctx.inferenceModeSelector, {
		anchor: "aboveInput",
		align: "left",
		maxHeight: 18,
	});
	overlay.focus();
}

// ── Thinking level selector ──────────────────────────────────────────

export function openThinkingLevelSelector(ctx: OverlayHandlersCtx): void {
	const thinkingLevels: ThinkingLevelInfo[] = [
		{
			id: "off",
			label: "Off",
			description: "No reasoning",
			active: ctx.thinkingLevel === "off",
		},
		{
			id: "minimal",
			label: "Minimal",
			description: "Very brief reasoning (~1k tokens)",
			active: ctx.thinkingLevel === "minimal",
		},
		{
			id: "low",
			label: "Low",
			description: "Light reasoning (~2k tokens)",
			active: ctx.thinkingLevel === "low",
		},
		{
			id: "medium",
			label: "Medium",
			description: "Moderate reasoning (~8k tokens)",
			active: ctx.thinkingLevel === "medium",
		},
		{
			id: "high",
			label: "High",
			description: "Deep reasoning (~16k tokens)",
			active: ctx.thinkingLevel === "high",
		},
		{
			id: "xhigh",
			label: "X-High",
			description: "Extra-high reasoning (~32k tokens)",
			active: ctx.thinkingLevel === "xhigh",
		},
	];

	ctx.thinkingLevelSelector.setLevels(thinkingLevels);
	ctx.thinkingLevelSelector.setMessage(
		"Enter selects thinking level for the next turn.",
	);
	ctx.thinkingLevelSelector.show();
	const overlay = ctx.tui.showOverlay(ctx.thinkingLevelSelector, {
		anchor: "aboveInput",
		align: "left",
		maxHeight: 18,
	});
	overlay.focus();
}

export function handleThinkingLevelSelectorAction(
	ctx: OverlayHandlersCtx,
	action: ThinkingLevelSelectorAction,
): void {
	if (action.type === "close") {
		ctx.tui.removeOverlay(ctx.thinkingLevelSelector);
		ctx.statusPanel.update({ phase: "ready" });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		return;
	}
	const selected = action.level;
	ctx.thinkingLevelSelector.setMessage(`Setting: ${selected.label}...`);
	ctx.tui.requestRender();
	ctx.bridge.updateSettings({
		thinkingLevel: selected.id as ThinkingLevel,
	});
	saveConfigField("thinkingLevel", selected.id);
	ctx.thinkingLevel = selected.id as ThinkingLevel;
	ctx.statusPanel.update({ phase: "ready", thinkingLevel: selected.id });
	ctx.tui.removeOverlay(ctx.thinkingLevelSelector);
	ctx.notify(`Thinking level: ${selected.label}`, "success");
	ctx.tui.requestRender();
}

export function handleInferenceModeSelectorAction(
	ctx: OverlayHandlersCtx,
	action: InferenceModeSelectorAction,
): void {
	if (action.type === "close") {
		ctx.tui.removeOverlay(ctx.inferenceModeSelector);
		return;
	}
	const selected = action.mode;
	ctx.inferenceModeSelector.setMessage(`Setting: ${selected.label}...`);
	ctx.tui.requestRender();
	ctx.bridge.updateSettings({
		inferenceMode: selected.id as InferenceMode,
	});
	saveConfigField("inferenceMode", selected.id);
	ctx.tui.removeOverlay(ctx.inferenceModeSelector);
	ctx.statusPanel.update({ inferenceMode: selected.id });
	ctx.inferenceMode = selected.id as InferenceMode;
	ctx.notify(`Inference mode: ${selected.label}`, "success");
	ctx.tui.requestRender();
}

// ── Queue manager ──────────────────────────────────────────────────────

export function openQueueManager(ctx: OverlayHandlersCtx): void {
	ctx.queueManager.setQueues(
		ctx.bridge.getSteeringMessages(),
		ctx.bridge.getFollowUpMessages(),
		ctx.bridge.getNextTurnMessages(),
	);
	ctx.queueManager.setMessage("");
	ctx.queueManager.show();
	const overlay = ctx.tui.showOverlay(ctx.queueManager, {
		anchor: "aboveInput",
		align: "left",
		maxHeight: 18,
	});
	overlay.focus();
}

export function handleQueueManagerAction(
	ctx: OverlayHandlersCtx,
	action: QueueManagerAction,
): void {
	if (action.type === "close") {
		ctx.tui.removeOverlay(ctx.queueManager);
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		return;
	}
	if (action.type === "clear") {
		ctx.bridge.clearQueue();
		ctx.notify("Queue cleared.", "info");
	} else {
		const removed =
			action.entry.dropIndex !== undefined
				? ctx.bridge.dropQueuedMessage(action.entry.dropIndex)
				: undefined;
		if (removed === undefined) {
			ctx.queueManager.setMessage("Can't remove a next-turn message directly.");
			ctx.tui.requestRender();
			return;
		}
		ctx.notify("Removed from queue.", "info");
	}
	ctx.queueManager.setQueues(
		ctx.bridge.getSteeringMessages(),
		ctx.bridge.getFollowUpMessages(),
		ctx.bridge.getNextTurnMessages(),
	);
	ctx.tui.requestRender();
}
