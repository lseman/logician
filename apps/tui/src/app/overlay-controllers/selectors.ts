// ── Selector and file-mention controllers ──────────────────────────────────

import type { ThinkingLevel } from "@logician/log-core";
import { saveConfigField } from "@logician/log-runtime/configuration";
import { listProjectFiles } from "@logician/log-runtime/context";
import {
	getReasonerIds,
	getReasonerMeta,
	type ReasonerMeta,
} from "@logician/log-runtime/reasoning";
import type { InferenceModeSelectorAction } from "../../overlays/inference-mode-selector.ts";
import { INFERENCE_MODES } from "../../overlays/inference-mode-selector.ts";
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

// ── Generic list-selector open helper ──────────────────────────────────────
// Shared by all list-selector opens: get data, set items, show overlay.

function openListSelector<T>(
	ctx: OverlayHandlersCtx,
	items: T[],
	overlay: {
		setItems(items: T[], preferredIndex?: number): void;
		setMessage(msg: string): void;
		show(): void;
		render(width: number): string[];
	},
	statusPhase: string,
	message: string,
): void {
	ctx.statusPanel.update({ phase: statusPhase });
	overlay.setItems(items);
	overlay.setMessage(message);
	overlay.show();
	const o = ctx.tui.showOverlay(overlay, {
		anchor: "aboveInput",
		align: "left",
		maxHeight: 18,
	});
	o.focus();
}

// ── Generic list-selector close helper ─────────────────────────────────────
// Shared by all close handlers: remove overlay, reset status, refresh transcript.

function closeListSelector(
	ctx: OverlayHandlersCtx,
	overlay: { hide(): void; render(width: number): string[] },
): void {
	ctx.tui.removeOverlay(overlay);
	ctx.statusPanel.update({ phase: "ready" });
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
}

// ── Reasoner selector ───────────────────────────────────────────────────

export function openReasonerSelector(ctx: OverlayHandlersCtx): void {
	const currentId = ctx.bridge.getReasonerStatus();
	const reasoners: ReasonerInfo[] = getReasonerIds().map(id => ({
		id,
		name: (getReasonerMeta(id) as ReasonerMeta).name,
		description: (getReasonerMeta(id) as ReasonerMeta).description,
		active: id === currentId,
	}));
	openListSelector(
		ctx,
		reasoners,
		ctx.reasonerSelector,
		"reasoner",
		"Enter selects reasoning mode for the next turn.",
	);
}

export function handleReasonerSelectorAction(
	ctx: OverlayHandlersCtx,
	action: ReasonerSelectorAction,
): void {
	if (action.type === "close") {
		closeListSelector(ctx, ctx.reasonerSelector);
		return;
	}
	const selected = action.item;
	ctx.reasonerSelector.setMessage(`Setting: ${selected.name}...`);
	ctx.tui.requestRender();
	ctx.bridge.updateSettings({ reasonerId: selected.id });
	saveConfigField("reasoner", selected.id);
	closeListSelector(ctx, ctx.reasonerSelector);
	ctx.statusPanel.update({ reasoner: selected.id });
	ctx.notify(`Reasoning mode: ${selected.name}`, "success");
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
	const modelInfos: ModelInfo[] = ctx.bridge.models.options().map(option => ({
		id: option.key,
		name: option.name,
		active: option.active,
		url: `${option.model} · ${option.url}`,
	}));
	openListSelector(
		ctx,
		modelInfos,
		ctx.modelSelector,
		"model",
		"Enter selects model for the current session.",
	);
}

export function handleModelSelectorAction(
	ctx: OverlayHandlersCtx,
	action: ModelSelectorAction,
): void {
	if (action.type === "close") {
		closeListSelector(ctx, ctx.modelSelector);
		return;
	}
	const selected = action.item;
	ctx.modelSelector.setMessage(`Switching to ${selected.name}...`);
	ctx.tui.requestRender();
	const applied = ctx.bridge.models.selectOption(selected.id);
	if (!applied) return;
	saveConfigField("model", applied.model);
	saveConfigField("baseUrl", applied.url);
	closeListSelector(ctx, ctx.modelSelector);
	ctx.statusPanel.update({ phase: "ready", model: applied.model });
	ctx.notify(`Model: ${selected.name}`, "success");
	ctx.tui.requestRender();
}

// ── Theme selector ───────────────────────────────────────────────────

export function openThemeSelector(ctx: OverlayHandlersCtx): void {
	const available = getAvailableThemes();
	const current = getCurrentThemeName();
	const themes: ThemeInfo[] = available.map(name => ({
		name,
		description: `${name.charAt(0).toUpperCase() + name.slice(1)} theme`,
		active: name === current,
	}));
	openListSelector(
		ctx,
		themes,
		ctx.themeSelector,
		"theme",
		"Enter selects a color theme.",
	);
}

export function handleThemeSelectorAction(
	ctx: OverlayHandlersCtx,
	action: ThemeSelectorAction,
): void {
	if (action.type === "close") {
		closeListSelector(ctx, ctx.themeSelector);
		return;
	}
	const selected = action.item;
	ctx.themeSelector.setMessage(`Setting: ${selected.name}...`);
	ctx.tui.requestRender();
	const ok = setThemeByName(selected.name);
	closeListSelector(ctx, ctx.themeSelector);
	ctx.statusPanel.update({ phase: "ready" });
	if (ok) {
		ctx.notify(`Theme: ${selected.name}`, "success");
	} else {
		ctx.notify(`Unknown theme: ${selected.name}`, "error");
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
	ctx.inferenceModeSelector.setItems(INFERENCE_MODES);
	ctx.inferenceModeSelector.activeId = ctx.inferenceMode;
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
	const levels: ThinkingLevelInfo[] = [
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
	openListSelector(
		ctx,
		levels,
		ctx.thinkingLevelSelector,
		"thinking",
		"Enter selects thinking level for the next turn.",
	);
}

export function handleThinkingLevelSelectorAction(
	ctx: OverlayHandlersCtx,
	action: ThinkingLevelSelectorAction,
): void {
	if (action.type === "close") {
		closeListSelector(ctx, ctx.thinkingLevelSelector);
		return;
	}
	const selected = action.item;
	ctx.thinkingLevelSelector.setMessage(`Setting: ${selected.label}...`);
	ctx.tui.requestRender();
	ctx.bridge.updateSettings({ thinkingLevel: selected.id as ThinkingLevel });
	saveConfigField("thinkingLevel", selected.id);
	ctx.thinkingLevel = selected.id as ThinkingLevel;
	closeListSelector(ctx, ctx.thinkingLevelSelector);
	ctx.statusPanel.update({ phase: "ready", thinkingLevel: selected.id });
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
	const selected = action.item;
	ctx.inferenceModeSelector.setMessage(`Setting: ${selected.label}...`);
	ctx.tui.requestRender();
	ctx.bridge.updateSettings({ inferenceMode: selected.id as InferenceMode });
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
