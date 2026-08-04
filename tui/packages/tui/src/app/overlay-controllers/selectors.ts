// ── Selector and file-mention controllers ──────────────────────────────────

import { getReasonerIds, getReasonerMeta, type ReasonerMeta } from "@logician/agent-capabilities/reasoning";
import { saveConfigField } from "@logician/coding-agent/configuration";
import { listProjectFiles } from "@logician/coding-agent/context";
import type { ModelInfo, ModelSelectorAction } from "../../overlays/model-selector.ts";
import type { ReasonerInfo, ReasonerSelectorAction } from "../../overlays/reasoner-selector.ts";
import type { ThemeInfo, ThemeSelectorAction } from "../../overlays/theme-selector.ts";
import { getAvailableThemes, setTheme } from "../../terminal/theme.ts";
import type { OverlayHandlersCtx } from "./context.ts";

// ── Reasoner selector ───────────────────────────────────────────────────

export async function openReasonerSelector(ctx: OverlayHandlersCtx): Promise<void> {
	ctx.statusPanel.update({ phase: "reasoner" });
	const currentId = ctx.bridge.getReasonerStatus();
	const reasoners: ReasonerInfo[] = getReasonerIds().map((id) => {
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
	ctx.bridge.setReasonerId(reasoner.id);
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
	const modelInfos: ModelInfo[] = ctx.bridge
		.getModelOptions()
		.map((option) => ({
			id: option.key,
			name: option.name,
			active: option.active,
			url: `${option.model} · ${option.url}`,
		}));
	ctx.modelSelector.setModels(modelInfos);
	ctx.modelSelector.setMessage(
		"Enter selects model for the current session.",
	);
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
	const applied = ctx.bridge.setModelOption(selected.id);
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

export async function openThemeSelector(ctx: OverlayHandlersCtx): Promise<void> {
	const available = getAvailableThemes();
	const themes: ThemeInfo[] = available.map((name) => ({
		name,
		description: `${name.charAt(0).toUpperCase() + name.slice(1)} theme`,
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
