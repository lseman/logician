import { AgentCoreBridge } from "@logician/coding-agent/application";
import { Transcript } from "@logician/coding-agent/sessions";
import { FileMentionPopup } from "../../overlays/file-mention-popup.ts";
import { InputBar } from "../../input/input-bar.ts";
import { McpManagerOverlay } from "../../overlays/mcp-manager.ts";
import { ModelSelectorOverlay } from "../../overlays/model-selector.ts";
import { PluginManagerOverlay } from "../../overlays/plugin-manager.ts";
import { ReasonerSelectorOverlay } from "../../overlays/reasoner-selector.ts";
import { SettingsSelectorOverlay } from "../../overlays/settings-overlay.ts";
import { StatusBar } from "../../status/status-bar.ts";
import { ThemeSelectorOverlay } from "../../overlays/theme-selector.ts";
import { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import { TUI } from "../../terminal/core.ts";
import type { InferenceMode } from "../inference-settings.ts";

export type NotifyFn = (
	message: string,
	level?: "info" | "success" | "warning" | "error",
) => void;

export interface OverlayHandlersCtx {
	tui: TUI;
	bridge: AgentCoreBridge;
	transcript: Transcript;
	transcriptDisplay: TranscriptDisplay;
	statusPanel: StatusBar;
	inputBar: InputBar;
	notify: NotifyFn;
	pluginManager: PluginManagerOverlay;
	mcpManager: McpManagerOverlay;
	reasonerSelector: ReasonerSelectorOverlay;
	fileMentionPopup: FileMentionPopup;
	fileMentionListedCwd: string | null;
	fileMentionListing: Promise<string[]> | null;
	modelSelector: ModelSelectorOverlay;
	themeSelector: ThemeSelectorOverlay;
	settingsSelector: SettingsSelectorOverlay;
	thinkingLevel: string;
	inferenceMode: InferenceMode;
}
