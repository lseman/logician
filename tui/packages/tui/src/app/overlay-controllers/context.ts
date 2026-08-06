import type { AgentCoreBridge } from "@logician/coding-agent/application";
import type { Transcript } from "@logician/coding-agent/sessions";
import type { InputBar } from "../../input/input-bar.ts";
import type { FileMentionPopup } from "../../overlays/file-mention-popup.ts";
import type { McpManagerOverlay } from "../../overlays/mcp-manager.ts";
import type { ModelSelectorOverlay } from "../../overlays/model-selector.ts";
import type { PluginManagerOverlay } from "../../overlays/plugin-manager.ts";
import type { ReasonerSelectorOverlay } from "../../overlays/reasoner-selector.ts";
import type { SettingsSelectorOverlay } from "../../overlays/settings-overlay.ts";
import type { ThemeSelectorOverlay } from "../../overlays/theme-selector.ts";
import type { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import type { StatusBar } from "../../status/status-bar.ts";
import type { TUI } from "../../terminal/core.ts";
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
