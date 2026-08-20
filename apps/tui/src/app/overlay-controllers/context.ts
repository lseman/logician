import type { AutoresearchSession } from "@logician/autoresearch";
import type { AgentCoreBridge } from "@logician/agent-core/application";
import type { Transcript } from "@logician/agent-core/sessions";
import type { InputBar } from "../../input/input-bar.ts";
import type { AutoresearchDashboardOverlay } from "../../overlays/autoresearch-dashboard.ts";
import type { FileMentionPopup } from "../../overlays/file-mention-popup.ts";
import type { InferenceModeSelector } from "../../overlays/inference-mode-selector.ts";
import type { McpManagerOverlay } from "../../overlays/mcp-manager.ts";
import type { ModelSelectorOverlay } from "../../overlays/model-selector.ts";
import type { PluginManagerOverlay } from "../../overlays/plugin-manager.ts";
import type { ReasonerSelectorOverlay } from "../../overlays/reasoner-selector.ts";
import type { SettingsSelectorOverlay } from "../../overlays/settings-overlay.ts";
import type { ThemeSelectorOverlay } from "../../overlays/theme-selector.ts";
import type { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import type { StatusBar } from "../../status/status-bar.ts";
import type { TuiHandle } from "../../terminal/core.ts";
import type { InferenceMode } from "../inference-settings.ts";

export type NotifyFn = (
	message: string,
	level?: "info" | "success" | "warning" | "error",
) => void;

export interface OverlayHandlersCtx {
	tui: TuiHandle;
	bridge: AgentCoreBridge;
	transcript: Transcript;
	transcriptDisplay: TranscriptDisplay;
	statusPanel: StatusBar;
	inputBar: InputBar;
	notify: NotifyFn;
	pluginManager: PluginManagerOverlay;
	autoresearchDashboard: AutoresearchDashboardOverlay;
	researchManager: AutoresearchSession;
	mcpManager: McpManagerOverlay;
	reasonerSelector: ReasonerSelectorOverlay;
	fileMentionPopup: FileMentionPopup;
	fileMentionListedCwd: string | null;
	fileMentionListing: Promise<string[]> | null;
	modelSelector: ModelSelectorOverlay;
	inferenceModeSelector: InferenceModeSelector;
	themeSelector: ThemeSelectorOverlay;
	settingsSelector: SettingsSelectorOverlay;
	thinkingLevel: string;
	inferenceMode: InferenceMode;
}
