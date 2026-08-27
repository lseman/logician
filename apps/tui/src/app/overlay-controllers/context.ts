import type { AutoresearchSession } from "@logician/log-autoresearch";
import type { AgentRuntime } from "@logician/log-runtime/application";
import type { Transcript } from "@logician/log-runtime/sessions";
import type { InputBar } from "../../input/input-bar.ts";
import type { AutoresearchDashboardOverlay } from "../../overlays/autoresearch-dashboard.ts";
import type { FileMentionPopup } from "../../overlays/file-mention-popup.ts";
import type { InferenceModeSelector } from "../../overlays/inference-mode-selector.ts";
import type { McpManagerOverlay } from "../../overlays/mcp-manager.ts";
import type { ModelSelectorOverlay } from "../../overlays/model-selector.ts";
import type { PluginManagerOverlay } from "../../overlays/plugin-manager.ts";
import type { QueueManagerOverlay } from "../../overlays/queue-manager.ts";
import type { ReasonerSelectorOverlay } from "../../overlays/reasoner-selector.ts";
import type { SessionTreeOverlay } from "../../overlays/session-tree.ts";
import type { SettingsSelectorOverlay } from "../../overlays/settings-overlay.ts";
import type { ThemeSelectorOverlay } from "../../overlays/theme-selector.ts";
import type { ThinkingLevelSelectorOverlay } from "../../overlays/thinking-level-selector.ts";
import type { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import type { StatusBar } from "../../status/status-bar.ts";
import type { TuiHandle } from "../../terminal/core.ts";
import type { InferenceMode } from "../inference-settings.ts";

type NotifyFn = (
	message: string,
	level?: "info" | "success" | "warning" | "error",
) => void;

export interface OverlayHandlersCtx {
	tui: TuiHandle;
	bridge: AgentRuntime;
	transcript: Transcript;
	transcriptDisplay: TranscriptDisplay;
	statusPanel: StatusBar;
	inputBar: InputBar;
	notify: NotifyFn;
	pluginManager: PluginManagerOverlay;
	autoresearchDashboard: AutoresearchDashboardOverlay;
	researchManager: AutoresearchSession;
	mcpManager: McpManagerOverlay;
	reasonerSelector: InstanceType<typeof ReasonerSelectorOverlay>;
	queueManager: QueueManagerOverlay;
	sessionTree: SessionTreeOverlay;
	fileMentionPopup: FileMentionPopup;
	fileMentionListedCwd: string | null;
	fileMentionListing: Promise<string[]> | null;
	modelSelector: InstanceType<typeof ModelSelectorOverlay>;
	inferenceModeSelector: InstanceType<typeof InferenceModeSelector>;
	themeSelector: InstanceType<typeof ThemeSelectorOverlay>;
	settingsSelector: SettingsSelectorOverlay;
	thinkingLevelSelector: InstanceType<typeof ThinkingLevelSelectorOverlay>;
	sessionService: import("@logician/log-runtime/sessions").TuiSessionService;
	thinkingLevel: string;
	inferenceMode: InferenceMode;
	workflowMode: "act" | "plan";
	planPhase: "idle" | "planning" | "awaiting_approval" | "executing";
	normalPermissionMode: "acceptAll" | "acceptEdits" | "ask";
}
