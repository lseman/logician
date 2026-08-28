// ── Slash command handlers ──────────────────────────────────────────────────
// The localHandlers registry passed to createSlashCommands(), plus the
// slashPopup submit dispatcher and its supporting async helpers
// (status/plugins/mcp/reasoner/theme). Extracted from the constructor's
// inline slash-command wiring block.

import type { AutoresearchSession } from "@logician/log-autoresearch";
import type {
	AgentRuntime,
	GoalTracker,
	LoopRunner,
} from "@logician/log-runtime/application";
import type {
	Transcript,
	TuiSessionService,
} from "@logician/log-runtime/sessions";
import type { ChoicePopup } from "../../overlays/choice-popup.ts";
import type { SlashPopup } from "../../overlays/slash-popup.ts";
import type { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import type { NotificationCenter } from "../../status/notification-center.ts";
import type { StatusBar } from "../../status/status-bar.ts";
import type { TuiHandle } from "../../terminal/core.ts";

export interface SlashCommandsCtx {
	tui: TuiHandle;
	bridge: AgentRuntime;
	transcript: Transcript;
	transcriptDisplay: TranscriptDisplay;
	statusPanel: StatusBar;
	notifications: NotificationCenter;
	choicePopup: ChoicePopup;
	choicePopupPreview: boolean;
	slashPopup: SlashPopup;
	thinkingDisplayMode: "collapsed" | "summary" | "expanded";
	currentSessionId: string | null;
	sessionService: TuiSessionService;
	loopManager: LoopRunner;
	loopActive: boolean;
	goalManager: GoalTracker;
	goalActive: boolean;
	researchManager: AutoresearchSession;
	inferenceMode: string;
	workflowMode: "act" | "plan";
	planPhase: "idle" | "planning" | "awaiting_approval" | "executing";
	normalPermissionMode: "acceptAll" | "acceptEdits" | "ask";
	applyThinkingLevel: (level: string) => void;
	setInferenceMode: (mode: string) => void;
	setExecutionProfile: (profile: "autonomous" | "minimal") => void;
	cycleExecutionProfile: () => "autonomous" | "minimal";
	togglePlanMode: () => "acceptEdits" | "plan";
	setPlanMode: (planning: boolean) => "acceptEdits" | "plan";
	cycleInferenceMode: () => void;
	openSettingsSelector: () => Promise<void>;
	openSessionManager: () => void;
	openModelSelector: () => void;
	openQueueManager: () => void;
	openPluginManager: () => Promise<void>;
	openMcpManager: () => Promise<void>;
	openReasonerSelector: () => Promise<void>;
	openThemeSelector: () => Promise<void>;
	openThinkingLevelSelector: () => void;
	setThemeByName: (name: string) => boolean;
	_autoSaveTurn: () => void;
	stop: () => Promise<void>;
	requestExit: () => void;
}
