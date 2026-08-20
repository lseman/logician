// ── Slash command handlers ──────────────────────────────────────────────────
// The localHandlers registry passed to createSlashCommands(), plus the
// slashPopup submit dispatcher and its supporting async helpers
// (status/plugins/mcp/reasoner/theme). Extracted from the constructor's
// inline slash-command wiring block.

import type { AutoresearchSession } from "@logician/autoresearch";
import type {
	AgentCoreBridge,
	GoalManager,
	LoopManager,
} from "@logician/agent-core/application";
import type {
	Transcript,
	TuiSessionService,
} from "@logician/agent-core/sessions";
import type { MemoryStore } from "@logician/memory";
import type { ChoicePopup } from "../../overlays/choice-popup.ts";
import type { SlashPopup } from "../../overlays/slash-popup.ts";
import type { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import type { NotificationCenter } from "../../status/notification-center.ts";
import type { StatusBar } from "../../status/status-bar.ts";
import type { TuiHandle } from "../../terminal/core.ts";

export interface SlashCommandsCtx {
	tui: TuiHandle;
	bridge: AgentCoreBridge;
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
	loopManager: LoopManager;
	loopActive: boolean;
	goalManager: GoalManager;
	goalActive: boolean;
	researchManager: AutoresearchSession;
	inferenceMode: string;
	applyThinkingLevel: (level: string) => void;
	setInferenceMode: (mode: string) => void;
	setExecutionProfile: (profile: "autonomous" | "minimal") => void;
	cycleExecutionProfile: () => "autonomous" | "minimal";
	cycleInferenceMode: () => void;
	openSettingsSelector: () => Promise<void>;
	openSessionManager: () => void;
	openModelSelector: () => void;
	openPluginManager: () => Promise<void>;
	openMcpManager: () => Promise<void>;
	openReasonerSelector: () => Promise<void>;
	openThemeSelector: () => Promise<void>;
	setThemeByName: (name: string) => boolean;
	_autoSaveTurn: () => void;
	stop: () => Promise<void>;
	requestExit: () => void;
	getMemoryStore: () => MemoryStore | null;
}
