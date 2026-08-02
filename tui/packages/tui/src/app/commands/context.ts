// ── Slash command handlers ──────────────────────────────────────────────────
// The localHandlers registry passed to createSlashCommands(), plus the
// slashPopup submit dispatcher and its supporting async helpers
// (status/plugins/mcp/reasoner/theme). Extracted from the constructor's
// inline slash-command wiring block.

import {
	AgentCoreBridge,
	GoalManager,
	LoopManager,
} from "@logician/coding-agent/application";
import { SessionStore, Transcript } from "@logician/coding-agent/sessions";
import type { MemoryStore } from "@logician/memory";
import { ChoicePopup } from "../../overlays/choice-popup.ts";
import { SlashPopup } from "../../overlays/slash-popup.ts";
import { NotificationCenter } from "../../status/notification-center.ts";
import { StatusBar } from "../../status/status-bar.ts";
import { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import { TUI } from "../../terminal/core.ts";

export interface SlashCommandsCtx {
	tui: TUI;
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
	sessionStore: SessionStore;
	loopManager: LoopManager;
	loopActive: boolean;
	goalManager: GoalManager;
	goalActive: boolean;
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
	getMemoryStore: () => MemoryStore | null;
}
