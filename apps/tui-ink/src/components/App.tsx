// ── Ink TUI — Main App Component ──────────────────────────────────────────────

import React, { useEffect, useMemo, useState } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import type { AgentRuntime } from "@logician/log-runtime/application";
import type { Transcript, TuiSessionService } from "@logician/log-runtime/sessions";
import type { SlashCommandDef } from "@logician/log-runtime/commands";
import { TuiState } from "../state.ts";
import type { AppConfig, OverlayKind } from "../types.ts";
import { getCurrentTheme } from "../theme.ts";
import { useTuiState } from "../hooks/useTuiState.ts";
import { buildSlashCommands, dispatchSlash } from "../slash.ts";
import { listFileSuggestions } from "../utils.ts";
import { TranscriptDisplay } from "./TranscriptDisplay.tsx";
import { InputBar } from "./InputBar.tsx";
import { StatusBar } from "./StatusBar.tsx";
import { SlashPopup } from "../overlays/SlashPopup.tsx";
import { ChoicePopup } from "../overlays/ChoicePopup.tsx";
import { PermissionPopup } from "../overlays/PermissionPopup.tsx";
import { SessionManager } from "../overlays/SessionManager.tsx";
import { ModelSelector } from "../overlays/ModelSelector.tsx";
import { ThemeSelector } from "../overlays/ThemeSelector.tsx";
import { SettingsSelector } from "../overlays/SettingsSelector.tsx";
import { PluginManager } from "../overlays/PluginManager.tsx";
import { McpManager } from "../overlays/McpManager.tsx";
import { ReasonerSelector } from "../overlays/ReasonerSelector.tsx";
import { QueueManager } from "../overlays/QueueManager.tsx";
import { AutoresearchDashboard } from "../overlays/AutoresearchDashboard.tsx";
import { ThinkingLevelSelector } from "../overlays/ThinkingLevelSelector.tsx";
import { InferenceModeSelector } from "../overlays/InferenceModeSelector.tsx";
import { SessionTree } from "../overlays/SessionTree.tsx";
import { FileMentionPopup } from "../overlays/FileMentionPopup.tsx";

interface AppProps {
	config: AppConfig;
	state: TuiState;
	transcript: Transcript;
	bridge: AgentRuntime;
	sessionService: TuiSessionService;
	currentSessionId: string;
	onSubmit: (message: string) => Promise<void>;
	onCancel: () => Promise<void>;
	onExit: () => Promise<void>;
}

export const App: React.FC<AppProps> = ({
	config,
	state,
	transcript,
	bridge,
	sessionService,
	currentSessionId,
	onSubmit,
	onCancel,
	onExit,
}) => {
	useTuiState(state);
	const { stdout } = useStdout();
	const [width, setWidth] = useState(stdout?.columns ?? 80);
	const [inputValue, setInputValue] = useState("");

	const theme = getCurrentTheme();
	const overlay = state.overlay.kind;
	const overlayOpen = overlay !== null;

	const slashCtx = useMemo(
		() => ({ bridge, transcript, state, onExit: () => void onExit() }),
		[bridge, transcript, state, onExit],
	);
	const slashCommands: SlashCommandDef[] = useMemo(
		() => buildSlashCommands(slashCtx),
		[slashCtx],
	);

	const openOverlay = (kind: OverlayKind): void => state.setOverlay(kind);
	const closeOverlay = (): void => state.setOverlay(null);

	// ── Window resize ────────────────────────────────────────────────────
	useEffect(() => {
		if (!stdout) return;
		const onResize = (): void => setWidth(stdout.columns ?? 80);
		stdout.on("resize", onResize);
		return () => {
			stdout.off("resize", onResize);
		};
	}, [stdout]);

	// ── Input-driven autocomplete popups ─────────────────────────────────
	useEffect(() => {
		if (inputValue.startsWith("/")) {
			state.setSlashQuery(inputValue.slice(1));
			if (overlay === null) state.setOverlay("slash");
			else if (overlay === "slash") state.touch();
			return;
		}
		const at = inputValue.lastIndexOf("@");
		if (at >= 0 && !/\s/.test(inputValue.slice(at + 1))) {
			const query = inputValue.slice(at + 1);
			state.fileMentionQuery = query;
			state.setFileSuggestions(listFileSuggestions(state.cwd, query));
			if (overlay === null) state.setOverlay("fileMention");
			return;
		}
		if (overlay === "slash" || overlay === "fileMention") state.setOverlay(null);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [inputValue]);

	// ── Global shortcuts (inert while an overlay owns the keyboard) ───────
	useInput(
		(input, key) => {
			if (key.ctrl && input === "c") {
				void onExit();
				return;
			}
			if (key.escape) {
				if (bridge.isActive()) void onCancel();
				return;
			}
			// Scroll shortcuts
			if (key.pageUp || (key.ctrl && input === "u")) {
				state.pageUp();
				return;
			}
			if (key.pageDown || (key.ctrl && input === "d")) {
				state.pageDown();
				return;
			}
			// Follow mode toggle
			if (key.home) {
				state.setFollowMode(true);
				return;
			}
			if (key.end) {
				state.setFollowMode(false);
				return;
			}
			if (key.shift && key.tab) {
				const next = state.workflowMode === "plan" ? "act" : "plan";
				state.setWorkflowMode(next);
				config.source.workflowMode = next;
				state.showNotification(`Workflow mode: ${next}`, "info");
				return;
			}
			if (!key.ctrl) return;
			// Ctrl+M/I/H/J collide with Enter/Tab/Backspace/LF in a raw TTY, so
			// they are not bound — use slash commands (/model, /thinking-steps …).
			const map: Record<string, OverlayKind> = {
				s: "sessionManager",
				p: "modelSelector",
				t: "themeSelector",
				r: "reasonerSelector",
				a: "autoresearchDashboard",
				q: "queueManager",
				o: "sessionTree",
				l: "settingsSelector",
				g: "fileMention",
				b: "thinkingLevelSelector",
				k: "inferenceModeSelector",
				y: "pluginManager",
			};
			if (map[input]) openOverlay(map[input]!);
		},
		{ isActive: !overlayOpen },
	);

	const submit = (): void => {
		const text = inputValue.trim();
		if (!text) return;
		if (text.startsWith("/")) {
			dispatchSlash(slashCtx, slashCommands, text);
			setInputValue("");
			state.setOverlay(null);
			return;
		}
		void onSubmit(text);
		setInputValue("");
	};

	const sessionOverlayItems = () => sessionService.listSessions();

	const loadSession = (sessionId: string): void => {
		if (!sessionId) return;
		try {
			const turns = sessionService.loadTurns(sessionId);
			transcript.loadTurns(turns);
			state.setTranscriptTurns(transcript.getTurns());
			bridge.useConversationSession(
				sessionId,
				sessionService.getRawSession(sessionId) ?? undefined,
			);
			state.setCurrentSession(
				sessionId,
				sessionService.getSession(sessionId)?.name ?? "Session",
			);
		} catch (error) {
			state.showNotification(
				`Failed to load session: ${error instanceof Error ? error.message : String(error)}`,
				"error",
			);
		}
	};

	return (
		<Box flexDirection="column" width="100%">
			<Box flexGrow={1} flexDirection="column" overflow="hidden">
				<TranscriptDisplay
					turns={state.transcriptTurns}
					thinkingMode={state.thinkingDisplayMode}
					maxMessageLength={config.source.truncation?.transcriptMessageMaxChars}
					scrollOffset={state.scrollOffset}
					maxVisibleTurns={stdout ? Math.max(5, stdout.rows - 12) : 60}
					hasNewOutputBelow={!state.followMode && state.transcriptTurns.length > 0}
				/>
			</Box>

			{state.notifications.length > 0 && (
				<Box flexDirection="column">
					{state.notifications.map(n => (
						<Text
							key={n.id}
							color={
								n.level === "error"
									? theme.fg.error
									: n.level === "warn"
										? theme.fg.warning
										: theme.fg.info
							}
						>
							{`• ${n.message}`}
						</Text>
					))}
				</Box>
			)}

			<Box>
				<Text color={theme.fg.secondary}>{"─".repeat(Math.min(width, 120))}</Text>
			</Box>

			<Box flexDirection="row">
				<Text color={theme.fg.accent} bold>
					{state.workflowMode === "plan" ? "plan> " : "> "}
				</Text>
				<InputBar
					value={inputValue}
					onValueChange={setInputValue}
					onSubmit={submit}
					isActive={!overlayOpen}
				/>
			</Box>

			<StatusBar state={state} config={config} width={width} />

			{overlay === "slash" && (
				<SlashPopup
					commands={slashCommands}
					query={state.slashQuery}
					isActive
					onSelect={(cmd: SlashCommandDef) => {
						// Selecting from the list runs the command now. Commands that
						// need arguments will print their usage; the user can then
						// retype with args.
						setInputValue("");
						state.setOverlay(null);
						dispatchSlash(slashCtx, slashCommands, cmd.command);
					}}
					onClose={closeOverlay}
				/>
			)}

			{overlay === "fileMention" && (
				<FileMentionPopup
					files={state.fileSuggestions}
					query={state.fileMentionQuery}
					isActive
					onSelect={(path: string) => {
						const at = inputValue.lastIndexOf("@");
						setInputValue(
							(at >= 0 ? inputValue.slice(0, at) : inputValue) + path + " ",
						);
						state.setOverlay(null);
					}}
					onClose={closeOverlay}
				/>
			)}

			{overlay === "choice" && (
				<ChoicePopup
					questionId={(state.overlay.data?.questionId as string) ?? ""}
					questions={
						(state.overlay.data?.questions as ChoiceQuestion[] | undefined) ?? []
					}
					isActive
					onSubmit={(questionId, answer) => {
						bridge.respondToQuestion(questionId, answer);
						state.setOverlay(null);
					}}
					onClose={() => {
						const qid = state.overlay.data?.questionId as string;
						if (qid) bridge.respondToQuestion(qid, "__dismissed__");
						state.setOverlay(null);
					}}
				/>
			)}

			{overlay === "permission" && (
				<PermissionPopup
					toolName={(state.overlay.data?.toolName as string) ?? ""}
					toolCallId={(state.overlay.data?.toolCallId as string) ?? ""}
					args={state.overlay.data?.args}
					isActive
					onDecision={decision => {
						const id = state.overlay.data?.toolCallId as string;
						if (id) bridge.respondToPermission(id, decision);
						state.setOverlay(null);
					}}
				/>
			)}

			{overlay === "sessionManager" && (
				<SessionManager
					sessions={sessionOverlayItems()}
					currentSessionId={state.currentSessionId ?? currentSessionId}
					isActive
					onSelect={(sessionId: string) => {
						loadSession(sessionId);
						closeOverlay();
					}}
					onNew={() => {
						const id = sessionService.createSession("New Session");
						transcript.clear();
						state.setTranscriptTurns(transcript.getTurns());
						bridge.useConversationSession(
							id,
							sessionService.getRawSession(id) ?? undefined,
						);
						state.setCurrentSession(id, "New Session");
						closeOverlay();
					}}
					onDelete={(sessionId: string) => {
						if (sessionId === (state.currentSessionId ?? currentSessionId)) {
							state.showNotification("Cannot delete the current session", "error");
							return;
						}
						sessionService.deleteSession(sessionId);
						state.touch();
					}}
					onClose={closeOverlay}
				/>
			)}

			{overlay === "modelSelector" && (
				<ModelSelector
					bridge={bridge}
					isActive
					onSelect={model => {
						if (model) {
							bridge.models.select(model);
							state.setModel(model);
						}
						closeOverlay();
					}}
					onClose={closeOverlay}
				/>
			)}

			{overlay === "themeSelector" && (
				<ThemeSelector
					currentTheme={theme.name}
					isActive
					onSelect={() => {
						state.touch();
						closeOverlay();
					}}
					onClose={closeOverlay}
				/>
			)}

			{overlay === "settingsSelector" && (
				<SettingsSelector state={state} config={config} bridge={bridge} isActive onClose={closeOverlay} />
			)}

			{overlay === "pluginManager" && (
				<PluginManager bridge={bridge} isActive onClose={closeOverlay} />
			)}

			{overlay === "mcpManager" && (
				<McpManager bridge={bridge} isActive onClose={closeOverlay} />
			)}

			{overlay === "reasonerSelector" && (
				<ReasonerSelector reasoner={state.reasoner} isActive onClose={closeOverlay} />
			)}

			{overlay === "queueManager" && (
				<QueueManager
					bridge={bridge}
					messages={state.steerMessages}
					isActive
					onClose={closeOverlay}
				/>
			)}

			{overlay === "autoresearchDashboard" && (
				<AutoresearchDashboard
					active={state.researchActive}
					status={state.researchStatus}
					iteration={state.researchIteration}
					isActive
					onClose={closeOverlay}
				/>
			)}

			{overlay === "thinkingLevelSelector" && (
				<ThinkingLevelSelector
					currentLevel={state.thinkingLevel}
					isActive
					onSelect={level => {
						state.setThinkingLevel(level);
						bridge.updateSettings({ thinkingLevel: level as never });
						closeOverlay();
					}}
					onClose={closeOverlay}
				/>
			)}

			{overlay === "inferenceModeSelector" && (
				<InferenceModeSelector
					currentMode={state.inferenceMode}
					isActive
					onSelect={mode => {
						state.setInferenceMode(mode);
						bridge.updateSettings({ inferenceMode: mode as never });
						closeOverlay();
					}}
					onClose={closeOverlay}
				/>
			)}

			{overlay === "sessionTree" && (
				<SessionTree
					sessions={sessionService.listSessions()}
					isActive
					onClose={closeOverlay}
				/>
			)}
		</Box>
	);
};

interface ChoiceQuestion {
	id: string;
	header?: string;
	question: string;
	choices: Array<{ value: string; label: string; description?: string }>;
}
