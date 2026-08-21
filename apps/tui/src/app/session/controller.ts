// ── Session management ───────────────────────────────────────────────────
// Restore/auto-save/select/rename/delete session actions, plus opening the
// session browser overlay. handleSessionAction reaches into overlay
// internals (tui.removeOverlay/requestRender) as well as session state.

import type { AgentRuntime } from "@logician/log-runtime/application";
import {
	inferSessionTitle,
	isGeneratedSessionTitle,
	type Transcript,
	type TuiSessionService,
	type Turn,
} from "@logician/log-runtime/sessions";
import type { SessionBrowserOverlay } from "../../overlays/session-manager.ts";
import type { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import type { StatusBar } from "../../status/status-bar.ts";
import type { TuiHandle } from "../../terminal/core.ts";
import { turnsToMessages } from "./messages.ts";

export interface SessionControllerCtx {
	tui: TuiHandle;
	bridge: AgentRuntime;
	transcript: Transcript;
	transcriptDisplay: TranscriptDisplay;
	statusPanel: StatusBar;
	sessionService: TuiSessionService;
	sessionManager: SessionBrowserOverlay;
	currentSessionId: string | null;
}

/**
 * Restore a session into BOTH the UI transcript and the model context.
 * Without the bridge restore, a resumed session renders its history but the
 * model starts cold ("continue" loses everything). Pass [] for a fresh
 * session (clears both).
 */
export function restoreSession(ctx: SessionControllerCtx, turns: Turn[]): void {
	if (ctx.currentSessionId) {
		ctx.sessionService.setCurrentSession(ctx.currentSessionId);
		ctx.bridge.useConversationSession(
			ctx.currentSessionId,
			ctx.sessionService.getRawSession(ctx.currentSessionId) ?? undefined,
		);
	}
	ctx.transcript.loadTurns(turns);
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.bridge.restoreHistory(turnsToMessages(turns));
}

/** Auto-save the latest turn to the current session. */
export function autoSaveTurn(ctx: SessionControllerCtx): void {
	if (!ctx.currentSessionId) return;
	const turns = ctx.transcript.getTurns();
	const latestTurn = turns[turns.length - 1];
	if (latestTurn?.isComplete) {
		ctx.sessionService.saveTurn(ctx.currentSessionId, latestTurn);
		// Generated placeholders follow the first meaningful topic. Explicitly
		// renamed sessions are never overwritten.
		if (latestTurn.userMessage?.content) {
			const current = ctx.sessionService.getSession(ctx.currentSessionId);
			const agentResponse =
				latestTurn.assistantMessage?.chunks
					.filter(chunk => chunk.type === "content" && chunk.contentText)
					.map(chunk => chunk.contentText)
					.join("") || "";
			const inferred = inferSessionTitle(
				latestTurn.userMessage.content,
				agentResponse,
			);
			if (current && inferred && isGeneratedSessionTitle(current.name)) {
				ctx.sessionService.renameSession(ctx.currentSessionId, inferred);
				ctx.bridge.renameConversationSession(ctx.currentSessionId, inferred);
				ctx.statusPanel.update({ sessionTitle: inferred });
			}
		}
	}
}

/** Handle session manager actions (select, rename, delete, new). */
export function handleSessionAction(
	ctx: SessionControllerCtx,
	action: {
		type: "close" | "select" | "rename" | "delete" | "new";
		sessionId?: string;
		title?: string;
	},
): void {
	switch (action.type) {
		case "close":
			ctx.tui.removeOverlay(ctx.sessionManager);
			break;

		case "select": {
			if (!action.sessionId) return;
			const session = ctx.sessionService.getSession(action.sessionId);
			if (!session) return;

			// Save current session
			autoSaveTurn(ctx);

			// Load new session turns into transcript + model context
			const turns = ctx.sessionService.loadTurns(action.sessionId);
			ctx.currentSessionId = action.sessionId;
			restoreSession(ctx, turns);
			ctx.statusPanel.update({
				sessionTitle: session.name,
				turnCount: turns.length,
			});
			ctx.tui.removeOverlay(ctx.sessionManager);
			ctx.tui.requestRender();
			break;
		}

		case "rename":
			if (!action.sessionId || !action.title) return;
			ctx.sessionService.renameSession(action.sessionId, action.title);
			ctx.bridge.renameConversationSession(action.sessionId, action.title);
			ctx.tui.removeOverlay(ctx.sessionManager);
			ctx.tui.requestRender();
			break;

		case "delete":
			if (!action.sessionId) return;
			ctx.sessionService.deleteSession(action.sessionId);
			if (ctx.currentSessionId === action.sessionId) {
				// Switch to the next most recent session or create new
				const remaining = ctx.sessionService.listSessions();
				if (remaining.length > 0) {
					ctx.currentSessionId = remaining[0].id;
					const turns = ctx.sessionService.loadTurns(ctx.currentSessionId);
					restoreSession(ctx, turns);
					ctx.statusPanel.update({ sessionTitle: remaining[0].name });
				} else {
					ctx.currentSessionId =
						ctx.sessionService.createSession("New Session");
					restoreSession(ctx, []);
				}
			}
			ctx.tui.removeOverlay(ctx.sessionManager);
			ctx.tui.requestRender();
			break;

		case "new":
			autoSaveTurn(ctx);
			ctx.currentSessionId = ctx.sessionService.createSession("New Session");
			restoreSession(ctx, []);
			ctx.statusPanel.update({ sessionTitle: "New Session" });
			ctx.tui.removeOverlay(ctx.sessionManager);
			ctx.tui.requestRender();
			break;
	}
}

/** Open the session manager overlay. */
export function openSessionManager(ctx: SessionControllerCtx): void {
	ctx.statusPanel.update({ phase: "sessions" });
	ctx.sessionManager.refresh();
	ctx.sessionManager.show();
	const overlay = ctx.tui.showOverlay(ctx.sessionManager, {
		anchor: "aboveInput",
		align: "left",
		maxHeight: 18,
	});
	overlay.focus();
}
