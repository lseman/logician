// ── Session management ───────────────────────────────────────────────────
// Restore/auto-save/select/rename/delete session actions, plus opening the
// session browser overlay. handleSessionAction reaches into overlay
// internals (tui.removeOverlay/requestRender) as well as session state.

import { AgentCoreBridge } from "@logician/coding-agent/application";
import { SessionStore, Transcript, type Turn } from "@logician/coding-agent/sessions";
import { SessionBrowserOverlay } from "../../overlays/session-manager.ts";
import { StatusBar } from "../../status/status-bar.ts";
import { TUI } from "../../terminal/core.ts";
import { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import { turnsToMessages } from "./messages.ts";

export interface SessionControllerCtx {
	tui: TUI;
	bridge: AgentCoreBridge;
	transcript: Transcript;
	transcriptDisplay: TranscriptDisplay;
	statusPanel: StatusBar;
	sessionStore: SessionStore;
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
	ctx.transcript.loadTurns(turns);
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.bridge.restoreHistory(turnsToMessages(turns));
}

/** Auto-save the latest turn to the current session. */
export function autoSaveTurn(ctx: SessionControllerCtx): void {
	if (!ctx.currentSessionId) return;
	const turns = ctx.transcript.getTurns();
	const latestTurn = turns[turns.length - 1];
	if (latestTurn && latestTurn.isComplete) {
		ctx.sessionStore.saveTurn(latestTurn);
		// Update the session title from the first user message
		if (latestTurn.userMessage?.content &&  (latestTurn.userMessage?.content || "").length > 0) {
			const title =  (latestTurn.userMessage?.content ?? "")
				.replace(/\n+/g, " ")
				.slice(0, 60);
			const current = ctx.sessionStore.getSession(ctx.currentSessionId);
			if (current && current.title === "Untitled Session") {
				ctx.sessionStore.renameSession(ctx.currentSessionId, title);
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
			const session = ctx.sessionStore.getSession(action.sessionId);
			if (!session) return;

			// Save current session
			autoSaveTurn(ctx);

			// Load new session turns into transcript + model context
			const turns = ctx.sessionStore.loadTurns(action.sessionId);
			ctx.currentSessionId = action.sessionId;
			restoreSession(ctx, turns);
			ctx.statusPanel.update({
				sessionTitle: session.title,
				turnCount: turns.length,
			});
			ctx.tui.removeOverlay(ctx.sessionManager);
			ctx.tui.requestRender();
			break;
		}

		case "rename":
			if (!action.sessionId || !action.title) return;
			ctx.sessionStore.renameSession(action.sessionId, action.title);
			ctx.tui.removeOverlay(ctx.sessionManager);
			ctx.tui.requestRender();
			break;

		case "delete":
			if (!action.sessionId) return;
			ctx.sessionStore.deleteSession(action.sessionId);
			if (ctx.currentSessionId === action.sessionId) {
				// Switch to the next most recent session or create new
				const remaining = ctx.sessionStore.listSessions();
				if (remaining.length > 0) {
					ctx.currentSessionId = remaining[0].id;
					const turns = ctx.sessionStore.loadTurns(ctx.currentSessionId);
					restoreSession(ctx, turns);
					ctx.statusPanel.update({ sessionTitle: remaining[0].title });
				} else {
					ctx.currentSessionId = ctx.sessionStore.createSession({
						title: "New Session",
					});
					restoreSession(ctx, []);
				}
			}
			ctx.tui.removeOverlay(ctx.sessionManager);
			ctx.tui.requestRender();
			break;

		case "new":
			autoSaveTurn(ctx);
			ctx.currentSessionId = ctx.sessionStore.createSession({
				title: "New Session",
			});
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
