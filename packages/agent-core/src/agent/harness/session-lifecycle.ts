// ── Session + hook lifecycle plumbing for AgentHarness ────────────────────
// SessionStart/SessionEnd/PreCompact/PostCompact hook emission and
// session list/resume helpers. Failures here must never break a turn, so
// every call swallows its own errors — matches prior harness behavior.

import {
	runHookEvent,
	runSessionStartHooks,
} from "../../tools/shared/plugins.ts";
import { Session, SessionManager } from "../core/session.ts";
import type {
	BeforeCompactContext,
	BeforeCompactResult,
	Message,
} from "../types/index.ts";

export interface HookContext {
	sessionId: string;
	transcriptPath: string;
	cwd: string;
}

export async function emitSessionStart(
	hooksEnabled: boolean,
	ctx: HookContext,
	source: string = "startup",
): Promise<boolean> {
	if (!hooksEnabled) return false;
	try {
		await runSessionStartHooks({
			source,
			session_id: ctx.sessionId,
			transcript_path: ctx.transcriptPath,
			cwd: ctx.cwd,
		});
		return true;
	} catch (_e: unknown) {
		console.error("[harness-session] emitSessionStart failed:", _e);
		return false;
	}
}

export async function emitSessionEnd(
	hooksEnabled: boolean,
	ctx: HookContext,
	reason: string = "other",
): Promise<void> {
	if (!hooksEnabled) return;
	try {
		await runHookEvent("SessionEnd", {
			session_id: ctx.sessionId,
			transcript_path: ctx.transcriptPath,
			cwd: ctx.cwd,
			reason,
		});
	} catch (_e: unknown) {
		// must not block cleanup
		console.error("[harness-session] emitSessionEnd failed:", _e);
	}
}

export async function emitPreCompact(
	hooksEnabled: boolean,
	ctx: HookContext,
	userHook:
		| ((
				ctx: BeforeCompactContext,
		  ) =>
				| Promise<BeforeCompactResult | undefined>
				| BeforeCompactResult
				| undefined)
		| undefined,
	compactCtx?: BeforeCompactContext,
): Promise<BeforeCompactResult | undefined> {
	let hookResult: BeforeCompactResult | undefined;
	if (compactCtx) {
		try {
			hookResult = await userHook?.(compactCtx);
		} catch (_e: unknown) {
			// must not block compaction
			console.error("[harness-session] emitPreCompact hook failed:", _e);
		}
	}
	if (!hooksEnabled) return hookResult;
	try {
		await runHookEvent("PreCompact", {
			session_id: ctx.sessionId,
			transcript_path: ctx.transcriptPath,
			cwd: ctx.cwd,
		});
	} catch (_e: unknown) {
		// must not block compaction
		console.error("[harness-session] emitPreCompact hookEvent failed:", _e);
	}
	return hookResult;
}

export async function emitPostCompact(
	hooksEnabled: boolean,
	ctx: HookContext,
): Promise<void> {
	if (!hooksEnabled) return;
	try {
		await runHookEvent("PostCompact", {
			session_id: ctx.sessionId,
			transcript_path: ctx.transcriptPath,
			cwd: ctx.cwd,
		});
	} catch (_e: unknown) {
		// must not block compaction
		console.error("[harness-session] emitPostCompact failed:", _e);
	}
}

export interface ResumedSession {
	session: Session;
	messages: Message[];
}

/** Load a persisted session's messages, or null if load failed/empty. */
export function loadSessionMessages(
	sessionId: string,
	baseDir: string | undefined,
): ResumedSession | null {
	try {
		const session = new Session(sessionId, { baseDir, enabled: true });
		const persisted = session.load();
		if (persisted.length === 0) return { session, messages: [] };
		const messages: Message[] = persisted.map(m => ({
			role: m.role as Message["role"],
			content: m.content,
			tool_call_id: m.tool_call_id,
			tool_calls: m.tool_calls,
			name: m.name,
			timestamp: m.timestamp,
		}));
		return { session, messages };
	} catch (_e: unknown) {
		console.error("[harness-session] loadSessionMessages failed:", _e);
		return null;
	}
}

export function listSessions(baseDir: string | undefined): Array<{
	id: string;
	name?: string;
	messageCount: number;
	lastActivity: number;
}> {
	try {
		const manager = new SessionManager({ baseDir });
		return manager
			.listSessions()
			.map(
				(m: {
					id: string;
					name?: string;
					messageCount: number;
					lastActivity: number;
				}) => ({
					id: m.id,
					name: m.name,
					messageCount: m.messageCount,
					lastActivity: m.lastActivity,
				}),
			);
	} catch (_e: unknown) {
		console.error("[harness-session] listSessions failed:", _e);
		return [];
	}
}
