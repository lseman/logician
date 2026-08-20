// ── Session + hook lifecycle plumbing for AgentHarness ────────────────────
// SessionStart/SessionEnd/PreCompact/PostCompact hook emission and
// session list/resume helpers. Failures here must never break a turn, so
// every call swallows its own errors — matches prior harness behavior.

import {
	runHookEvent,
	runSessionStartHooks,
} from "../../../adapters/claude-code/plugin-runtime.ts";
import type {
	BeforeCompactContext,
	BeforeCompactResult,
} from "../../types/index.ts";

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
