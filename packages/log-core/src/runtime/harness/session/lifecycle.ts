// ── Session + hook lifecycle plumbing for AgentHarness ────────────────────
// SessionStart/SessionEnd/PreCompact/PostCompact hook emission and
// session list/resume helpers. Failures here must never break a turn, so
// every call swallows its own errors — matches prior harness behavior.

import type {
	BeforeCompactContext,
	BeforeCompactResult,
} from "../../../system/types/types-messages.ts";
import type { HarnessPluginLifecycle } from "../types.ts";

export interface HookContext {
	sessionId: string;
	transcriptPath: string;
	cwd: string;
}

export async function emitSessionStart(
	hooksEnabled: boolean,
	ctx: HookContext,
	source: string = "startup",
	pluginLifecycle?: HarnessPluginLifecycle,
): Promise<boolean> {
	if (!hooksEnabled || !pluginLifecycle) return false;
	try {
		await pluginLifecycle.sessionStart(
			{ ...ctx, enabled: true, tools: [] },
			source,
		);
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
	pluginLifecycle?: HarnessPluginLifecycle,
): Promise<void> {
	if (!hooksEnabled || !pluginLifecycle) return;
	try {
		await pluginLifecycle.sessionEnd(
			{ ...ctx, enabled: true, tools: [] },
			reason,
		);
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
	pluginLifecycle?: HarnessPluginLifecycle,
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
	if (!hooksEnabled || !pluginLifecycle) return hookResult;
	try {
		await pluginLifecycle.preCompact({ ...ctx, enabled: true, tools: [] });
	} catch (_e: unknown) {
		// must not block compaction
		console.error("[harness-session] emitPreCompact hookEvent failed:", _e);
	}
	return hookResult;
}

export async function emitPostCompact(
	hooksEnabled: boolean,
	ctx: HookContext,
	pluginLifecycle?: HarnessPluginLifecycle,
): Promise<void> {
	if (!hooksEnabled || !pluginLifecycle) return;
	try {
		await pluginLifecycle.postCompact({ ...ctx, enabled: true, tools: [] });
	} catch (_e: unknown) {
		// must not block compaction
		console.error("[harness-session] emitPostCompact failed:", _e);
	}
}
