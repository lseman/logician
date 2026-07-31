// ── Async slash-command helpers ────────────────────────────────────────────

import { getReasonerMeta } from "@logician/agent-capabilities/reasoning";
import { formatContextSize } from "@logician/coding-agent";
import { getAvailableThemes } from "../../terminal/theme.ts";
import { getGitVersion } from "../git-status.ts";
import type { SlashCommandsCtx } from "./context.ts";

export async function handleStatus(ctx: SlashCommandsCtx): Promise<void> {
	try {
		const state = await ctx.bridge.getState();
		const runtime = (state.runtime_state ?? {}) as {
			phase?: string;
			isStreaming?: boolean;
			pendingToolCalls?: string[];
			retry?: { attempt?: number; maxRetries?: number };
			lastError?: string;
			outcome?: { status?: string; summary?: string; source?: string };
			lastTurnDurationMs?: number;
			lastRunDurationMs?: number;
		};
		const lines = [
			`Agent: ${state.agent_name || "unknown"}`,
			`Model: ${state.model || "unknown"}`,
			`Base URL: ${state.base_url || "unknown"}`,
			`Project: ${getGitVersion() || "-"}`,
			`Tools: ${(state.tools as string[])?.length || 0} loaded`,
			`MCP: ${state.mcp_servers || 0} server(s), ${state.mcp_tools || 0} tool(s)`,
			`Context: ${formatContextSize(
				Number(state.context_tokens || 0),
				Number(state.context_max_tokens || 0) || undefined,
			)}`,
			`Runtime: ${runtime.phase || "idle"}${runtime.isStreaming ? " (streaming)" : ""}`,
			`Active tools: ${runtime.pendingToolCalls?.length || 0}`,
			...(runtime.lastTurnDurationMs !== undefined
				? [`Last turn: ${runtime.lastTurnDurationMs}ms`]
				: []),
			...(runtime.lastRunDurationMs !== undefined
				? [`Last run: ${runtime.lastRunDurationMs}ms`]
				: []),
			...(runtime.retry
				? [
						`Retry: ${runtime.retry.attempt || 0}/${runtime.retry.maxRetries || 0}`,
					]
				: []),
			...(runtime.lastError ? [`Last error: ${runtime.lastError}`] : []),
			...(runtime.outcome
				? [
						`Outcome: ${runtime.outcome.status || "unknown"} (${runtime.outcome.source || "unknown"})`,
						...(runtime.outcome.summary
							? [`Outcome summary: ${runtime.outcome.summary}`]
							: []),
					]
				: []),
			`Hooks: ${state.hooks_enabled === false ? "disabled" : "enabled"}`,
			`Hook transcript: ${state.hook_transcript_path || "-"}`,
			`Config: ${state.config_path || "-"}`,
			`Connected: ${state.connected !== false}`,
		];
		const mcpErrors = Array.isArray(state.mcp_errors)
			? state.mcp_errors
					.map((item) => String(item || "").trim())
					.filter(Boolean)
			: [];
		if (mcpErrors.length) {
			lines.push("", "MCP errors:", ...mcpErrors.map((err) => `- ${err}`));
		}
		ctx.transcript.addSystemMessage(lines.join("\n"));
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	} catch (e: unknown) {
		ctx.transcript.addSystemMessage(
			`Status error: ${e instanceof Error ? e.message : String(e)}`,
		);
		ctx.tui.requestRender();
	}
}

export async function handlePlugins(ctx: SlashCommandsCtx, args: string): Promise<void> {
	try {
		const normalized = args.trim().toLowerCase();
		if (!normalized || normalized === "list") {
			await ctx.openPluginManager();
			return;
		}
		const result = await ctx.bridge.runPluginCommand(args);
		ctx.transcript.addSystemMessage(result);
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	} catch (e: unknown) {
		ctx.transcript.addSystemMessage(
			`Plugins error: ${e instanceof Error ? e.message : String(e)}`,
		);
		ctx.tui.requestRender();
	}
}

export async function handleMcp(ctx: SlashCommandsCtx, args: string): Promise<void> {
	try {
		const normalized = args.trim().toLowerCase();
		if (normalized === "list" || normalized === "") {
			const snapshot = await ctx.bridge.getMcpSnapshot();
			if (snapshot.servers.length === 0) {
				ctx.transcript.addSystemMessage("No MCP servers configured.");
			} else {
				const lines = snapshot.servers.map((s) => {
					const status = s.enabled ? "✓" : "✗";
					const serverType = s.server.url ? "http" : "stdio";
					const error = s.error ? ` (${s.error})` : "";
					return `  ${status} ${s.serverName}  [${serverType}]  tools:${s.toolCount}${error}`;
				});
				lines.unshift(
					`MCP servers (${snapshot.servers.length} configured, ${Object.keys(snapshot.loadedServers).length} loaded):`,
				);
				ctx.transcript.addSystemMessage(lines.join("\n"));
			}
			ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
			ctx.tui.requestRender();
			return;
		}
		await ctx.openMcpManager();
	} catch (e: unknown) {
		ctx.transcript.addSystemMessage(
			`MCP error: ${e instanceof Error ? e.message : String(e)}`,
		);
	}
}

export async function handleReasoner(ctx: SlashCommandsCtx, args: string): Promise<void> {
	try {
		const normalized = args.trim().toLowerCase();
		if (!normalized || normalized === "list") {
			await ctx.openReasonerSelector();
			return;
		}
		// Direct set: /reasoner ssr, /reasoner none, etc.
		// reasoner removed;
		const meta = getReasonerMeta(normalized);
		const label = meta ? meta.name : normalized;
		ctx.transcript.addSystemMessage(`Reasoning mode: ${label}`);
		ctx.statusPanel.update({ reasoner: normalized });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	} catch (e: unknown) {
		ctx.transcript.addSystemMessage(
			`Reasoner error: ${e instanceof Error ? e.message : String(e)}`,
		);
		ctx.tui.requestRender();
	}
}

export async function handleTheme(ctx: SlashCommandsCtx, args: string): Promise<void> {
	try {
		const normalized = args.trim().toLowerCase();
		if (!normalized || normalized === "list") {
			await ctx.openThemeSelector();
			return;
		}
		// Direct set: /theme dark, /theme light, etc.
		const ok = ctx.setThemeByName(normalized);
		if (ok) {
			ctx.transcript.addSystemMessage(`Theme: ${normalized}`);
		} else {
			const available = getAvailableThemes();
			ctx.transcript.addSystemMessage(
				`Unknown theme "${normalized}". Available: ${available.join(", ")}`,
			);
		}
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	} catch (e: unknown) {
		ctx.transcript.addSystemMessage(
			`Theme error: ${e instanceof Error ? e.message : String(e)}`,
		);
		ctx.tui.requestRender();
	}
}
