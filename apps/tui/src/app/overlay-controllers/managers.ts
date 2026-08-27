// ── Plugin, MCP, and autoresearch dashboard manager controllers ────────────

import type { AutoresearchDashboardAction } from "../../overlays/autoresearch-dashboard.ts";
import type { McpManagerAction } from "../../overlays/mcp-manager.ts";
import type { PluginManagerAction } from "../../overlays/plugin-manager.ts";
import type { SessionTreeAction } from "../../overlays/session-tree.ts";
import type { Turn } from "@logician/log-runtime/sessions";
import type { OverlayHandlersCtx } from "./context.ts";
import { turnsToMessages } from "../session/messages.ts";

// ── Plugin manager ───────────────────────────────────────────────────────

export async function openPluginManager(
	ctx: OverlayHandlersCtx,
): Promise<void> {
	ctx.statusPanel.update({ phase: "plugins" });
	// Show the popup immediately with a loading message rather than waiting on
	// the disk-scanning snapshot fetch below — otherwise Enter appears to do
	// nothing until the async round-trip resolves.
	ctx.pluginManager.setMessage("Loading plugins...");
	ctx.pluginManager.show();
	ctx.tui.renderNow();
	try {
		await yieldToRenderer();
		const snapshot = await ctx.bridge.getPluginSnapshot();
		ctx.pluginManager.setSnapshot({
			pluginsDir: String(snapshot.plugins_dir || ""),
			plugins: snapshot.plugins || [],
			sessionStartHooks: snapshot.session_start_hooks || {},
		});
		ctx.pluginManager.setMessage(
			"Space toggles enabled state in the Claude plugin registry.",
		);
	} catch (e: unknown) {
		ctx.pluginManager.setMessage(
			`Plugins error: ${e instanceof Error ? e.message : String(e)}`,
		);
	} finally {
		ctx.statusPanel.update({ phase: "ready" });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	}
}

export function handlePluginManagerAction(
	ctx: OverlayHandlersCtx,
	action: PluginManagerAction,
): void {
	if (action.type === "close") {
		// The plugin manager is registered once in buildLayout().
		ctx.pluginManager.hide();
		return;
	}
	if (action.type === "refresh") {
		void openPluginManager(ctx);
		return;
	}

	const plugin = action.plugin;
	const nextEnabled = !plugin.enabled;
	ctx.pluginManager.setBusy(plugin.pluginId);
	ctx.pluginManager.setMessage(
		`${nextEnabled ? "Enabling" : "Disabling"} ${plugin.pluginId}...`,
	);
	ctx.tui.requestRender();
	void ctx.bridge
		.setPluginEnabled(plugin.pluginId, nextEnabled)
		.then(async result => {
			ctx.pluginManager.setMessage(
				String(result.message || `${plugin.pluginId} updated.`),
			);
			const snapshot = await ctx.bridge.getPluginSnapshot();
			ctx.pluginManager.setSnapshot({
				pluginsDir: String(snapshot.plugins_dir || ""),
				plugins: snapshot.plugins || [],
				sessionStartHooks: snapshot.session_start_hooks || {},
			});
		})
		.catch((e: unknown) => {
			ctx.pluginManager.setMessage(
				`Plugin update failed: ${e instanceof Error ? e.message : String(e)}`,
			);
		})
		.finally(() => {
			ctx.pluginManager.setBusy(null);
			ctx.statusPanel.update({ phase: "ready" });
			ctx.tui.requestRender();
		});
}

// ── MCP manager ───────────────────────────────────────────────────────

export async function openMcpManager(ctx: OverlayHandlersCtx): Promise<void> {
	ctx.statusPanel.update({ phase: "mcp" });
	// Show the popup immediately with a loading message rather than waiting on
	// the snapshot fetch below — otherwise Enter appears to do nothing until
	// the async round-trip resolves.
	ctx.mcpManager.setMessage("Loading MCP servers...");
	ctx.mcpManager.show();
	ctx.tui.renderNow();
	try {
		await yieldToRenderer();
		const snapshot = await ctx.bridge.getMcpSnapshot();
		ctx.mcpManager.setSnapshot({
			configPath: snapshot.configPath,
			servers: snapshot.servers.map(s => ({
				server_name: s.serverName,
				server: s.server,
				url: s.server.url || "",
				command: s.server.command || "",
				type: s.server.type || (s.server.url ? "http" : "stdio"),
				enabled: s.enabled,
			})),
			loadedServers: snapshot.loadedServers,
			errors: snapshot.errors,
		});
		ctx.mcpManager.setMessage(
			"Space toggles enabled state in the MCP config file.",
		);
	} catch (e: unknown) {
		ctx.mcpManager.setMessage(
			`MCP error: ${e instanceof Error ? e.message : String(e)}`,
		);
	} finally {
		ctx.statusPanel.update({ phase: "ready" });
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
	}
}

/** Let the immediate loading frame reach the terminal before discovery work. */
function yieldToRenderer(): Promise<void> {
	return new Promise(resolve => setImmediate(resolve));
}

export function handleMcpManagerAction(
	ctx: OverlayHandlersCtx,
	action: McpManagerAction,
): void {
	if (action.type === "close") {
		// The MCP manager is registered once in buildLayout(). Keep it in the
		// overlay stack so a later `/mcp list` can show the same component.
		ctx.mcpManager.hide();
		return;
	}
	if (action.type === "refresh") {
		void openMcpManager(ctx);
		return;
	}

	const server = action.server;
	const nextEnabled = !server.enabled;
	ctx.mcpManager.setBusy(server.serverName);
	ctx.mcpManager.setMessage(
		`${nextEnabled ? "Enabling" : "Disabling"} ${server.serverName}...`,
	);
	ctx.tui.requestRender();
	void ctx.bridge
		.setMcpServerEnabled(server.serverName, nextEnabled)
		.then(async result => {
			ctx.mcpManager.setMessage(result.message);
			const snapshot = await ctx.bridge.getMcpSnapshot();
			ctx.mcpManager.setSnapshot({
				configPath: snapshot.configPath,
				servers: snapshot.servers.map(s => ({
					server_name: s.serverName,
					server: s.server,
					url: s.server.url || "",
					command: s.server.command || "",
					type: s.server.type || (s.server.url ? "http" : "stdio"),
					enabled: s.enabled,
				})),
				loadedServers: snapshot.loadedServers,
				errors: snapshot.errors,
			});
		})
		.catch((e: unknown) => {
			ctx.mcpManager.setMessage(
				`MCP update failed: ${e instanceof Error ? e.message : String(e)}`,
			);
		})
		.finally(() => {
			ctx.mcpManager.setBusy(null);
			ctx.statusPanel.update({ phase: "ready" });
			ctx.tui.requestRender();
		});
}

// ── Autoresearch dashboard ──────────────────────────────────────────────

/** Opens the fullscreen dashboard (Ctrl+A). All data is local/sync
 * (AutoresearchSession.getDashboardData() just reads in-memory state), so
 * unlike plugin/MCP managers there's no loading round-trip to show first. */
export function openAutoresearchDashboard(ctx: OverlayHandlersCtx): void {
	ctx.autoresearchDashboard.show();
	// This overlay is pre-registered once at startup and never re-pushed, so
	// its stack position is frozen below overlays registered after it (e.g.
	// plugin/MCP managers). Bring it to front on every open or it can render
	// hidden underneath whichever of those is currently visible.
	ctx.tui.bringToFront(ctx.autoresearchDashboard);
	ctx.tui.requestRender();
}

export function handleAutoresearchDashboardAction(
	ctx: OverlayHandlersCtx,
	action: AutoresearchDashboardAction,
): void {
	if (action.type === "close") {
		ctx.autoresearchDashboard.hide();
		ctx.tui.requestRender();
	}
}

// ── Session tree ──────────────────────────────────────────────────────────

export function openSessionTree(ctx: OverlayHandlersCtx): void {
	ctx.statusPanel.update({ phase: "sessions" });
	ctx.sessionTree.show();
	const overlay = ctx.tui.showOverlay(ctx.sessionTree, {
		anchor: "aboveInput",
		align: "left",
		maxHeight: 18,
	});
	overlay.focus();
}

export function handleSessionTreeAction(
	ctx: OverlayHandlersCtx,
	action: SessionTreeAction,
): void {
	if (action.type === "close") {
		ctx.sessionTree.hide();
		ctx.tui.removeOverlay(ctx.sessionTree);
		ctx.statusPanel.update({ phase: "ready" });
		ctx.tui.requestRender();
		return;
	}
	if (action.type === "navigate") {
		// Navigate to the selected entry in the session tree.
		const sessionId = ctx.sessionService.getCurrentSessionId();
		if (!sessionId) return;
		const turns: Turn[] = ctx.sessionService.checkoutTurn(sessionId, action.entryId);
		ctx.transcript.loadTurns(turns);
		ctx.transcriptDisplay.setTurns(turns);
		ctx.bridge.restoreHistory(turnsToMessages(turns));
		ctx.statusPanel.update({ phase: "ready" });
		ctx.tui.removeOverlay(ctx.sessionTree);
		ctx.tui.requestRender();
	}
}
