// ── Plugin and MCP manager controllers ─────────────────────────────────────

import type { McpManagerAction } from "../../overlays/mcp-manager.ts";
import type { PluginManagerAction } from "../../overlays/plugin-manager.ts";
import type { OverlayHandlersCtx } from "./context.ts";

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
