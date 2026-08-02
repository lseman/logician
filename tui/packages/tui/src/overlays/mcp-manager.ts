import { SelectorController } from "./selector-controller.ts";
import type { InkListOverlayModel } from "./ink-overlay-model.ts";
import { parsePopupListNav } from "./popup-utils.ts";

export interface McpServerItem {
	serverName: string;
	url: string;
	type: "stdio" | "http" | "streamable-http";
	command?: string;
	enabled: boolean;
	toolCount: number;
	configPath: string;
}

export type McpManagerAction =
	| { type: "toggle"; server: McpServerItem }
	| { type: "refresh" }
	| { type: "close" };

export class McpManagerOverlay {
	public visible = false;
	private servers: McpServerItem[] = [];
	private configPath = "";
	private selection = new SelectorController();
	private busyServerName: string | null = null;
	private message = "";

	setSnapshot(snapshot: {
		configPath?: string;
		servers: Array<Record<string, unknown>>;
		loadedServers?: Record<string, unknown>;
		errors?: string[];
	}): void {
		this.configPath = snapshot.configPath || "";
		this.servers = snapshot.servers.map((server) => {
			const serverName = String(server.server_name || server.name || "");
			const loadedServers = snapshot.loadedServers || {};
			const toolCount = Number(
				(loadedServers[serverName] as { toolCount?: number })?.toolCount || 0,
			);
			return {
				serverName,
				url: String(server.url || server.command || ""),
				type: (server.type || (server.url ? "http" : "stdio")) as
					| "stdio"
					| "http"
					| "streamable-http",
				command: String(server.command || ""),
				enabled: server.enabled !== false,
				toolCount,
				configPath: snapshot.configPath || "",
			};
		});
		this.selection.set(this.selection.index, this.servers.length);
		this.invalidate();
	}

	setBusy(serverName: string | null): void {
		this.busyServerName = serverName;
		this.invalidate();
	}

	setMessage(message: string): void {
		this.message = message;
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	hide(): void {
		this.visible = false;
		this.busyServerName = null;
		this.invalidate();
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	handleInput(data: string): McpManagerAction | null {
		if (!this.visible) return null;

		if (data === "r" || data === "R") {
			return { type: "refresh" };
		}
		if (data === " ") {
			const server = this.servers[this.selection.index];
			return server ? { type: "toggle", server } : null;
		}

		const nav = parsePopupListNav(data);
		if (nav?.type === "close" || nav?.type === "confirm") {
			return { type: "close" };
		}
		if (nav?.type === "move") {
			this.moveSelection(nav.delta);
		}
		return null;
	}

	invalidate(): void {
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		return {
			kind: "list",
			title: "MCP Servers",
			subtitle: ` (${this.servers.length})`,
			hints: "space toggle · r refresh · enter/esc close",
			headerLines: this.configPath ? [`Config: ${this.configPath}`] : undefined,
			items: this.servers.map((server, index) => {
				const endpoint = server.url || server.command || "-";
				const busy = this.busyServerName === server.serverName ? " · updating…" : "";
				return {
					label: `${server.serverName} (${server.type === "stdio" ? "cmd" : "http"})`,
					metadata: `${server.toolCount === 0 ? "0 tools" : `${server.toolCount} tool(s)`} · ${endpoint.slice(0, 50)}${busy}`,
					selected: index === this.selection.index,
					current: server.enabled,
				};
			}),
			emptyText: "No MCP servers configured.",
			footer: this.message || "Changes apply on next reconnect.",
			selectedIndex: this.selection.index,
		};
	}

	private moveSelection(delta: number): void {
		this.selection.move(delta, this.servers.length);
		this.invalidate();
	}
}
