import type { Component } from "../terminal/core.ts";
import {
	clampPopupLines,
	type ListItem,
	POPUP_FRAME_OVERHEAD,
	parsePopupListNav,
	renderListItem,
	renderListPopupBody,
	renderListPopupFrame,
} from "./popup-utils.ts";
import { SelectorController } from "./selector-controller.ts";

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

export class McpManagerOverlay implements Component {
	public visible = false;
	private servers: McpServerItem[] = [];
	private configPath = "";
	private _selection = new SelectorController();
	private busyServerName: string | null = null;
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	/** @internal Exposed for tests. */
	get selection(): SelectorController { return this._selection; }

	setSnapshot(snapshot: {
		configPath?: string;
		servers: Array<Record<string, unknown>>;
		loadedServers?: Record<string, unknown>;
		errors?: string[];
	}): void {
		this.configPath = snapshot.configPath || "";
		this.servers = snapshot.servers.map(server => {
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
		this._selection.set(this._selection.index, this.servers.length);
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
			const server = this.servers[this._selection.index];
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
		this.cachedLines = null;
	}

	render(width: number): string[] {
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}
		this.cachedWidth = width;

		if (!this.visible) return [];

		const popupWidth = Math.max(1, width);
		const innerWidth = Math.max(1, popupWidth - POPUP_FRAME_OVERHEAD);

		const bodyLines = renderListPopupBody(
			this.servers,
			this.selection,
			innerWidth,
			10,
			(server, i) => {
				const typeIcon =
					server.type === "http" || server.type === "streamable-http"
						? "http"
						: "cmd";
				const toolText =
					server.toolCount > 0 ? `${server.toolCount} tool(s)` : "0 tools";
				const urlText = server.url
					? server.url.slice(0, 50)
					: server.command
						? `${server.command.split(" ").slice(0, 3).join(" ")}...`
						: "-";
				const busy =
					this.busyServerName === server.serverName ? "  updating…" : "";

				const item: ListItem = {
					label: `${server.serverName} (${typeIcon})`,
					metadata: `${toolText} · ${urlText}${busy}`,
					selected: i === this._selection.index,
					statusDot:
						this.busyServerName === server.serverName
							? "yellow"
							: server.enabled
								? "green"
								: "gray",
				};

				return renderListItem(item, innerWidth);
			},
			"No MCP servers configured.",
		);

		const lines = renderListPopupFrame({
			popupWidth,
			innerWidth,
			title: "MCP Servers",
			subtitle: ` (${this.servers.length})`,
			hints: " space toggle · r refresh · enter/esc close",
			extraHeaderLines: this.configPath
				? [`Config: ${this.configPath}`]
				: undefined,
			bodyLines,
			bottomText:
				this.message ||
				"Toggle enables/disables MCP servers in config. Changes apply on next reconnect.",
		});

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		this._selection.move(delta, this.servers.length);
		this.invalidate();
	}
}
