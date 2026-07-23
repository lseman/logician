import { type Component, clampLineToWidth, visibleWidth } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";
import {
	renderListItem,
	renderSeparator,
	renderStatusLine,
	clampPopupLines,
	POPUP_FRAME_OVERHEAD,
	type ListItem,
} from "./popup-utils.ts";

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
	private selectedIndex = 0;
	private busyServerName: string | null = null;
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

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
		if (this.selectedIndex >= this.servers.length) {
			this.selectedIndex = Math.max(0, this.servers.length - 1);
		}
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

		if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
			return { type: "close" };
		}
		if (data === "\r" || data === "\n") {
			return { type: "close" };
		}
		if (data === "r" || data === "R") {
			return { type: "refresh" };
		}
		if (data === "\x1b[A" || data === "\x1bOA" || data === "k") {
			this.moveSelection(-1);
			return null;
		}
		if (data === "\x1b[B" || data === "\x1bOB" || data === "j") {
			this.moveSelection(1);
			return null;
		}
		if (data === "\x1b[5~") {
			this.moveSelection(-8);
			return null;
		}
		if (data === "\x1b[6~") {
			this.moveSelection(8);
			return null;
		}
		if (data === " ") {
			const server = this.servers[this.selectedIndex];
			return server ? { type: "toggle", server } : null;
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
		const lines: string[] = [];

		const headerFg = theme.fg("header", "");

		// ── Top rule ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		// ── Title row ──
		const titleText = "MCP Servers";
		const subtitleText = ` (${this.servers.length})`;
		const hintsText = " space toggle · r refresh · enter/esc close";
		const titleLine = `${titleText}${theme.fg("muted", "")}${subtitleText}${hintsText}`;
		const titleVisible = visibleWidth(titleLine);
		const titlePad = Math.max(0, innerWidth - titleVisible);
		lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);
		if (this.configPath) {
			lines.push(renderStatusLine(`Config: ${this.configPath}`, innerWidth));
		}

		// ── Separator ──
		lines.push(renderSeparator(popupWidth));

		// ── Server list ──
		if (!this.servers.length) {
			lines.push(
				renderStatusLine(
					"No MCP servers configured.",
					innerWidth,
					theme.fg("warning", ""),
				),
			);
		} else {
			const maxRows = 10;
			const start = Math.max(
				0,
				Math.min(
					this.selectedIndex - Math.floor(maxRows / 2),
					Math.max(0, this.servers.length - maxRows),
				),
			);
			const end = Math.min(this.servers.length, start + maxRows);
			if (start > 0) {
				lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
			}
			for (let i = start; i < end; i++) {
				const server = this.servers[i];
				const isSelected = i === this.selectedIndex;
				const typeIcon =
					server.type === "http" || server.type === "streamable-http"
						? "http"
						: "cmd";
				const toolText =
					server.toolCount > 0 ? `${server.toolCount} tool(s)` : "0 tools";
				const urlText = server.url
					? server.url.slice(0, 50)
					: server.command
						? server.command.split(" ").slice(0, 3).join(" ") + "..."
						: "-";
				const busy =
					this.busyServerName === server.serverName ? "  updating…" : "";

				const item: ListItem = {
					label: `${server.serverName} (${typeIcon})`,
					metadata: `${toolText} · ${urlText}${busy}`,
					selected: isSelected,
					statusDot:
						this.busyServerName === server.serverName
							? "yellow"
							: server.enabled
								? "green"
								: "gray",
				};

				lines.push(renderListItem(item, innerWidth));
			}
			if (end < this.servers.length) {
				lines.push(renderStatusLine(`↓ ${this.servers.length - end} more`, innerWidth));
			}
		}

		// ── Bottom bar ──
		lines.push(renderSeparator(popupWidth));
		const bottomText = this.message
			? this.message
			: "Toggle enables/disables MCP servers in config. Changes apply on next reconnect.";
		lines.push(renderStatusLine(bottomText, innerWidth));

		// ── Bottom rule ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.servers.length;
		if (!n) return;
		this.selectedIndex = (this.selectedIndex + delta + n) % n;
		this.invalidate();
	}
}
