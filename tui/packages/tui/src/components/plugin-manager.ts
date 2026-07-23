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

export interface PluginListItem {
	pluginId: string;
	name: string;
	version: string;
	enabled: boolean;
	installPath: string;
	hookCount: number;
	skillCount: number;
	onDisk: boolean;
}

export type PluginManagerAction =
	| { type: "toggle"; plugin: PluginListItem }
	| { type: "refresh" }
	| { type: "close" };

export class PluginManagerOverlay implements Component {
	public visible = false;
	private plugins: PluginListItem[] = [];
	private pluginsDir = "";
	private selectedIndex = 0;
	private busyPluginId: string | null = null;
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setSnapshot(snapshot: {
		pluginsDir?: string;
		plugins: Array<Record<string, unknown>>;
		sessionStartHooks?: Record<string, number>;
	}): void {
		const hooks = snapshot.sessionStartHooks || {};
		this.pluginsDir = snapshot.pluginsDir || "";
		this.plugins = snapshot.plugins.map((plugin) => {
			const pluginId = String(plugin.plugin_id || plugin.name || "");
			return {
				pluginId,
				name: String(plugin.name || pluginId),
				version: String(plugin.version || ""),
				enabled: Boolean(plugin.enabled),
				installPath: String(plugin.install_path || ""),
				hookCount: Number(hooks[pluginId] || 0),
				skillCount: Number(plugin.skill_count || 0),
				onDisk: plugin.on_disk !== false,
			};
		});
		if (this.selectedIndex >= this.plugins.length) {
			this.selectedIndex = Math.max(0, this.plugins.length - 1);
		}
		this.invalidate();
	}

	setBusy(pluginId: string | null): void {
		this.busyPluginId = pluginId;
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
		this.busyPluginId = null;
		this.invalidate();
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	handleInput(data: string): PluginManagerAction | null {
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
			const plugin = this.plugins[this.selectedIndex];
			return plugin ? { type: "toggle", plugin } : null;
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
		const titleText = "Plugins";
		const subtitleText = ` (${this.plugins.length})`;
		const hintsText = " space toggle · r refresh · enter/esc close";
		const titleLine = `${titleText}${theme.fg("muted", "")}${subtitleText}${hintsText}`;
		const titleVisible = visibleWidth(titleLine);
		const titlePad = Math.max(0, innerWidth - titleVisible);
		lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);
		if (this.pluginsDir) {
			lines.push(renderStatusLine(this.pluginsDir, innerWidth));
		}

		// ── Separator ──
		lines.push(renderSeparator(popupWidth));

		// ── Plugin list ──
		if (!this.plugins.length) {
			lines.push(
				renderStatusLine(
					"No plugins installed.",
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
					Math.max(0, this.plugins.length - maxRows),
				),
			);
			const end = Math.min(this.plugins.length, start + maxRows);
			if (start > 0) {
				lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
			}
			for (let i = start; i < end; i++) {
				const plugin = this.plugins[i];
				const isSelected = i === this.selectedIndex;
				const hookText = plugin.hookCount
					? `hooks:${plugin.hookCount}`
					: "hooks:-";
				const skillText = plugin.skillCount
					? `skills:${plugin.skillCount}`
					: "";
				const metaParts = [hookText];
				if (skillText) metaParts.push(skillText);
				const metaStr = metaParts.join(" · ");
				const diskText = plugin.onDisk ? "" : "  missing";
				const busy = this.busyPluginId === plugin.pluginId ? "  updating…" : "";
				const meta = `v${plugin.version || "?"} · ${metaStr}${diskText}${busy}`;

				const item: ListItem = {
					label: plugin.pluginId,
					metadata: meta,
					selected: isSelected,
					statusDot: !plugin.onDisk
						? "red"
						: this.busyPluginId === plugin.pluginId
							? "yellow"
							: plugin.enabled
								? "green"
								: "gray",
				};

				lines.push(renderListItem(item, innerWidth));
			}
			if (end < this.plugins.length) {
				lines.push(renderStatusLine(`↓ ${this.plugins.length - end} more`, innerWidth));
			}
		}

		// ── Bottom bar ──
		lines.push(renderSeparator(popupWidth));
		const bottomText = this.message
			? this.message
			: `Skills: ${this.plugins.reduce((s, p) => s + p.skillCount, 0)} total · Enabled plugins expose skills + hooks to Logician.`;
		lines.push(renderStatusLine(bottomText, innerWidth));

		// ── Bottom rule ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.plugins.length;
		if (!n) return;
		this.selectedIndex = (this.selectedIndex + delta + n) % n;
		this.invalidate();
	}
}
