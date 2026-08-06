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
	private selection = new SelectorController();
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
		this.plugins = snapshot.plugins.map(plugin => {
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
		this.selection.set(this.selection.index, this.plugins.length);
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

		if (data === "r" || data === "R") {
			return { type: "refresh" };
		}
		if (data === " ") {
			const plugin = this.plugins[this.selection.index];
			return plugin ? { type: "toggle", plugin } : null;
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
			this.plugins,
			this.selection,
			innerWidth,
			10,
			(plugin, i) => {
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
					selected: i === this.selection.index,
					statusDot: !plugin.onDisk
						? "red"
						: this.busyPluginId === plugin.pluginId
							? "yellow"
							: plugin.enabled
								? "green"
								: "gray",
				};

				return renderListItem(item, innerWidth);
			},
			"No plugins installed.",
		);

		const lines = renderListPopupFrame({
			popupWidth,
			innerWidth,
			title: "Plugins",
			subtitle: ` (${this.plugins.length})`,
			hints: " space toggle · r refresh · enter/esc close",
			extraHeaderLines: this.pluginsDir ? [this.pluginsDir] : undefined,
			bodyLines,
			bottomText:
				this.message ||
				`Skills: ${this.plugins.reduce((s, p) => s + p.skillCount, 0)} total · Enabled plugins expose skills + hooks to Logician.`,
		});

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		this.selection.move(delta, this.plugins.length);
		this.invalidate();
	}
}
