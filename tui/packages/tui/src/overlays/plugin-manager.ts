import { SelectorController } from "./selector-controller.ts";
import type { InkListOverlayModel } from "./ink-overlay-model.ts";
import { parsePopupListNav } from "./popup-utils.ts";

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

export class PluginManagerOverlay {
	public visible = false;
	private plugins: PluginListItem[] = [];
	private pluginsDir = "";
	private selection = new SelectorController();
	private busyPluginId: string | null = null;
	private message = "";

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
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		return {
			kind: "list",
			title: "Plugins",
			subtitle: ` (${this.plugins.length})`,
			hints: "space toggle · r refresh · enter/esc close",
			headerLines: this.pluginsDir ? [this.pluginsDir] : undefined,
			items: this.plugins.map((plugin, index) => {
				const meta = [
					`v${plugin.version || "?"}`,
					plugin.hookCount ? `hooks:${plugin.hookCount}` : "hooks:-",
					plugin.skillCount ? `skills:${plugin.skillCount}` : "",
					plugin.onDisk ? "" : "missing",
					this.busyPluginId === plugin.pluginId ? "updating…" : "",
				].filter(Boolean).join(" · ");
				return {
					label: plugin.pluginId,
					metadata: meta,
					selected: index === this.selection.index,
					current: plugin.enabled,
				};
			}),
			emptyText: "No plugins installed.",
			footer: this.message || `Skills: ${this.plugins.reduce((sum, plugin) => sum + plugin.skillCount, 0)} total`,
			selectedIndex: this.selection.index,
		};
	}

	private moveSelection(delta: number): void {
		this.selection.move(delta, this.plugins.length);
		this.invalidate();
	}
}
