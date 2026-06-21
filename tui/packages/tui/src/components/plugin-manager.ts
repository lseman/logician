import { type Component, clampLineToWidth, visibleWidth } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const getHeader = (): string => theme.fg("header", "");
const getSelected = (): string => theme.fg("selected", "");
const getMuted = (): string => theme.fg("muted", "");
const getWarn = (): string => theme.fg("active", "");

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

		const overlayWidth = Math.max(48, Math.min(width, 110));
		const innerWidth = Math.max(1, overlayWidth - 4);
		const lines: string[] = [];

		lines.push(`${getHeader()}┌${"─".repeat(overlayWidth - 2)}┐${RESET}`);
		lines.push(
			boxLine(
				`${BOLD}Plugins${RESET}${DIM} (${this.plugins.length})${RESET}`,
				"space toggle · r refresh · enter/esc close",
				innerWidth,
			),
		);
		if (this.pluginsDir) {
			lines.push(boxLine(`${DIM}${this.pluginsDir}${RESET}`, "", innerWidth));
		}
		lines.push(`${getHeader()}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);

		if (!this.plugins.length) {
			lines.push(
				boxLine(`${getMuted()}No plugins installed.${RESET}`, "", innerWidth),
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
				lines.push(
					boxLine(`${getMuted()}↑ ${start} more${RESET}`, "", innerWidth),
				);
			}
			for (let i = start; i < end; i++) {
				const plugin = this.plugins[i];
				const selected = i === this.selectedIndex;
				const checkbox = plugin.enabled ? "[x]" : "[ ]";
				const cursor = selected ? "▸" : " ";
				const hookText = plugin.hookCount
					? `hooks:${plugin.hookCount}`
					: "hooks:-";
				const skillText = plugin.skillCount
					? `skills:${plugin.skillCount}`
					: "";
				const metaParts = [hookText];
				if (skillText) metaParts.push(skillText);
				const metaStr = metaParts.join(" · ");
				const diskText = plugin.onDisk ? "" : ` ${getWarn()}missing${RESET}`;
				const busy =
					this.busyPluginId === plugin.pluginId
						? ` ${DIM}updating...${RESET}`
						: "";
				const name = selected
					? `${getSelected()}${BOLD}${plugin.pluginId}${RESET}`
					: plugin.pluginId;
				const meta = `${DIM}v${plugin.version || "?"} · ${metaStr}${RESET}${diskText}${busy}`;
				lines.push(boxLine(`${cursor} ${checkbox} ${name}`, meta, innerWidth));
			}
			if (end < this.plugins.length) {
				lines.push(
					boxLine(
						`${getMuted()}↓ ${this.plugins.length - end} more${RESET}`,
						"",
						innerWidth,
					),
				);
			}
		}

		lines.push(`${getHeader()}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);
		lines.push(
			boxLine(
				this.message
					? `${DIM}${this.message}${RESET}`
					: `${getMuted()}Skills: ${this.plugins.reduce((s, p) => s + p.skillCount, 0)} total · Enabled plugins expose skills + hooks to Logician.${RESET}`,
				"",
				innerWidth,
			),
		);
		lines.push(`${getHeader()}└${"─".repeat(overlayWidth - 2)}┘${RESET}`);

		this.cachedLines = lines.map((line) => clampLineToWidth(line, width));
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.plugins.length;
		if (!n) return;
		this.selectedIndex = (this.selectedIndex + delta + n) % n;
		this.invalidate();
	}
}

function boxLine(left: string, right: string, width: number): string {
	const leftWidth = visibleWidth(left);
	const rightWidth = visibleWidth(right);
	const gap = Math.max(1, width - leftWidth - rightWidth);
	const content = right ? `${left}${" ".repeat(gap)}${right}` : left;
	const pad = Math.max(0, width - visibleWidth(content));
	return `${getHeader()}│${RESET} ${content}${" ".repeat(pad)} ${getHeader()}│${RESET}`;
}
