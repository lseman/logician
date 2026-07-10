// ── ModelSelectorOverlay ────────────────────────────────────────────────────────
// Overlay for selecting an active model from the configured list.
// Pattern: list, select, confirm, close — mirrors ThemeSelector/ReasonerSelector.

import {
	type Component,
	clampLineToWidth,
	visibleWidth,
} from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const getHeader = (): string => theme.fg("header", "");
const getSelected = (): string => theme.fg("selected", "");
const getMuted = (): string => theme.fg("muted", "");
const getActive = (): string => theme.fg("active", "");

export interface ModelInfo {
	id: string;
	name: string;
	active: boolean;
	url?: string;
}

export type ModelSelectorAction =
	| { type: "select"; model: ModelInfo }
	| { type: "close" };

export class ModelSelectorOverlay implements Component {
	public visible = false;
	private models: ModelInfo[] = [];
	private selectedIndex = 0;
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setModels(models: ModelInfo[]): void {
		this.models = models;
		if (this.selectedIndex >= this.models.length) {
			this.selectedIndex = Math.max(0, this.models.length - 1);
		}
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
		this.invalidate();
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	handleInput(data: string): ModelSelectorAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
			return { type: "close" };
		}
		if (data === "\r" || data === "\n") {
			const model = this.models[this.selectedIndex];
			return model ? { type: "select", model } : { type: "close" };
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
				`${BOLD}Model${RESET}${DIM} (${this.models.length})${RESET}`,
				"↑↓ select · enter confirm · esc close",
				innerWidth,
			),
		);
		lines.push(`${getHeader()}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);

		if (!this.models.length) {
			lines.push(
				boxLine(
					`${getMuted()}No models configured. Add "models" array to settings.json.${RESET}`,
					"",
					innerWidth,
				),
			);
		} else {
			const maxRows = 10;
			const start = Math.max(
				0,
				Math.min(
					this.selectedIndex - Math.floor(maxRows / 2),
					Math.max(0, this.models.length - maxRows),
				),
			);
			const end = Math.min(this.models.length, start + maxRows);
			if (start > 0) {
				lines.push(
					boxLine(`${getMuted()}↑ ${start} more${RESET}`, "", innerWidth),
				);
			}
			for (let i = start; i < end; i++) {
				const m = this.models[i];
				const selected = i === this.selectedIndex;
				const cursor = selected ? "▸" : " ";
				const activeMark = m.active ? `${getActive()}● active${RESET}` : "";
				const name = selected
					? `${getSelected()}${BOLD}${m.name}${RESET}`
					: m.name;
				const desc = `${DIM}${m.url ?? m.id}${RESET}`;
				const meta = activeMark ? `${desc}  ${activeMark}` : desc;
				lines.push(boxLine(`${cursor} ${name}`, meta, innerWidth));
			}
			if (end < this.models.length) {
				lines.push(
					boxLine(
						`${getMuted()}↓ ${this.models.length - end} more${RESET}`,
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
					: `${getMuted()}Select a model for this session.${RESET}`,
				"",
				innerWidth,
			),
		);
		lines.push(`${getHeader()}└${"─".repeat(overlayWidth - 2)}┘${RESET}`);

		this.cachedLines = lines.map((line) => clampLineToWidth(line, width));
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.models.length;
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
