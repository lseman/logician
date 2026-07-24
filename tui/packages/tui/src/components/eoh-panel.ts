// ── EoH (Evolution of Heuristics) Panel ──────────────────────────────────────
// Interactive TUI overlay for controlling EoH evolution.
// Shows population, best heuristic, and provides action buttons.

import { type Component, clampLineToWidth, visibleWidth, RESET, BOLD, DIM } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";

// ── EoH data shapes ──────────────────────────────────────────────────────────

export interface EohHeuristicEntry {
	id: string;
	thought: string;
	code: string;
	fitness: number;
	generation: number;
	createdBy: string;
}

export interface EohPanelState {
	running: boolean;
	generation: number;
	totalLLMCalls: number;
	populationSize: number;
	bestFitness: number;
	meanFitness: number;
	worstFitness: number;
	problemName: string;
	recentLog: string[];
	bestHeuristic: {
		thought: string;
		code: string;
		fitness: number;
		generation: number;
		createdBy: string;
	} | null;
	population: EohHeuristicEntry[];
}

export type EohPanelAction =
	| { type: "start"; generations?: number }
	| { type: "stop" }
	| { type: "best" }
	| { type: "toggle-population" }
	| { type: "toggle-details" }
	| { type: "close" };

// ── Constants ─────────────────────────────────────────────────────────────────

const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const BLUE = "\x1b[34m";
const CYAN = "\x1b[36m";
const MAGENTA = "\x1b[35m";
const RED = "\x1b[31m";
const MAX_VISIBLE_POP = 10;
const MAX_VISIBLE_LOG = 5;

const getHeaderColor = (): string => theme.fg("header", "");
const getSelectedColor = (): string => theme.fg("selected", "");

// ── EoHPanelOverlay ──────────────────────────────────────────────────────────

export class EohPanelOverlay implements Component {
	public visible = false;
	private state: EohPanelState = {
		running: false,
		generation: 0,
		totalLLMCalls: 0,
		populationSize: 0,
		bestFitness: 0,
		meanFitness: 0,
		worstFitness: 0,
		problemName: "Online Bin Packing",
		recentLog: [],
		bestHeuristic: null,
		population: [],
	};
	private selectedIndex = 0;
	private showPopulation = false;
	private showDetails = false;
	private selectedHeuristic: EohHeuristicEntry | null = null;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private onAction?: (action: EohPanelAction) => void;
	private message = "";

	setState(state: EohPanelState): void {
		this.state = state;
		this.selectedIndex = 0;
		this.selectedHeuristic = null;
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

	setOnAction(cb: (action: EohPanelAction) => void): void {
		this.onAction = cb;
	}

	handleInput(data: string): void {
		if (data === "\x1b" || data === "\x03") {
			this.onAction?.({ type: "close" });
			return;
		}

		if (data === "\r" || data === "\n") {
			this._submit();
			return;
		}

		if (data === "\x1b[A" || data === "\x1bOA") {
			this.selectedIndex = Math.max(0, this.selectedIndex - 1);
			this.invalidate();
			return;
		}

		if (data === "\x1b[B" || data === "\x1bOB") {
			this.selectedIndex = Math.min(this._menuCount() - 1, this.selectedIndex + 1);
			this.invalidate();
			return;
		}

		// 1-4 shortcuts for menu items
		if (data === "1" || data === "2" || data === "3" || data === "4") {
			this.selectedIndex = parseInt(data) - 1;
			this._submit();
			return;
		}

		if (data === "p" || data === "P") {
			this.showPopulation = !this.showPopulation;
			this.invalidate();
			return;
		}

		if (data === "d" || data === "D") {
			this.showDetails = !this.showDetails;
			this.invalidate();
			return;
		}

		if (data === "q" || data === "Q") {
			this.onAction?.({ type: "close" });
			return;
		}

		// Tab — toggle population
		if (data === "\t") {
			this.showPopulation = !this.showPopulation;
			this.invalidate();
			return;
		}
	}

	private _menuCount(): number {
		let count = 0;
		if (!this.state.running) count += 1; // start
		if (this.state.running) count += 1; // stop
		count += 1; // best
		count += 1; // close
		return count;
	}

	private _submit(): void {
		let action: EohPanelAction;
		if (!this.state.running && this.selectedIndex === 0) {
			action = { type: "start" };
		} else if (this.state.running && this.selectedIndex === 0) {
			action = { type: "stop" };
		} else {
			const idx = this.state.running ? this.selectedIndex - 1 : this.selectedIndex;
			if (idx === 1) action = { type: "best" };
			else action = { type: "close" };
		}
		this.onAction?.(action);
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

		const contentWidth = Math.max(1, width - 2);
		const lines: string[] = [];
		const s = this.state;

		// ── Header ───────────────────────────────────────────────────────────
		const headerColor = getHeaderColor();
		const statusIcon = s.running ? `${GREEN}●${RESET}` : `${DIM}○${RESET}`;
		const statusText = s.running ? `${GREEN}Running${RESET}` : `${DIM}Idle${RESET}`;
		lines.push(
			` ${headerColor}EoH: Evolution of Heuristics${RESET}  ${statusIcon} ${statusText}`,
		);
		lines.push(` ${DIM}Problem: ${s.problemName}${RESET}`);
		lines.push("");

		// ── Stats row ────────────────────────────────────────────────────────
		if (s.populationSize > 0) {
			const bestColor = s.bestFitness >= 1.0 ? GREEN : CYAN;
			lines.push(
				` ${DIM}Gen:${RESET} ${s.generation}  ${DIM}LLM:${RESET} ${s.totalLLMCalls}  ${DIM}Pop:${RESET} ${s.populationSize}  ${DIM}Best:${RESET} ${bestColor}${s.bestFitness.toFixed(4)}${RESET}  ${DIM}Mean:${RESET} ${s.meanFitness.toFixed(4)}  ${DIM}Worst:${RESET} ${s.worstFitness.toFixed(4)}`,
			);
		} else {
			lines.push(` ${DIM}No heuristics yet — start evolution${RESET}`);
		}

		// ── Message ──────────────────────────────────────────────────────────
		if (this.message) {
			lines.push("");
			lines.push(` ${DIM}${this.message}${RESET}`);
			this.message = "";
		}

		// ── Population view ──────────────────────────────────────────────────
		if (this.showPopulation && s.population.length > 0) {
			lines.push("");
			lines.push(` ${DIM}── Population (${s.population.length}) ──${RESET}`);
			const pop = s.population.slice(0, MAX_VISIBLE_POP);
			for (let i = 0; i < pop.length; i++) {
				const h = pop[i];
				const idx = i + 1;
				const isSelected = idx === this.selectedIndex;
				const prefix = isSelected ? `${getSelectedColor()}▸${RESET}` : " ";
				const fitColor = h.fitness >= 1.0 ? GREEN : CYAN;
				const line = `${prefix} ${DIM}${idx}.${RESET} ${h.thought.slice(0, 40)}${" ".repeat(Math.max(0, 40 - h.thought.length))} ${fitColor}${h.fitness.toFixed(4)}${RESET} [${h.createdBy.slice(0, 12)}]`;
				lines.push(clampLineToWidth(line, contentWidth));
			}
			if (s.population.length > MAX_VISIBLE_POP) {
				lines.push(` ${DIM}  ... and ${s.population.length - MAX_VISIBLE_POP} more${RESET}`);
			}
		}

		// ── Best heuristic detail ────────────────────────────────────────────
		if (this.showDetails && s.bestHeuristic) {
			const bh = s.bestHeuristic;
			lines.push("");
			lines.push(` ${DIM}── Best Heuristic (gen ${bh.generation}, fitness ${bh.fitness.toFixed(4)}) ──${RESET}`);
			lines.push(` ${DIM}Thought:${RESET} ${bh.thought.slice(0, contentWidth - 10)}`);
			lines.push("");
			// Show first 5 lines of code
			const codeLines = bh.code.split("\n").slice(0, 5);
			for (const cl of codeLines) {
				lines.push(` ${DIM}  ${RESET}${cl.slice(0, contentWidth - 8)}`);
			}
			if (bh.code.split("\n").length > 5) {
				lines.push(` ${DIM}  ... (${bh.code.split("\n").length - 5} more lines)${RESET}`);
			}
		}

		// ── Recent log ───────────────────────────────────────────────────────
		if (s.recentLog.length > 0) {
			lines.push("");
			lines.push(` ${DIM}── Recent ──${RESET}`);
			const logEntries = s.recentLog.slice(-MAX_VISIBLE_LOG);
			for (const log of logEntries) {
				lines.push(` ${DIM}  ${RESET}${log.slice(0, contentWidth - 6)}`);
			}
		}

		// ── Action menu ──────────────────────────────────────────────────────
		lines.push("");
		lines.push(` ${DIM}── Actions ──${RESET}`);

		const menuItems: Array<{ label: string; key: string; action: EohPanelAction }> = [];

		if (!s.running) {
			menuItems.push({ label: `Start evolution`, key: "1", action: { type: "start" } });
		}
		if (s.running) {
			menuItems.push({ label: `Stop evolution`, key: "1", action: { type: "stop" } });
		}
		menuItems.push({ label: "Show best heuristic", key: "2", action: { type: "best" } });
		menuItems.push({ label: "Close", key: "q", action: { type: "close" } });

		for (let i = 0; i < menuItems.length; i++) {
			const item = menuItems[i];
			const isSelected = i === this.selectedIndex;
			const prefix = isSelected ? `${getSelectedColor()}▸${RESET}` : " ";
			const keyColor = isSelected ? BOLD : DIM;
			const line = `${prefix} ${keyColor}[${item.key}]${RESET} ${item.label}`;
			lines.push(clampLineToWidth(line, contentWidth));
		}

		// ── Help footer ──────────────────────────────────────────────────────
		lines.push("");
		lines.push(` ${DIM}↑↓ select · Enter confirm · P show/population · D details · q close · Esc close${RESET}`);

		this.cachedLines = lines;
		return lines;
	}
}
