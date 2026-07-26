// ── File mention popup ──────────────────────────────────────────────────────
// Inline @-mention autocomplete: fuzzy-matches project files as the user types
// after an "@". Mirrors SlashPopup's inline-autocomplete pattern (input bar
// keeps focus; this popup renders below and only intercepts nav/accept keys).

import { theme } from "../layers/theme/theme.ts";
import { type Component, RESET, BOLD, DIM } from "../layers/core/tui-core.ts";

const MAX_VISIBLE_ENTRIES = 8;
const MAX_MATCHES = 50;
const getHeaderColor = (): string => theme.fg("header", "");
const getSelectedColor = (): string => theme.fg("selected", "");

interface ScoredFile {
	path: string;
	score: number;
}

/** Fuzzy-score a file path against a query, favoring basename and prefix matches. */
function scoreFile(query: string, path: string): number {
	const lowerPath = path.toLowerCase();
	const base = lowerPath.slice(lowerPath.lastIndexOf("/") + 1);

	if (base === query) return 3000;
	if (base.startsWith(query)) return 2500 - (base.length - query.length);
	if (lowerPath.startsWith(query)) return 2200 - (lowerPath.length - query.length);
	if (base.includes(query)) return 2000 - base.indexOf(query) * 8;
	if (lowerPath.includes(query)) return 1500 - lowerPath.indexOf(query) * 4;
	if (subsequenceMatch(query, base)) return 800;
	return -1;
}

function subsequenceMatch(query: string, text: string): boolean {
	let qi = 0;
	for (let i = 0; i < text.length && qi < query.length; i++) {
		if (text[i] === query[qi]) qi++;
	}
	return qi === query.length;
}

export class FileMentionPopup implements Component {
	private files: string[] = [];
	private query = "";
	private selectedIndex = 0;
	public visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private matches: string[] = [];

	setFiles(files: string[]): void {
		this.files = files;
		this.matches = this._computeMatches();
		if (this.selectedIndex >= this.matches.length) {
			this.selectedIndex = Math.max(0, this.matches.length - 1);
		}
		this.invalidate();
	}

	setQuery(query: string): void {
		this.query = query;
		this.matches = this._computeMatches();
		if (this.selectedIndex >= this.matches.length) {
			this.selectedIndex = Math.max(0, this.matches.length - 1);
		}
		this.invalidate();
	}

	hasMatches(): boolean {
		return this.matches.length > 0;
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	hide(): void {
		this.visible = false;
		this.invalidate();
	}

	moveSelection(delta: number): void {
		const n = this.matches.length;
		if (n === 0) return;
		this.selectedIndex = (this.selectedIndex + delta + n) % n;
		this.invalidate();
	}

	/** File path of the highlighted row, or null when no match. */
	currentFile(): string | null {
		return this.matches.length > 0 ? this.matches[this.selectedIndex] : null;
	}

	private _computeMatches(): string[] {
		const q = this.query.toLowerCase();
		if (!q) return this.files.slice(0, MAX_MATCHES);
		const scored: ScoredFile[] = [];
		for (const path of this.files) {
			const score = scoreFile(q, path);
			if (score >= 0) scored.push({ path, score });
		}
		scored.sort((a, b) => b.score - a.score || a.path.length - b.path.length);
		return scored.slice(0, MAX_MATCHES).map((s) => s.path);
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

		const lines: string[] = [];

		const hint = `${DIM}↑↓ select · Tab/⏎ insert · Esc close${RESET}`;
		lines.push(
			` ${getHeaderColor()}files${RESET}${DIM} (${this.matches.length})${RESET}  ${hint}`,
		);

		if (this.matches.length === 0) {
			lines.push(`${DIM}  No matching files${RESET}`);
			this.cachedLines = lines;
			return lines;
		}

		const { items, start, hiddenAbove, hiddenBelow } = windowAroundSelection(
			this.matches,
			this.selectedIndex,
		);
		if (hiddenAbove > 0) {
			lines.push(`${DIM}  ↑ ${hiddenAbove} more above${RESET}`);
		}
		for (let i = 0; i < items.length; i++) {
			const path = items[i];
			const isSelected = start + i === this.selectedIndex;
			const prefix = isSelected ? "▸ " : "  ";
			const line = isSelected
				? ` ${getSelectedColor()}${prefix}${BOLD}${path}${RESET}`
				: ` ${prefix}${path}`;
			lines.push(line);
		}
		if (hiddenBelow > 0) {
			lines.push(`${DIM}  ↓ ${hiddenBelow} more below${RESET}`);
		}

		this.cachedLines = lines;
		return lines;
	}
}

function windowAroundSelection<T>(
	items: T[],
	selection: number,
): { items: T[]; start: number; hiddenAbove: number; hiddenBelow: number } {
	if (items.length <= MAX_VISIBLE_ENTRIES) {
		return { items, start: 0, hiddenAbove: 0, hiddenBelow: 0 };
	}
	const half = Math.floor(MAX_VISIBLE_ENTRIES / 2);
	const start = Math.max(
		0,
		Math.min(selection - half, items.length - MAX_VISIBLE_ENTRIES),
	);
	const end = Math.min(items.length, start + MAX_VISIBLE_ENTRIES);
	return {
		items: items.slice(start, end),
		start,
		hiddenAbove: start,
		hiddenBelow: items.length - end,
	};
}
