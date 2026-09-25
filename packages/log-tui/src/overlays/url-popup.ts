// ── URL popup ──────────────────────────────────────────────────────────────────
// Inline autocomplete for internal-URL tokens (memory://, ssh://, log://,
// local://, …) whose protocol handler supports completion. One generic popup
// for all schemes: items are full URLs, the typed token is the completion
// prefix. Mirrors SkillPopup's inline-autocomplete pattern.

import type { Component } from "../terminal/core.ts";
import {
	clampPopupLines,
	type ListItem,
	POPUP_FRAME_OVERHEAD,
	renderListItem,
	renderListPopupBody,
	renderListPopupFrame,
} from "./popup-utils.ts";

const MAX_VISIBLE_ENTRIES = 8;
const MAX_MATCHES = 50;

/** A completion candidate with its display description. */
export interface UrlSuggestion {
	value: string;
	description?: string;
}

interface ScoredSuggestion {
	value: string;
	description?: string;
	score: number;
}

/** Fuzzy-score a candidate against a query, favoring prefix and exact matches. */
function scoreCandidate(query: string, target: string): number {
	const lower = target.toLowerCase();
	const q = query.toLowerCase();

	if (lower === q) return 3000;
	if (lower.startsWith(q)) return 2500 - (lower.length - q.length);
	if (lower.includes(q)) return 2000 - lower.indexOf(q) * 8;
	if (subsequenceMatch(q, lower)) return 800;
	return -1;
}

function subsequenceMatch(query: string, text: string): boolean {
	let qi = 0;
	for (let i = 0; i < text.length && qi < query.length; i++) {
		if (text[i] === query[qi]) qi++;
	}
	return qi === query.length;
}

/** The part of a `scheme://rest` URL after the slashes; the whole value for plain values. */
function urlTail(value: string): string {
	const idx = value.indexOf("://");
	return idx >= 0 ? value.slice(idx + 3) : value;
}

export class UrlPopup implements Component {
	private items: UrlSuggestion[] = [];
	private token = "";
	private selectedIndex = 0;
	public visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private matches: UrlSuggestion[] = [];

	/** Replace the candidate set (full URL values, e.g. `memory://memories`). */
	setItems(items: UrlSuggestion[]): void {
		this.items = items;
		this.matches = this._computeMatches();
		if (this.selectedIndex >= this.matches.length) {
			this.selectedIndex = Math.max(0, this.matches.length - 1);
		}
		this.invalidate();
	}

	/** The full token the user typed at the cursor, e.g. `memory://mem`. */
	setQuery(token: string): void {
		this.token = token;
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

	/** Full URL of the highlighted row, or null when no match. */
	currentValue(): string | null {
		return this.matches.length > 0
			? this.matches[this.selectedIndex].value
			: null;
	}

	private _computeMatches(): UrlSuggestion[] {
		const query = urlTail(this.token);
		if (!query) {
			return this.items.slice(0, MAX_MATCHES);
		}
		const scored: ScoredSuggestion[] = [];
		for (const item of this.items) {
			const score = scoreCandidate(query, urlTail(item.value));
			if (score >= 0) {
				scored.push({
					value: item.value,
					description: item.description,
					score,
				});
			}
		}
		scored.sort((a, b) => b.score - a.score || a.value.length - b.value.length);
		return scored.slice(0, MAX_MATCHES);
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
		const schemeIdx = this.token.indexOf("://");
		const scheme = schemeIdx >= 0 ? this.token.slice(0, schemeIdx) : "";
		const selection = {
			window: (count: number, maxRows: number) => {
				const start = Math.max(
					0,
					Math.min(
						this.selectedIndex - Math.floor(maxRows / 2),
						Math.max(0, count - maxRows),
					),
				);
				return { start, end: Math.min(count, start + maxRows) };
			},
		};
		const bodyLines = renderListPopupBody(
			this.matches,
			selection,
			innerWidth,
			MAX_VISIBLE_ENTRIES,
			(item, index) => {
				const entry: ListItem = {
					label: item.value,
					metadata: item.description,
					selected: index === this.selectedIndex,
				};
				return renderListItem(entry, innerWidth);
			},
			"No matching URLs.",
		);
		this.cachedLines = clampPopupLines(
			renderListPopupFrame({
				popupWidth,
				innerWidth,
				title: scheme ? `${scheme}://` : "url",
				subtitle: ` (${this.matches.length})`,
				hints: "↑↓ select · tab/enter insert · esc close",
				bodyLines,
				bottomText: this.token ? `Matching ${this.token}` : "Pick a URL.",
			}),
			width,
		);
		return this.cachedLines;
	}
}
