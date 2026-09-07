// ── Skill popup ──────────────────────────────────────────────────────────────
// Inline skill://-autocomplete: fuzzy-matches available skills as the user types
// after "skill://". Mirrors FileMentionPopup's inline-autocomplete pattern.

import type { Component } from "../terminal/core.ts";
import type { Skill } from "@logician/log-runtime/skills";
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

interface ScoredSkill {
	name: string;
	score: number;
}

/** Fuzzy-score a skill name against a query, favoring prefix and exact matches. */
function scoreSkill(query: string, name: string): number {
	const lower = name.toLowerCase();
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

export class SkillPopup implements Component {
	private skills: Skill[] = [];
	private query = "";
	private selectedIndex = 0;
	public visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private matches: string[] = [];

	setSkills(skills: Skill[]): void {
		this.skills = skills;
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

	/** Skill name of the highlighted row, or null when no match. */
	currentSkill(): string | null {
		return this.matches.length > 0 ? this.matches[this.selectedIndex] : null;
	}

	private _computeMatches(): string[] {
		const q = this.query;
		if (!q) return this.skills.slice(0, MAX_MATCHES).map(s => s.name);
		const scored: ScoredSkill[] = [];
		for (const skill of this.skills) {
			const score = scoreSkill(q, skill.name);
			if (score >= 0) scored.push({ name: skill.name, score });
		}
		scored.sort((a, b) => b.score - a.score || a.name.length - b.name.length);
		return scored.slice(0, MAX_MATCHES).map(s => s.name);
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
			(name, index) => {
				const item: ListItem = {
					label: name,
					selected: index === this.selectedIndex,
				};
				return renderListItem(item, innerWidth);
			},
			"No matching skills.",
		);
		this.cachedLines = clampPopupLines(
			renderListPopupFrame({
				popupWidth,
				innerWidth,
				title: "skills",
				subtitle: ` (${this.matches.length})`,
				hints: "↑↓ select · tab/enter insert · esc close",
				bodyLines,
				bottomText: this.query
					? `Matching skill://${this.query}`
					: "Pick a skill.",
			}),
			width,
		);
		return this.cachedLines;
	}
}
