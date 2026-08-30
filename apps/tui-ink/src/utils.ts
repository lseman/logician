// ── Ink TUI Utilities ─────────────────────────────────────────────────────────

import { readdirSync } from "node:fs";
import { join, relative } from "node:path";
import type React from "react";
import type {
	GitStatus,
	Notification,
	NotificationLevel,
	OverlayKind,
	OverlayState,
	SessionInfo,
	ThinkingDisplayMode,
	ThinkingLevel,
	TuiPhase,
	Turn,
	WorkflowMode,
	InferenceMode,
	InferenceSettings,
	ReasonerStatus,
	SteerMessage,
	TodoItem,
} from "./types.ts";

// ── Formatting helpers ────────────────────────────────────────────────────────

export function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value));
}

export function truncate(str: string, maxLen: number): string {
	if (str.length <= maxLen) return str;
	return str.slice(0, maxLen - 1) + "…";
}

export function ellipsis(str: string, maxLen: number): string {
	const ellipsisStr = "…";
	const ellipsisLen = ellipsisStr.length;
	if (str.length <= maxLen) return str;
	if (maxLen <= ellipsisLen) return str.slice(0, maxLen);
	const half = Math.floor((maxLen - ellipsisLen) / 2);
	const remainder = maxLen - ellipsisLen - half;
	return str.slice(0, half) + ellipsisStr + str.slice(-remainder);
}

export function padRight(text: string, width: number, char = " "): string {
	if (text.length >= width) return text.slice(0, width);
	return text + char.repeat(width - text.length);
}

export function padLeft(text: string, width: number, char = " "): string {
	if (text.length >= width) return text.slice(-width);
	return char.repeat(width - text.length) + text;
}

export function stripAnsi(text: string): string {
	return text.replace(/\x1b\[[0-9;]*m/g, "");
}

// ── Box drawing characters ────────────────────────────────────────────────────

export const Box = {
	horizontal: "─",
	vertical: "│",
	topLeft: "┌",
	topRight: "┐",
	bottomLeft: "└",
	bottomRight: "┘",
	verticalLeft: "├",
	verticalRight: "┤",
	horizontalUp: "┬",
	horizontalDown: "┴",
	cross: "┼",
	dot: "•",
	arrow: "→",
	downArrow: "↓",
	upArrow: "↑",
	leftArrow: "←",
	rightArrow: "→",
	checkboxEmpty: "☐",
	checkboxChecked: "☑",
	radioEmpty: "○",
	radioSelected: "◉",
	separator: "│",
	downwardsArrow: "↓",
};

// ── Turn formatting ───────────────────────────────────────────────────────────

export function formatTurnContent(
	content: string,
	thinking?: string,
	thinkingMode: ThinkingDisplayMode = "collapsed",
	maxLength?: number,
): string {
	if (maxLength && content.length > maxLength) {
		content = content.slice(0, maxLength) + "…";
	}

	let result = "";

	if (thinking && thinkingMode !== "collapsed") {
		const thinkingContent = thinkingMode === "expanded"
			? thinking
			: thinking.length > 200
				? thinking.slice(0, 200) + "…"
				: thinking;
		result += `[thinking]\n${thinkingContent}\n[/thinking]\n\n`;
	}

	result += content;
	return result;
}

// ── Time formatting ───────────────────────────────────────────────────────────

export function formatDuration(ms: number): string {
	const seconds = Math.floor(ms / 1000);
	const minutes = Math.floor(seconds / 60);
	const hours = Math.floor(minutes / 60);

	if (hours > 0) return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
	if (minutes > 0) return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
	return `${seconds}s`;
}

export function formatTokenCount(tokens: number): string {
	if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
	if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}K`;
	return String(tokens);
}

// ── List helpers ──────────────────────────────────────────────────────────────

export function formatListItem(
	index: number,
	text: string,
	selected: boolean,
	total: number,
	maxWidth: number,
): { marker: string; number: string; text: string } {
	const numPad = String(total).length;
	const num = padLeft(`${index + 1}`, numPad);
	const marker = selected ? "▸ " : "  ";
	const availableWidth = maxWidth - marker.length - numPad - 2;
	const itemText = text.length > availableWidth
		? text.slice(0, availableWidth - 1) + "…"
		: text;
	return { marker, number: num, text: itemText };
}

export function formatTodoItem(
	index: number,
	item: TodoItem,
	selected: boolean,
	total: number,
	maxWidth: number,
): { marker: string; number: string; checkbox: string; text: string } {
	const numPad = String(total).length;
	const num = padLeft(`${index + 1}`, numPad);
	const marker = selected ? "▸ " : "  ";
	const checkbox = item.done ? "✓" : " ";
	const availableWidth = maxWidth - marker.length - numPad - 5;
	const text = item.text.length > availableWidth
		? item.text.slice(0, availableWidth - 1) + "…"
		: item.text;
	return { marker, number: num, checkbox, text };
}

// ── Status bar helpers ────────────────────────────────────────────────────────

export function formatPhaseEmoji(phase: TuiPhase): string {
	switch (phase) {
		case "ready": return "●";
		case "thinking": return "◌";
		case "working": return "◉";
		case "cancelling": return "✕";
		case "error": return "✗";
		default: return "●";
	}
}

export function formatPhaseColor(phase: TuiPhase): string {
	switch (phase) {
		case "ready": return "green";
		case "thinking": return "yellow";
		case "working": return "cyan";
		case "cancelling": return "red";
		case "error": return "red";
		default: return "gray";
	}
}

export function formatModelShort(model: string): string {
	if (!model) return "?";
	return model.split("/").pop() ?? model;
}

export function formatTokenUsage(tokens: number, maxTokens: number): string {
	const pct = Math.round((tokens / maxTokens) * 100);
	return `${formatTokenCount(tokens)}/${formatTokenCount(maxTokens)} (${pct}%)`;
}

// ── Git formatting ────────────────────────────────────────────────────────────

export function formatGitStatus(git: GitStatus): string[] {
	const parts: string[] = [];
	if (git.staged > 0) parts.push(`+${git.staged}`);
	if (git.modified > 0) parts.push(`~${git.modified}`);
	if (git.untracked > 0) parts.push(`?${git.untracked}`);
	return parts;
}

// ── File mention suggestions ─────────────────────────────────────────────────

const FILE_IGNORE = new Set([
	"node_modules",
	".git",
	"dist",
	"build",
	".next",
	".cache",
	"coverage",
]);

/**
 * Shallow, best-effort file/dir suggestions under `root` matching `query`
 * (case-insensitive substring). Walks at most `limit` entries, two levels deep.
 */
export function listFileSuggestions(
	root: string,
	query: string,
	limit = 12,
): string[] {
	const q = query.toLowerCase();
	const out: string[] = [];

	const walk = (dir: string, depth: number): void => {
		if (out.length >= limit || depth > 2) return;
		let entries: import("node:fs").Dirent[];
		try {
			entries = readdirSync(dir, { withFileTypes: true });
		} catch {
			return;
		}
		for (const entry of entries) {
			if (out.length >= limit) return;
			if (entry.name.startsWith(".") || FILE_IGNORE.has(entry.name)) continue;
			const full = join(dir, entry.name);
			const rel = relative(root, full);
			if (entry.isDirectory()) {
				if (!q || rel.toLowerCase().includes(q)) out.push(`${rel}/`);
				walk(full, depth + 1);
			} else if (!q || rel.toLowerCase().includes(q)) {
				out.push(rel);
			}
		}
	};

	walk(root, 0);
	return out.slice(0, limit);
}

// ── Overlay helpers ───────────────────────────────────────────────────────────

export function isOverlayOpen(overlay: OverlayState): boolean {
	return overlay.kind !== null;
}

export function closeOverlay(): OverlayState {
	return { kind: null };
}

// ── Scroll helpers ────────────────────────────────────────────────────────────

export interface ScrollState {
	position: number;
	maxPosition: number;
	follow: "end" | "manual";
}

export function scrollToEnd(state: ScrollState): ScrollState {
	return { ...state, position: state.maxPosition, follow: "end" };
}

export function scrollUp(state: ScrollState, amount: number): ScrollState {
	const newPosition = Math.max(0, state.position - amount);
	return { ...state, position: newPosition, follow: "manual" };
}

export function scrollDown(state: ScrollState, amount: number): ScrollState {
	const newPosition = Math.min(state.maxPosition, state.position + amount);
	return { ...state, position: newPosition, follow: state.follow };
}

export function scrollToTop(state: ScrollState): ScrollState {
	return { ...state, position: 0, follow: "manual" };
}

export function pageDown(state: ScrollState, pageSize: number): ScrollState {
	return scrollDown(state, pageSize);
}

export function pageUp(state: ScrollState, pageSize: number): ScrollState {
	return scrollUp(state, pageSize);
}

// ── Clipboard / Kill ring ─────────────────────────────────────────────────────

export class KillRing {
	private items: string[] = [];
	private index = -1;
	private yankMarker = -1;

	push(text: string): void {
		this.items = this.items.slice(0, this.index + 1);
		this.items.push(text);
		this.index = this.items.length - 1;
		this.yankMarker = this.index;
	}

	yank(): string | null {
		if (this.index < 0) return null;
		const text = this.items[this.yankMarker >= 0 ? this.yankMarker : this.index];
		return text ?? null;
	}

	yankPop(): string | null {
		if (this.items.length === 0) return null;
		this.yankMarker = (this.yankMarker + 1) % this.items.length;
		return this.items[this.yankMarker];
	}

	pop(): string | null {
		if (this.index < 0) return null;
		const text = this.items[this.index];
		this.index--;
		this.yankMarker = this.index;
		return text ?? null;
	}

	isEmpty(): boolean {
		return this.index < 0;
	}

	clear(): void {
		this.items = [];
		this.index = -1;
		this.yankMarker = -1;
	}
}

// ── Undo stack ────────────────────────────────────────────────────────────────

export class UndoStack<T> {
	private states: T[] = [];
	private index = -1;

	push(state: T): void {
		this.states = this.states.slice(0, this.index + 1);
		this.states.push(state);
		this.index = this.states.length - 1;
	}

	undo(): T | null {
		if (this.index <= 0) return null;
		this.index--;
		return this.states[this.index];
	}

	redo(): T | null {
		if (this.index >= this.states.length - 1) return null;
		this.index++;
		return this.states[this.index];
	}

	canUndo(): boolean {
		return this.index > 0;
	}

	canRedo(): boolean {
		return this.index < this.states.length - 1;
	}

	isEmpty(): boolean {
		return this.index < 0;
	}
}

