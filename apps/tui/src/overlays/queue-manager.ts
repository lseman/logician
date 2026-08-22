// ── QueueManagerOverlay ──────────────────────────────────────────────────────
// Interactive view of queued steering/follow-up/next-turn messages. Lets the
// user select an entry and delete it, or clear everything, instead of typing
// /queue-drop <n> by number. Mirrors McpManagerOverlay's shape: a Component
// with its own handleInput (nav + a mutating key) and the shared popup frame.

import type { Component } from "../terminal/core.ts";
import {
	clampPopupLines,
	type ListItem,
	POPUP_FRAME_OVERHEAD,
	parsePopupListNav,
	renderListItem,
	renderListPopupFrame,
	renderStatusLine,
} from "./popup-utils.ts";
import { SelectorController } from "./selector-controller.ts";

export type QueueEntryKind = "steering" | "followUp" | "nextTurn";

export interface QueueEntry {
	kind: QueueEntryKind;
	/** Index into the combined [steering, followUp] list that /queue-drop and
	 * the bridge's dropQueuedMessage() use. Undefined for nextTurn entries,
	 * which the backend has no drop path for (they're already committed to
	 * the next user-initiated turn). */
	dropIndex: number | undefined;
	content: string;
}

export type QueueManagerAction =
	| { type: "drop"; entry: QueueEntry }
	| { type: "clear" }
	| { type: "close" };

const KIND_META: Record<
	QueueEntryKind,
	{ label: string; color: string; dot: "blue" | "gray" | "green" }
> = {
	steering: { label: "QUEUE", color: "\x1b[36m", dot: "blue" },
	followUp: { label: "LATER", color: "\x1b[90m", dot: "gray" },
	nextTurn: { label: "NEXT", color: "\x1b[32m", dot: "green" },
};

export class QueueManagerOverlay implements Component {
	public visible = false;
	private entries: QueueEntry[] = [];
	private _selection = new SelectorController();
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	/** @internal Exposed for tests. */
	get selection(): SelectorController {
		return this._selection;
	}

	setQueues(
		steering: readonly string[],
		followUp: readonly string[],
		nextTurn: readonly string[],
	): void {
		this.entries = [
			...steering.map((content, i) => ({
				kind: "steering" as const,
				dropIndex: i,
				content,
			})),
			...followUp.map((content, i) => ({
				kind: "followUp" as const,
				dropIndex: steering.length + i,
				content,
			})),
			...nextTurn.map(content => ({
				kind: "nextTurn" as const,
				dropIndex: undefined,
				content,
			})),
		];
		this._selection.set(this._selection.index, this.entries.length);
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

	handleInput(data: string): QueueManagerAction | null {
		if (!this.visible) return null;

		if (data === "d" || data === "D" || data === "\x7f" || data === "\x08") {
			const entry = this.entries[this._selection.index];
			return entry ? { type: "drop", entry } : null;
		}
		if (data === "c" || data === "C") {
			return this.entries.length ? { type: "clear" } : null;
		}

		const nav = parsePopupListNav(data);
		if (nav?.type === "close") return { type: "close" };
		if (nav?.type === "confirm") return { type: "close" };
		if (nav?.type === "move") {
			this._selection.move(nav.delta, this.entries.length);
			this.invalidate();
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

		const bodyLines = this.entries.length
			? this.entries.map((entry, i) => renderEntry(entry, i, innerWidth, this._selection.index))
			: [renderStatusLine("Queue is empty.", innerWidth)];

		const steeringCount = this.entries.filter(e => e.kind === "steering").length;
		const followUpCount = this.entries.filter(e => e.kind === "followUp").length;
		const nextTurnCount = this.entries.filter(e => e.kind === "nextTurn").length;
		const parts: string[] = [];
		if (steeringCount) parts.push(`${steeringCount} queued`);
		if (followUpCount) parts.push(`${followUpCount} follow-up`);
		if (nextTurnCount) parts.push(`${nextTurnCount} next turn`);

		const lines = renderListPopupFrame({
			popupWidth,
			innerWidth,
			title: "Message Queue",
			subtitle: parts.length ? ` — ${parts.join(" · ")}` : " (0)",
			hints: " ↑↓ select · d delete · c clear all · enter/esc close",
			bodyLines,
			bottomText: this.message,
		});

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}
}

function renderEntry(
	entry: QueueEntry,
	index: number,
	innerWidth: number,
	selectedIndex: number,
): string {
	const meta = KIND_META[entry.kind];
	const flat = entry.content.replace(/\s+/g, " ").trim();
	const item: ListItem = {
		label: flat,
		badge: { text: meta.label, color: meta.color },
		statusDot: meta.dot,
		selected: index === selectedIndex,
	};
	return renderListItem(item, innerWidth);
}
