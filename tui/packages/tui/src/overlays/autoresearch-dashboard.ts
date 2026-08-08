// ── Autoresearch fullscreen dashboard overlay ───────────────────────────────
// Ctrl+Shift+F opens a scrollable results table for the active autoresearch
// session — every logged run, most recent last, with status/metric/commit.
// Built on the shared ListSelectorOverlay base (same shape as the theme/
// model/reasoner selectors): scroll with ↑↓/j-k/PageUp-PageDown, close with
// Escape or q. There's nothing to "select" here (rows aren't actionable),
// so handleInput only ever returns close/null — Enter is a no-op.

import type {
	AutoresearchDashboardRow,
	AutoresearchSession,
} from "@logician/autoresearch";
import { formatNum } from "@logician/autoresearch";
import type { Component } from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import {
	type ListItem,
	ListSelectorOverlay,
	parsePopupListNav,
} from "./popup-utils.ts";

export type AutoresearchDashboardAction = { type: "close" };

const STATUS_DOT: Record<AutoresearchDashboardRow["status"], ListItem["statusDot"]> = {
	keep: "green",
	discard: "gray",
	crash: "red",
	checks_failed: "yellow",
};

function formatTimestamp(ms: number): string {
	const d = new Date(ms);
	const hh = String(d.getHours()).padStart(2, "0");
	const mm = String(d.getMinutes()).padStart(2, "0");
	return `${hh}:${mm}`;
}

export class AutoresearchDashboardOverlay implements Component {
	private readonly inner: ListSelectorOverlay<AutoresearchDashboardRow>;
	private summaryLine = "";

	constructor(private readonly session: AutoresearchSession) {
		this.inner = new ListSelectorOverlay<AutoresearchDashboardRow>({
			title: "Autoresearch",
			hints: " ↑↓/jk scroll · PgUp/PgDn page · esc/q close",
			emptyText: "No experiments logged yet.",
			defaultMessage: "",
			maxRows: 16,
			toItem: row => this.toItem(row),
		});
	}

	get visible(): boolean {
		return this.inner.visible;
	}

	/** Refresh from the live session — call before show() so the table
	 * reflects the latest logged runs, and while visible so it stays live. */
	refresh(): void {
		const data = this.session.getDashboardData();
		this.summaryLine = this.formatSummary(data.summary);
		// Most-recent-last in storage, most-recent-first is easier to scan
		// when you just opened the dashboard mid-loop — but keep chronological
		// (matches the JSONL log and the widget) rather than surprising anyone
		// diffing against .auto/log.jsonl by eye.
		this.inner.setItems(data.rows, data.rows.length - 1);
		this.inner.setMessage(this.summaryLine);
	}

	show(): void {
		this.refresh();
		this.inner.show();
	}

	hide(): void {
		this.inner.hide();
	}

	isVisibleOverlay(): boolean {
		return this.inner.isVisibleOverlay();
	}

	invalidate(): void {
		this.inner.invalidate();
	}

	handleInput(data: string): AutoresearchDashboardAction | null {
		if (!this.inner.visible) return null;
		const nav = parsePopupListNav(data);
		if (nav?.type === "close") return { type: "close" };
		// Enter/confirm has no action for a read-only table — swallow it
		// rather than forwarding ListSelectorOverlay's "select" action.
		this.inner.handleListInput(data);
		return null;
	}

	render(width: number): string[] {
		return this.inner.render(width);
	}

	private toItem(row: AutoresearchDashboardRow): ListItem {
		const label = `#${row.run} ${row.description || "(no description)"}`;
		const metadata = `${row.metricFormatted}  ·  ${row.commit || "—"}  ·  ${formatTimestamp(row.timestamp)}`;
		return {
			label,
			metadata,
			statusDot: STATUS_DOT[row.status],
			badge: row.isBest ? { text: "best", color: theme.fgRaw("success") } : undefined,
		};
	}

	private formatSummary(
		summary: ReturnType<AutoresearchSession["getWidgetSummary"]>,
	): string {
		if (!summary) return "No active session.";
		const parts: string[] = [];
		if (summary.name) parts.push(summary.name);
		if (summary.bestMetric !== null) {
			parts.push(
				`best ${summary.metricName} ${formatNum(summary.bestMetric, summary.metricUnit)}`,
			);
		}
		if (summary.confidence !== null) {
			parts.push(`confidence ${summary.confidence.toFixed(1)}×`);
		}
		parts.push(summary.active ? "active" : "off");
		return parts.join("  ·  ");
	}
}
