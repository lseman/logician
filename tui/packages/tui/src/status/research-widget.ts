// ── Autoresearch status widget ───────────────────────────────────────────────
// Persistent one-line summary of the active autoresearch session (best
// metric, run count, confidence, currently-running command), shown above the
// input bar the same way WorkSurface shows the current turn's working set.
// Renders zero lines whenever there's nothing to show — see
// AutoresearchSession.getWidgetSummary().

import type { AutoresearchSession } from "@logician/autoresearch";
import { formatNum } from "@logician/autoresearch";
import {
	type Component,
	clampLineToWidth,
	RESET,
	visibleWidth,
} from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";

function formatElapsed(ms: number): string {
	const seconds = Math.floor(ms / 1000);
	if (seconds < 60) return `${seconds}s`;
	const minutes = Math.floor(seconds / 60);
	if (minutes < 60) return `${minutes}m${seconds % 60}s`;
	const hours = Math.floor(minutes / 60);
	return `${hours}h${minutes % 60}m`;
}

function confidenceLabel(confidence: number): string {
	const str = `${confidence.toFixed(1)}×`;
	if (confidence >= 2.0) return theme.fg("success", str);
	if (confidence >= 1.0) return theme.fg("warning", str);
	return theme.fg("error", str);
}

/** Polls AutoresearchSession.getWidgetSummary() on each render — cheap
 * (plain object read, no I/O) so no separate invalidation wiring is needed;
 * the widget just needs to be re-rendered on the normal frame cadence,
 * which already happens on every tool call / turn boundary. */
export class ResearchWidget implements Component {
	constructor(private readonly session: AutoresearchSession) {}

	render(width: number): string[] {
		const summary = this.session.getWidgetSummary();
		if (!summary) return [];

		const parts: string[] = [];

		const stateGlyph = summary.running
			? theme.fg("warning", "●")
			: summary.active
				? theme.fg("success", "◆")
				: theme.fg("dim", "◇");
		const label = summary.name
			? theme.fg("text", summary.name)
			: theme.fg("muted", "autoresearch");
		parts.push(`${stateGlyph} ${label}`);

		if (summary.bestMetric !== null) {
			parts.push(
				`${theme.fg("muted", summary.metricName)} ${theme.fg("text", formatNum(summary.bestMetric, summary.metricUnit))}`,
			);
		}

		if (summary.runCount > 0) {
			const limitSuffix =
				summary.maxExperiments !== null ? `/${summary.maxExperiments}` : "";
			parts.push(theme.fg("dim", `${summary.runCount}${limitSuffix} runs`));
		}

		if (summary.confidence !== null) {
			parts.push(
				`${theme.fg("muted", "conf")} ${confidenceLabel(summary.confidence)}`,
			);
		}

		if (summary.running) {
			parts.push(
				theme.fg(
					"warning",
					`running ${formatElapsed(summary.running.elapsedMs)}`,
				),
			);
		} else if (!summary.active) {
			parts.push(theme.fg("dim", "off"));
		}

		const line = parts.join(theme.fg("dim", "  ·  ") + RESET);
		const clipped = clampLineToWidth(line, width);
		return [clipped + " ".repeat(Math.max(0, width - visibleWidth(clipped)))];
	}
}
