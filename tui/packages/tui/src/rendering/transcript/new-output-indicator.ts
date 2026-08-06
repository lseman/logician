import { clampLineToWidth, type Component, visibleWidth } from "../../terminal/core.ts";
import { theme } from "../../terminal/theme.ts";
import type { TranscriptDisplay } from "./display.ts";

/** Bottom-anchored overlay: "↓ new output below" shown while the transcript
 * ScrollView isn't following the end and new turns arrived. Replaces the
 * line that used to be painted directly into TranscriptDisplay's clipped
 * viewport — now that ScrollView owns clipping, the transcript component no
 * longer knows what's visible, so this reads its signal instead. */
export class NewOutputIndicator implements Component {
	constructor(private readonly transcriptDisplay: TranscriptDisplay) {}

	render(width: number): string[] {
		if (!this.transcriptDisplay.hasNewOutputBelow()) return [];
		const barColor = theme.fgRaw("separator");
		const reset = "\x1b[0m";
		const indicator = `${theme.fg("accent", "↓")} ${theme.fg("muted", "new output below")}`;
		const clipped = clampLineToWidth(indicator, Math.max(1, width - 2));
		const pad = " ".repeat(Math.max(0, width - 2 - visibleWidth(clipped)));
		return [`${pad}${clipped}${barColor}│${reset}`];
	}
}
