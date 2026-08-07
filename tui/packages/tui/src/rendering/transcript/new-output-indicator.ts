import {
	type Component,
	clampLineToWidth,
	visibleWidth,
} from "../../terminal/core.ts";
import { theme } from "../../terminal/theme.ts";
import type { TranscriptDisplay } from "./display.ts";

/** Bottom-anchored overlay: "↓ new output below" shown while the transcript
 * ScrollView isn't following the end and new turns arrived. Replaces the
 * line that used to be painted directly into TranscriptDisplay's clipped
 * viewport — now that ScrollView owns clipping, the transcript component no
 * longer knows what's visible, so this reads its signal instead. */
export class NewOutputIndicator implements Component {
	constructor(private readonly transcriptDisplay: TranscriptDisplay) {}

	private cachedWidth = -1;
	private cachedVisible: boolean | null = null;
	private cachedLines: string[] | null = null;

	// TUI.handleInput() treats a pushed overlay as capturing/"visible" by
	// default unless it exposes this — without it, this always-in-the-stack
	// overlay would permanently block scroll-key and click routing to the
	// transcript, whether or not new output is actually pending.
	get visible(): boolean {
		return this.transcriptDisplay.hasNewOutputBelow();
	}

	render(width: number): string[] {
		const visible = this.transcriptDisplay.hasNewOutputBelow();
		if (
			this.cachedLines !== null &&
			this.cachedWidth === width &&
			this.cachedVisible === visible
		) {
			return this.cachedLines;
		}
		this.cachedWidth = width;
		this.cachedVisible = visible;
		if (!visible) {
			this.cachedLines = [];
			return this.cachedLines;
		}
		const barColor = theme.fgRaw("separator");
		const reset = "\x1b[0m";
		const indicator = `${theme.fg("accent", "↓")} ${theme.fg("muted", "new output below")}`;
		const clipped = clampLineToWidth(indicator, Math.max(1, width - 2));
		const pad = " ".repeat(Math.max(0, width - 2 - visibleWidth(clipped)));
		this.cachedLines = [`${pad}${clipped}${barColor}│${reset}`];
		return this.cachedLines;
	}
}
