import type { Component } from "../terminal/primitives.ts";
import { theme } from "../terminal/theme.ts";

/** A single horizontal rule, styled to match the divider lines the fixed
 * layout used to draw inline in TUI._doRenderInner. */
export class Separator implements Component {
	private cachedWidth = -1;
	private cachedLines: string[] | null = null;

	render(width: number): string[] {
		if (this.cachedLines !== null && this.cachedWidth === width) {
			return this.cachedLines;
		}
		this.cachedWidth = width;
		this.cachedLines = [theme.fg("separator", "─".repeat(Math.max(0, width)))];
		return this.cachedLines;
	}
}
