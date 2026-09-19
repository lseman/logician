import type { Component } from "../terminal/primitives.ts";
import { theme } from "../terminal/theme.ts";

/** A single horizontal rule, styled to match the divider lines the fixed
 * layout used to draw inline in TUI._doRenderInner. */
export class Separator implements Component {
	private cachedWidth = -1;
	private cachedLines: string[] | null = null;
	private cachedColor = "";

	constructor(private color: "separator" | "inputBarBorder" = "separator") {}

	render(width: number): string[] {
		const color = theme.fgRaw(this.color);
		if (
			this.cachedLines !== null &&
			this.cachedWidth === width &&
			this.cachedColor === color
		) {
			return this.cachedLines;
		}
		this.cachedWidth = width;
		this.cachedColor = color;
		this.cachedLines = [theme.fg(this.color, "─".repeat(Math.max(0, width)))];
		return this.cachedLines;
	}
}
