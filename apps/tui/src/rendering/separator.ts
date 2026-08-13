import type { Component } from "../terminal/primitives.ts";

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
		this.cachedLines = [
			`\x1b[38;5;236m${"─".repeat(Math.max(0, width))}\x1b[0m`,
		];
		return this.cachedLines;
	}
}
