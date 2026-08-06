import type { Component } from "../terminal/primitives.ts";

/** A single horizontal rule, styled to match the divider lines the fixed
 * layout used to draw inline in TUI._doRenderInner. */
export class Separator implements Component {
	render(width: number): string[] {
		return [`\x1b[38;5;236m${"─".repeat(Math.max(0, width))}\x1b[0m`];
	}
}
