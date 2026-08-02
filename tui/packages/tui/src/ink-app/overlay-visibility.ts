import type { Component, OverlayOptions } from "../terminal/core.ts";

export interface OverlayEntry {
	component: Component;
	options?: OverlayOptions;
	hidden: boolean;
	focusOrder: number;
}

/** Account for both stack-level and component-owned overlay visibility. */
export function isEntryVisible(entry: OverlayEntry): boolean {
	if (entry.hidden) return false;
	if (
		"visible" in entry.component &&
		typeof (entry.component as { visible?: unknown }).visible === "boolean"
	) {
		return (entry.component as { visible: boolean }).visible;
	}
	return true;
}
