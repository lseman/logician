// ── Overlay visibility ──────────────────────────────────────────────────────
// Shared by AppShell and OverlayLayer (ink-app/): an overlay-stack entry can
// be hidden at the stack level (entry.hidden) or hidden by the component's
// own visibility flag (e.g. SlashPopup.visible, toggled by show()/hide()).
// Both must be checked -- missing the latter leaves a fully functional
// component invisible on screen.

import type { Component, OverlayOptions } from "./core.ts";

export interface OverlayEntry {
	component: Component;
	options?: OverlayOptions;
	hidden: boolean;
	focusOrder: number;
}

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
