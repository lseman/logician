import type {
	OverlayHandle,
	OverlayOptions,
	OverlayStackEntry,
} from "./overlay-types.ts";
import type { Component } from "./primitives.ts";

export interface OverlayManagerHost {
	getFocus: () => Component | null;
	setFocus: (component: Component | null) => void;
	requestRender: () => void;
}

/** Owns overlay stacking, visibility, and focus restoration policy. */
export class OverlayManager {
	private readonly stack: OverlayStackEntry[] = [];
	private focusOrder = 0;

	constructor(private readonly host: OverlayManagerHost) {}

	get entries(): readonly OverlayStackEntry[] {
		return this.stack;
	}

	show(component: Component, options?: OverlayOptions): OverlayHandle {
		const entry: OverlayStackEntry = {
			component,
			options,
			preFocus: this.host.getFocus(),
			hidden: false,
			focusOrder: ++this.focusOrder,
		};
		this.stack.push(entry);
		this.host.requestRender();

		return {
			hide: () => this.removeEntry(entry),
			setHidden: hidden => this.setHidden(entry, hidden),
			isHidden: () => entry.hidden,
			isFocused: () => this.host.getFocus() === component,
			focus: () => {
				if (!this.stack.includes(entry) || !this.isVisible(entry)) return;
				entry.focusOrder = ++this.focusOrder;
				this.host.setFocus(component);
				this.host.requestRender();
			},
			unfocus: target => {
				if (this.host.getFocus() !== component) return;
				this.host.setFocus(
					target ?? this.topmost()?.component ?? entry.preFocus,
				);
				this.host.requestRender();
			},
		};
	}

	hideTop(): void {
		const entry = this.stack.at(-1);
		if (entry) this.removeEntry(entry);
	}

	bringToFront(component: Component): void {
		const index = this.stack.findIndex(entry => entry.component === component);
		if (index < 0 || index === this.stack.length - 1) return;
		const [entry] = this.stack.splice(index, 1);
		this.stack.push(entry);
		this.host.requestRender();
	}

	remove(component: Component): void {
		const entry = this.stack.find(item => item.component === component);
		if (!entry) return;
		if (
			"visible" in entry.component &&
			typeof (entry.component as { visible?: unknown }).visible === "boolean"
		) {
			(entry.component as { visible: boolean }).visible = false;
		}
		this.removeEntry(entry);
	}

	isVisible(entry: OverlayStackEntry): boolean {
		if (entry.hidden) return false;
		if (
			"visible" in entry.component &&
			typeof (entry.component as { visible?: unknown }).visible === "boolean"
		) {
			return (entry.component as { visible: boolean }).visible;
		}
		return true;
	}

	topmost(): OverlayStackEntry | undefined {
		let topmost: OverlayStackEntry | undefined;
		for (const entry of this.stack) {
			if (entry.options?.nonCapturing || !this.isVisible(entry)) continue;
			if (!topmost || entry.focusOrder > topmost.focusOrder) topmost = entry;
		}
		return topmost;
	}

	private setHidden(entry: OverlayStackEntry, hidden: boolean): void {
		if (entry.hidden === hidden) return;
		entry.hidden = hidden;
		if (hidden && this.host.getFocus() === entry.component) {
			this.host.setFocus(this.topmost()?.component ?? entry.preFocus);
		} else if (
			!hidden &&
			!entry.options?.nonCapturing &&
			this.isVisible(entry)
		) {
			entry.focusOrder = ++this.focusOrder;
			this.host.setFocus(entry.component);
		}
		this.host.requestRender();
	}

	private removeEntry(entry: OverlayStackEntry): void {
		const index = this.stack.indexOf(entry);
		if (index < 0) return;
		this.stack.splice(index, 1);
		if (this.host.getFocus() === entry.component) {
			this.host.setFocus(this.topmost()?.component ?? entry.preFocus);
		}
		this.host.requestRender();
	}
}
