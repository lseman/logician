import { expect, test } from "bun:test";
import { OverlayManager } from "../terminal/overlay-manager.ts";
import type { Component } from "../terminal/primitives.ts";

const component = (text: string): Component => ({ render: () => [text] });

test("overlay manager owns stacking and restores focus on hide", () => {
	const base = component("base");
	const first = component("first");
	const second = component("second");
	let focused: Component | null = base;
	let renders = 0;
	const overlays = new OverlayManager({
		getFocus: () => focused,
		setFocus: component => {
			focused = component;
		},
		requestRender: () => {
			renders++;
		},
	});
	const firstHandle = overlays.show(first);
	const secondHandle = overlays.show(second);

	firstHandle.focus();
	expect(focused).toBe(first);
	secondHandle.focus();
	expect(overlays.topmost()?.component).toBe(second);
	secondHandle.hide();
	expect(focused).toBe(first);
	expect(overlays.entries.map(entry => entry.component)).toEqual([first]);
	expect(renders).toBeGreaterThan(0);
});

test("hidden and non-capturing overlays do not own focus", () => {
	const base = component("base");
	const hidden = Object.assign(component("hidden"), { visible: false });
	let focused: Component | null = base;
	const overlays = new OverlayManager({
		getFocus: () => focused,
		setFocus: component => {
			focused = component;
		},
		requestRender: () => {},
	});

	const hiddenHandle = overlays.show(hidden);
	hiddenHandle.focus();
	expect(focused).toBe(base);
	overlays.show(component("passive"), { nonCapturing: true });
	expect(overlays.topmost()).toBeUndefined();
});
