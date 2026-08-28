import type { Component } from "./primitives.ts";

export type OverlaySize = number | `${number}%`;

export interface OverlayMargin {
	top?: number;
	right?: number;
	bottom?: number;
	left?: number;
}

export type OverlayAnchor =
	| "center"
	| "top-left"
	| "top-center"
	| "top-right"
	| "bottom-left"
	| "bottom-center"
	| "bottom-right"
	| "left-center"
	| "right-center"
	| "top"
	| "bottom"
	| "aboveInput";

export interface OverlayOptions {
	width?: OverlaySize;
	minWidth?: number;
	maxHeight?: OverlaySize;
	anchor?: OverlayAnchor;
	offsetX?: number;
	offsetY?: number;
	row?: OverlaySize;
	col?: OverlaySize;
	margin?: OverlayMargin | number;
	align?: "center" | "left";
	visible?: (termWidth: number, termHeight: number) => boolean;
	nonCapturing?: boolean;
	onClick?: () => void;
}

export interface OverlayHandle {
	hide(): void;
	setHidden(hidden: boolean): void;
	isHidden(): boolean;
	focus(): void;
	unfocus(target?: Component | null): void;
	isFocused(): boolean;
}

export interface OverlayStackEntry {
	component: Component;
	options?: OverlayOptions;
	preFocus: Component | null;
	hidden: boolean;
	focusOrder: number;
}
