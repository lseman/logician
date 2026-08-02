// ── Frame layout ─────────────────────────────────────────────────────────────
// Pure layout/overlay-composition logic shared by both renderer backends: the
// legacy absolute-cursor cell-diff TUI (terminal/core.ts) and the Ink-backed
// InkTUI (ink-app/ink-tui.ts). Produces the final terminal-height line array
// plus cursor marker position; each backend decides how to paint that array.

import {
	CURSOR_MARKER,
	clampLineToWidth,
	visibleWidth,
	type Component,
	type OverlayOptions,
	type Scrollable,
} from "./core.ts";

export interface OverlayEntry {
	component: Component;
	options?: OverlayOptions;
	hidden: boolean;
	focusOrder: number;
}

export interface FixedLayoutInput {
	termWidth: number;
	termHeight: number;
	scrollableComponent: Scrollable | null;
	inputBarComponent: Component | null;
	fixedBottomComponent: Component | null;
	fixedAboveInputComponent: Component | null;
	overlayStack: readonly OverlayEntry[];
}

export interface FixedLayoutResult {
	lines: string[];
	transcriptHeight: number;
	/** Row/col of the CURSOR_MARKER in the pre-strip composed frame, or -1 if absent. */
	markerRow: number;
	markerCol: number;
	/** Fallback hardware-cursor row when no marker was emitted (keeps cursor off the footer). */
	fallbackRow: number;
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

function renderAboveInputOverlays(
	overlayStack: readonly OverlayEntry[],
	termWidth: number,
): string[] {
	const entries = overlayStack.filter(
		(entry) => entry.options?.anchor === "aboveInput" && isEntryVisible(entry),
	);
	if (entries.length === 0) return [];

	// Only the most recently focused selector owns the composer region.
	const entry = entries.reduce((latest, candidate) =>
		candidate.focusOrder > latest.focusOrder ? candidate : latest,
	);
	const width = Math.max(1, termWidth - 1);
	const rendered = entry.component.render(width);
	const maxHeight = entry.options?.maxHeight ?? rendered.length;
	return rendered.slice(0, maxHeight).map((line) => {
		const clamped = clampLineToWidth(line, width);
		return clamped + " ".repeat(Math.max(0, termWidth - visibleWidth(clamped)));
	});
}

function composeOverlays(
	lines: string[],
	overlayStack: readonly OverlayEntry[],
	termWidth: number,
	transcriptHeight: number,
): string[] {
	const result = [...lines];

	const visibleEntries = overlayStack.filter(
		(e) => e.options?.anchor !== "aboveInput" && isEntryVisible(e),
	);

	for (const entry of visibleEntries) {
		const leftAligned = entry.options?.align === "left";
		const overlayWidth = leftAligned
			? Math.max(1, termWidth)
			: Math.max(
					40,
					Math.min(
						termWidth - 8,
						entry.options?.maxHeight ? termWidth * 0.6 : termWidth - 8,
					),
				);
		const overlayLines = entry.component.render(Math.max(1, overlayWidth));
		const overlayHeight = Math.min(
			overlayLines.length,
			entry.options?.maxHeight || 999,
		);

		let row = 0;
		switch (entry.options?.anchor) {
			case "center":
				row = Math.max(0, Math.floor((transcriptHeight - overlayHeight) / 2));
				break;
			case "bottom":
				row = Math.max(0, transcriptHeight - overlayHeight);
				break;
			default:
				row = 0;
				break;
		}

		const margin = leftAligned
			? 0
			: Math.max(2, Math.floor((termWidth - overlayWidth) / 2));

		for (let i = 0; i < overlayHeight; i++) {
			const idx = row + i;
			if (idx >= 0 && idx < result.length) {
				const srcLine = overlayLines[i] || "";
				const srcVis = visibleWidth(srcLine);
				const basePad = " ".repeat(margin);
				const afterPad = " ".repeat(Math.max(0, termWidth - margin - srcVis));
				result[idx] = basePad + srcLine + afterPad;
			}
		}
	}

	return result;
}

/**
 * Build one full terminal-height frame: scrollable transcript + separators +
 * pinned-above-input region + input bar + separator + status bar, with
 * overlays composited on top. Pure function of current component state — no
 * painting or diffing. Also drives the scrollable component's viewport
 * height as a side effect, matching the legacy renderer's contract (the
 * scrollable component clips/slices its own content to that height).
 */
export function buildFixedLayoutFrame(input: FixedLayoutInput): FixedLayoutResult {
	const {
		termWidth,
		termHeight,
		scrollableComponent,
		inputBarComponent,
		fixedBottomComponent,
		fixedAboveInputComponent,
		overlayStack,
	} = input;

	let inputLines: string[];
	try {
		inputLines = inputBarComponent
			? inputBarComponent.render(termWidth)
			: [" ".repeat(termWidth)];
	} catch (_e: unknown) {
		inputLines = [" ".repeat(termWidth)];
	}

	let statusLines: string[];
	try {
		statusLines = fixedBottomComponent
			? fixedBottomComponent.render(termWidth)
			: [" ".repeat(termWidth)];
	} catch (_e: unknown) {
		statusLines = [" ".repeat(termWidth)];
	}

	const inputHeight = Math.max(1, inputLines.length);
	const statusHeight = Math.max(1, statusLines.length);

	let aboveInputLines: string[] = [];
	try {
		aboveInputLines = fixedAboveInputComponent
			? fixedAboveInputComponent.render(termWidth)
			: [];
	} catch (_e: unknown) {
		aboveInputLines = [];
	}
	aboveInputLines.push(...renderAboveInputOverlays(overlayStack, termWidth));
	const aboveInputHeight = aboveInputLines.length;

	const transcriptHeight = Math.max(
		1,
		termHeight - 2 - aboveInputHeight - inputHeight - statusHeight,
	);
	const transcriptWidth = termWidth;

	const lines: string[] = [];

	if (scrollableComponent) {
		scrollableComponent.setViewportHeight(transcriptHeight);
		let transcriptLines: string[];
		try {
			transcriptLines = scrollableComponent.render(transcriptWidth);
		} catch (_e: unknown) {
			transcriptLines = Array(transcriptHeight)
				.fill(0)
				.map(() => " ".repeat(transcriptWidth));
		}
		const totalLines = Math.max(
			transcriptLines.length,
			scrollableComponent.totalHeight,
		);
		const maxScroll = Math.max(0, totalLines - transcriptHeight);
		const scrollOff = Math.min(
			maxScroll,
			Math.max(0, scrollableComponent.scrollOffset),
		);
		const visibleLines = (
			scrollableComponent as unknown as { rendersViewport?: boolean }
		).rendersViewport
			? transcriptLines
			: transcriptLines.slice(scrollOff, scrollOff + transcriptHeight);

		while (lines.length < transcriptHeight) {
			lines.push(
				lines.length < visibleLines.length
					? visibleLines[lines.length]
					: " ".repeat(termWidth),
			);
		}
	} else {
		for (let i = 0; i < transcriptHeight; i++) {
			lines.push(" ".repeat(termWidth));
		}
	}

	lines.push(`\x1b[38;5;236m${"─".repeat(termWidth)}\x1b[0m`);

	for (let i = 0; i < aboveInputHeight; i++) {
		lines.push(aboveInputLines[i] || " ".repeat(termWidth));
	}

	for (let i = 0; i < inputHeight; i++) {
		lines.push(inputLines[i] || " ".repeat(termWidth));
	}

	lines.push(`\x1b[38;5;236m${"─".repeat(termWidth)}\x1b[0m`);

	for (let i = 0; i < statusHeight; i++) {
		lines.push(statusLines[i] || " ".repeat(termWidth));
	}

	while (lines.length < termHeight) {
		lines.push(" ".repeat(termWidth));
	}

	const finalLines = composeOverlays(lines, overlayStack, termWidth, transcriptHeight);

	let markerRow = -1;
	let markerCol = 0;
	for (let row = 0; row < finalLines.length; row++) {
		const line = finalLines[row];
		if (line.includes(CURSOR_MARKER)) {
			markerRow = row;
			markerCol = visibleWidth(line.slice(0, line.indexOf(CURSOR_MARKER)));
			break;
		}
	}

	const fallbackRow = Math.min(termHeight, transcriptHeight + 2 + aboveInputHeight);

	return { lines: finalLines, transcriptHeight, markerRow, markerCol, fallbackRow };
}
