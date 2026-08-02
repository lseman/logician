import { Box, Text, useCursor } from "ink";
import React, { useEffect, useMemo } from "react";
import {
	CURSOR_MARKER,
	clampLineToWidth,
	visibleWidth,
	type TUIComponentsFrame,
} from "../terminal/core.ts";
import { isEntryVisible } from "../terminal/frame-layout.ts";
import { RawLines } from "./components/raw-lines.tsx";
import { OverlayLayer } from "./components/overlay-layer.tsx";

export interface AppShellProps {
	frame: TUIComponentsFrame;
	/** Bumped by the host app whenever any underlying component's content changes. */
	renderTick: number;
}

/**
 * Alternate-screen app shell: scrollable transcript (grows to fill remaining
 * space) above a fixed dock (pinned region + input bar + status bar), with
 * overlays composited on top. Box/flexbox owns the transcript-vs-dock height
 * split; TUI (terminal/core.ts) only supplies the components and their
 * input/scroll/overlay state.
 */
export function AppShell({ frame, renderTick }: AppShellProps): React.ReactElement {
	const {
		termWidth,
		termHeight,
		scrollableComponent,
		inputBarComponent,
		fixedBottomComponent,
		fixedAboveInputComponent,
		overlayStack,
		showHardwareCursor,
	} = frame;
	const width = Math.max(1, termWidth - 1);
	const { setCursorPosition } = useCursor();

	const aboveInputLines = useMemo(() => {
		const lines = fixedAboveInputComponent?.render(width) ?? [];
		// Interactive selectors (settings, model picker, ...) participate in the
		// fixed composer stack instead of floating over transcript content --
		// same contract as frame-layout.ts's renderAboveInputOverlays. Only the
		// most recently focused one owns the region.
		const selectorEntries = overlayStack.filter(
			(e) => e.options?.anchor === "aboveInput" && isEntryVisible(e),
		);
		if (selectorEntries.length > 0) {
			const entry = selectorEntries.reduce((latest, candidate) =>
				candidate.focusOrder > latest.focusOrder ? candidate : latest,
			);
			const rendered = entry.component.render(width);
			const maxHeight = entry.options?.maxHeight ?? rendered.length;
			for (const line of rendered.slice(0, maxHeight)) {
				const clamped = clampLineToWidth(line, width);
				lines.push(clamped + " ".repeat(Math.max(0, width - visibleWidth(clamped))));
			}
		}
		return lines;
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
	}, [fixedAboveInputComponent, overlayStack, width, renderTick]);
	const inputLines = useMemo(
		() => inputBarComponent?.render(width) ?? [" ".repeat(width)],
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[inputBarComponent, width, renderTick],
	);
	const statusLines = useMemo(
		() => fixedBottomComponent?.render(width) ?? [" ".repeat(width)],
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[fixedBottomComponent, width, renderTick],
	);

	const dockHeight =
		1 + aboveInputLines.length + inputLines.length + 1 + statusLines.length;
	const transcriptHeight = Math.max(1, termHeight - dockHeight);

	useEffect(() => {
		scrollableComponent?.setViewportHeight(transcriptHeight);
	}, [scrollableComponent, transcriptHeight]);

	const transcriptLines = useMemo(
		() => scrollableComponent?.render(width) ?? Array(transcriptHeight).fill(" ".repeat(width)),
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[scrollableComponent, width, transcriptHeight, renderTick],
	);

	// The InputBar marks the edit position with CURSOR_MARKER so the hardware
	// cursor can be parked exactly there. Its on-screen row is the input
	// region's offset within the dock (transcript + separator + above-input
	// lines) plus the marker's row within the input bar's own output.
	const markerInInput = useMemo(() => {
		for (let row = 0; row < inputLines.length; row++) {
			const idx = inputLines[row].indexOf(CURSOR_MARKER);
			if (idx >= 0) {
				return { row, col: visibleWidth(inputLines[row].slice(0, idx)) };
			}
		}
		return null;
	}, [inputLines]);

	// Called directly in the render body, not an effect: Ink's own useCursor
	// example does the same, since cursor position must be current for this
	// exact commit rather than lagging a frame behind (an effect would apply
	// the previous frame's position, visible as the cursor sitting one row
	// off -- e.g. on the separator above the input bar -- whenever content
	// shifts the input region's row offset between renders).
	if (markerInInput && showHardwareCursor) {
		setCursorPosition({
			x: markerInInput.col,
			// Ink's cursor suffix measures y from the implicit line immediately
			// after its output. Fullscreen output has no trailing newline, so its
			// public cursor coordinate is one greater than our zero-based layout
			// row. Passing the layout row directly parks the terminal cursor on
			// the line above the marked input position.
			y: transcriptHeight + 1 + aboveInputLines.length + markerInInput.row + 1,
		});
	} else {
		setCursorPosition(undefined);
	}

	const displayInputLines = useMemo(
		() => inputLines.map((line) => line.replace(CURSOR_MARKER, "")),
		[inputLines],
	);

	return (
		<Box flexDirection="column" width={termWidth} height={termHeight}>
			<Box flexDirection="column" height={transcriptHeight} overflow="hidden">
				<RawLines lines={transcriptLines} />
			</Box>
			<Text dimColor>{"─".repeat(width)}</Text>
			{aboveInputLines.length > 0 && (
				<Box flexDirection="column">
					<RawLines lines={aboveInputLines} />
				</Box>
			)}
			<Box flexDirection="column">
				<RawLines lines={displayInputLines} />
			</Box>
			<Text dimColor>{"─".repeat(width)}</Text>
			<Box flexDirection="column">
				<RawLines lines={statusLines} />
			</Box>
			<OverlayLayer
				overlayStack={overlayStack}
				termWidth={termWidth}
				transcriptHeight={transcriptHeight}
				renderTick={renderTick}
			/>
		</Box>
	);
}
