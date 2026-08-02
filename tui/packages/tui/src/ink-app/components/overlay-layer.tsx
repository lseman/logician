import { Box } from "ink";
import React, { useMemo } from "react";
import type { Component, OverlayOptions } from "../../terminal/core.ts";
import { isEntryVisible } from "../../terminal/frame-layout.ts";
import { RawLines } from "./raw-lines.tsx";

interface OverlayEntry {
	component: Component;
	options?: OverlayOptions;
	hidden: boolean;
	focusOrder: number;
}

export interface OverlayLayerProps {
	overlayStack: readonly OverlayEntry[];
	termWidth: number;
	transcriptHeight: number;
	renderTick: number;
}

/**
 * Composites the overlay stack on top of the app shell as real absolutely
 * positioned Boxes, honoring each overlay's anchor (center/bottom/top) and
 * width/align rules. aboveInput-anchored entries are not handled here --
 * AppShell renders those inline as part of the fixed dock instead, since
 * they participate in the composer's layout rather than floating over it.
 */
export function OverlayLayer({
	overlayStack,
	termWidth,
	transcriptHeight,
	renderTick,
}: OverlayLayerProps): React.ReactElement {
	const visibleEntries = useMemo(
		() =>
			overlayStack.filter(
				(e) => e.options?.anchor !== "aboveInput" && isEntryVisible(e),
			),
		[overlayStack],
	);

	return (
		<>
			{visibleEntries.map((entry) => (
				<OverlayBox
					// biome-ignore lint/suspicious/noArrayIndexKey: overlay stack order is stable per frame; entries don't carry a stable id
					key={entry.focusOrder}
					entry={entry}
					termWidth={termWidth}
					transcriptHeight={transcriptHeight}
					renderTick={renderTick}
				/>
			))}
		</>
	);
}

function OverlayBox({
	entry,
	termWidth,
	transcriptHeight,
	renderTick,
}: {
	entry: OverlayEntry;
	termWidth: number;
	transcriptHeight: number;
	renderTick: number;
}): React.ReactElement {
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
	const overlayLines = useMemo(
		() => entry.component.render(Math.max(1, overlayWidth)),
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[entry.component, overlayWidth, renderTick],
	);
	const overlayHeight = Math.min(
		overlayLines.length,
		entry.options?.maxHeight || 999,
	);

	let top = 0;
	switch (entry.options?.anchor) {
		case "center":
			top = Math.max(0, Math.floor((transcriptHeight - overlayHeight) / 2));
			break;
		case "bottom":
			top = Math.max(0, transcriptHeight - overlayHeight);
			break;
		default:
			top = 0;
			break;
	}

	const left = leftAligned
		? 0
		: Math.max(2, Math.floor((termWidth - overlayWidth) / 2));

	return (
		<Box position="absolute" top={top} left={left} flexDirection="column">
			<RawLines lines={overlayLines} />
		</Box>
	);
}
