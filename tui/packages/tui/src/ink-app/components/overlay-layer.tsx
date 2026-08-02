import { Box } from "ink";
import React, { useMemo } from "react";
import { ansiToInkTextRow } from "../../terminal/core.ts";
import { InkTextRows } from "./status-region.tsx";
import { ListOverlay, listOverlayHeight } from "./list-overlay.tsx";
import {
	isEntryVisible,
	type OverlayEntry,
} from "../overlay-visibility.ts";
import { hasInkOverlayModel } from "../../overlays/ink-overlay-model.ts";

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
		() => hasInkOverlayModel(entry.component)
			? []
			: entry.component.render?.(Math.max(1, overlayWidth)) ?? [],
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[entry.component, overlayWidth, renderTick],
	);
	const overlayHeight = Math.min(
		hasInkOverlayModel(entry.component)
			? listOverlayHeight(entry.component.getInkOverlayModel())
			: overlayLines.length,
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
			{hasInkOverlayModel(entry.component) ? (
				<ListOverlay
					model={entry.component.getInkOverlayModel()}
					width={overlayWidth}
				/>
			) : (
				<InkTextRows rows={overlayLines.map(ansiToInkTextRow)} />
			)}
		</Box>
	);
}
