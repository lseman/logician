import { Box, Text } from "ink";
import React, { useMemo } from "react";
import {
	type TUIComponentsFrame,
	type TUI,
} from "../terminal/core.ts";
import { OverlayLayer } from "./components/overlay-layer.tsx";
import { ComposerRegion, composerHeight } from "./components/composer-region.tsx";
import { PinnedRegion } from "./components/pinned-region.tsx";
import { StatusRegion, statusHeight } from "./components/status-region.tsx";
import { ListOverlay, listOverlayHeight } from "./components/list-overlay.tsx";
import { isEntryVisible } from "./overlay-visibility.ts";
import type { InkListOverlayModel } from "../overlays/ink-overlay-model.ts";
import { theme } from "../terminal/theme.ts";

export interface AppShellProps {
	tui: TUI;
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
export function AppShell({ tui, frame, renderTick }: AppShellProps): React.ReactElement {
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

	const aboveInput = useMemo((): { rows: import("../terminal/core.ts").InkTextRow[]; model: InkListOverlayModel | null; modelHeight: number } => {
		const rows = fixedAboveInputComponent?.getInkTextRows(width) ?? [];
		let model: InkListOverlayModel | null = null;
		let modelHeight = 0;
		// Interactive selectors (settings, model picker, ...) participate in the
		// fixed composer stack instead of floating over transcript content --
		// Only the most recently focused selector owns the region.
		const selectorEntries = overlayStack.filter(
			(e) => e.options?.anchor === "aboveInput" && isEntryVisible(e),
		);
		if (selectorEntries.length > 0) {
			const entry = selectorEntries.reduce((latest, candidate) =>
				candidate.focusOrder > latest.focusOrder ? candidate : latest,
			);
			model = entry.component.getInkOverlayModel();
			modelHeight = Math.min(
				listOverlayHeight(model),
				entry.options?.maxHeight ?? Number.POSITIVE_INFINITY,
			);
		}
		return { rows, model, modelHeight };
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
	}, [fixedAboveInputComponent, overlayStack, width, renderTick]);
	const aboveInputHeight = aboveInput.rows.length + aboveInput.modelHeight;
	const inputHeight = composerHeight(inputBarComponent, width);
	const footerHeight = statusHeight(fixedBottomComponent, width);

	const dockHeight =
		1 + aboveInputHeight + inputHeight + 1 + footerHeight;
	const transcriptHeight = Math.max(1, termHeight - dockHeight);

	// Transcript rendering and mouse hit-testing both need the current Ink
	// layout during this frame, not one commit later.
	tui.setViewportHeight(transcriptHeight);

	const transcriptRows = useMemo(
		() => scrollableComponent?.getInkTextRows(width) ?? Array(transcriptHeight).fill([{ text: " ".repeat(width) }]),
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[scrollableComponent, width, transcriptHeight, renderTick],
	);

	return (
		<Box flexDirection="column" width={termWidth} height={termHeight}>
			<Box flexDirection="column" height={transcriptHeight} overflow="hidden">
				<PinnedRegion rows={transcriptRows} />
			</Box>
			<Text color={theme.inkColor("separator")}>{"─".repeat(width)}</Text>
			<PinnedRegion rows={aboveInput.rows} />
			{aboveInput.model && (
				<Box height={aboveInput.modelHeight} overflow="hidden">
					<ListOverlay model={aboveInput.model} width={width} />
				</Box>
			)}
			<ComposerRegion
				component={inputBarComponent}
				width={width}
				originY={transcriptHeight + 1 + aboveInputHeight}
				showHardwareCursor={showHardwareCursor}
				renderTick={renderTick}
			/>
			<Text color={theme.inkColor("separator")}>{"─".repeat(width)}</Text>
			<StatusRegion
				component={fixedBottomComponent}
				width={width}
				renderTick={renderTick}
			/>
			<OverlayLayer
				overlayStack={overlayStack}
				termWidth={termWidth}
				transcriptHeight={transcriptHeight}
				renderTick={renderTick}
			/>
		</Box>
	);
}
