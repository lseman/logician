import { Box, Text, useStdout } from "ink";
import React, { useEffect, useMemo, useState } from "react";
import type { Scrollable, Component } from "../terminal/core.ts";
import { RawLines } from "./components/raw-lines.tsx";

export interface AppShellProps {
	/** The scrollable transcript. Owns its own viewport slicing via render(width). */
	transcript: Scrollable;
	/** Optional pinned region directly above the input bar (todo list, selectors). */
	aboveInput?: Component | null;
	/** The input bar. Fixed height at the bottom, above the status bar. */
	inputBar: Component;
	/** The status/footer bar. Fixed height, last row(s). */
	statusBar: Component;
	/** Bumped by the host app whenever any underlying component's content changes. */
	renderTick: number;
}

/**
 * Alternate-screen app shell: scrollable transcript (grows to fill remaining
 * space) above a fixed dock (pinned region + input bar + status bar).
 * Mirrors the fixed-dock/scrollable-transcript split from terminal/core.ts's
 * TUI._doRenderInner, but layout, resize, and diffing are owned by Ink
 * instead of manual row-slicing + absolute cursor addressing.
 */
export function AppShell(props: AppShellProps): React.ReactElement {
	const { transcript, aboveInput, inputBar, statusBar, renderTick } = props;
	const { stdout } = useStdout();
	const [size, setSize] = useState(() => ({
		columns: stdout?.columns ?? 80,
		rows: stdout?.rows ?? 24,
	}));

	useEffect(() => {
		if (!stdout) return;
		const onResize = () => {
			setSize({ columns: stdout.columns ?? 80, rows: stdout.rows ?? 24 });
		};
		stdout.on("resize", onResize);
		return () => {
			stdout.off("resize", onResize);
		};
	}, [stdout]);

	const width = Math.max(1, size.columns - 1);

	// Render fixed-height regions first so their line counts can be subtracted
	// from total terminal rows to get the transcript's viewport height — same
	// two-pass approach the legacy renderer uses, just expressed as effects
	// instead of manual arithmetic threaded through one function.
	const aboveInputLines = useMemo(
		() => aboveInput?.render(width) ?? [],
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[aboveInput, width, renderTick],
	);
	const inputLines = useMemo(
		() => inputBar.render(width),
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[inputBar, width, renderTick],
	);
	const statusLines = useMemo(
		() => statusBar.render(width),
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[statusBar, width, renderTick],
	);

	const dockHeight =
		1 + aboveInputLines.length + inputLines.length + 1 + statusLines.length;
	const transcriptHeight = Math.max(1, size.rows - dockHeight);

	useEffect(() => {
		transcript.setViewportHeight(transcriptHeight);
	}, [transcript, transcriptHeight]);

	const transcriptLines = useMemo(
		() => transcript.render(width),
		// biome-ignore lint/correctness/useExhaustiveDependencies: renderTick is the invalidation signal; component identity is stable
		[transcript, width, transcriptHeight, renderTick],
	);

	return (
		<Box flexDirection="column" width={size.columns} height={size.rows}>
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
				<RawLines lines={inputLines} />
			</Box>
			<Text dimColor>{"─".repeat(width)}</Text>
			<Box flexDirection="column">
				<RawLines lines={statusLines} />
			</Box>
		</Box>
	);
}
