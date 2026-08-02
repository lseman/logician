import { Box, useCursor, useStdin } from "ink";
import React, { useEffect } from "react";
import {
	CURSOR_MARKER,
	normalizeKeyboardInput,
	type RenderedFrame,
	type TUI,
} from "../terminal/core.ts";
import { RawLines } from "./components/raw-lines.tsx";

export interface InkTUIViewProps {
	/**
	 * A TUI constructed with { externalIO: true, onFrame: ... } by the host
	 * app (e.g. LogicianTUI), with its components already wired
	 * (setScrollableComponent, setInputBarComponent, addChild, ...). This
	 * component does not construct TUI itself so the host retains full
	 * control over construction order and component wiring, exactly like
	 * the legacy renderer.
	 */
	tui: TUI;
	/** The frame most recently produced by tui's onFrame callback. */
	frame: RenderedFrame | null;
}

/**
 * Paints TUI frames via Ink instead of the legacy cell-diff writer, and
 * forwards raw stdin into TUI's own input routing (mouse, scroll keys,
 * overlay stack) unchanged. TUI must be started by the host (tui.start())
 * after this component -- or a component wrapping it -- has mounted, since
 * TUI's externalIO mode calls onFrame synchronously.
 */
export function InkTUIView({ tui, frame }: InkTUIViewProps): React.ReactElement | null {
	const { stdin, setRawMode, isRawModeSupported } = useStdin();
	const { setCursorPosition } = useCursor();

	useEffect(() => {
		if (isRawModeSupported) setRawMode(true);
		const onData = (chunk: Buffer | string) => {
			const str = Buffer.isBuffer(chunk) ? chunk.toString("utf-8") : chunk;
			tui.feedInput(normalizeKeyboardInput(str));
			tui.requestRender();
		};
		stdin?.on("data", onData);
		return () => {
			stdin?.off("data", onData);
			if (isRawModeSupported) setRawMode(false);
		};
	}, [stdin, setRawMode, isRawModeSupported, tui]);

	useEffect(() => {
		if (!frame) return;
		if (frame.cursorRow >= 0 && frame.showHardwareCursor) {
			setCursorPosition({ x: frame.cursorCol, y: frame.cursorRow });
		} else {
			setCursorPosition(undefined);
		}
	}, [frame, setCursorPosition]);

	if (!frame) return null;

	const displayLines = frame.lines.map((line) =>
		line.includes(CURSOR_MARKER) ? line.replace(CURSOR_MARKER, "") : line,
	);

	return (
		<Box flexDirection="column" width={frame.termWidth} height={frame.termHeight}>
			<RawLines lines={displayLines} />
		</Box>
	);
}
