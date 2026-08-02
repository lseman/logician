import { useStdin } from "ink";
import React, { useEffect } from "react";
import { normalizeKeyboardInput, type TUI, type TUIComponentsFrame } from "../terminal/core.ts";
import { AppShell } from "./app-shell.tsx";

export interface InkTUIViewProps {
	/**
	 * A TUI (with onComponentsFrame set) constructed by the host app (e.g.
	 * LogicianTUI), with its components already wired (setScrollableComponent,
	 * setInputBarComponent, ...). This component does not construct
	 * TUI itself so the host retains full control over construction order and
	 * component wiring.
	 */
	tui: TUI;
	/** The frame most recently produced by tui's onComponentsFrame callback. */
	frame: TUIComponentsFrame | null;
	/** Bumped by the host whenever any underlying component's content changes. */
	renderTick: number;
}

/**
 * Lays out and paints TUI's components via Ink's own Box/flexbox (AppShell),
 * and forwards raw stdin into TUI's own input routing (mouse, scroll keys,
 * overlay stack). TUI must be started by the host (tui.start()) after this
 * component -- or a component wrapping it -- has mounted, since TUI calls
 * onComponentsFrame synchronously from start().
 */
export function InkTUIView({ tui, frame, renderTick }: InkTUIViewProps): React.ReactElement | null {
	const { stdin, setRawMode, isRawModeSupported } = useStdin();

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

	if (!frame) return null;
	return <AppShell tui={tui} frame={frame} renderTick={renderTick} />;
}
