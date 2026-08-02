import { render } from "ink";
import React, { useEffect, useState } from "react";
import type { LogicianTUI } from "../app/tui.ts";
import type { TUIComponentsFrame } from "../terminal/core.ts";
import { InkTUIView } from "./ink-tui-view.tsx";

function Root({ app }: { app: LogicianTUI }): React.ReactElement | null {
	const [state, setState] = useState<{ frame: TUIComponentsFrame; tick: number } | null>(
		null,
	);

	useEffect(() => {
		// buildLayout() + enableMouse() + tui.start() run here (post-mount) so
		// TUI's first onComponentsFrame call -- which happens synchronously
		// inside start() -- lands after Ink is ready to receive it.
		let tick = 0;
		app.tui.setOnComponentsFrame((frame) => {
			tick += 1;
			setState({ frame, tick });
		});
		app.start();
		return () => {
			void app.stop();
		};
		// app is expected to be stable (constructed once by the caller before
		// mount); re-running this effect would restart the whole app.
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	if (!state) return null;
	return <InkTUIView tui={app.tui} frame={state.frame} renderTick={state.tick} />;
}

/** Mount a fully-wired LogicianTUI (constructed with externalIO TUI options) under Ink. */
export function mountLogicianTui(app: LogicianTUI): ReturnType<typeof render> {
	return render(<Root app={app} />, { alternateScreen: true });
}
