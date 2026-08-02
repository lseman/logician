/** Result of parsing a keypress against the standard list-popup bindings. */
export type PopupListNavResult =
	| { type: "move"; delta: number }
	| { type: "confirm" }
	| { type: "close" }
	| null;

/** Shared navigation for native Ink list overlays. */
export function parsePopupListNav(data: string): PopupListNavResult {
	if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
		return { type: "close" };
	}
	if (data === "\r" || data === "\n") return { type: "confirm" };
	if (data === "\x1b[A" || data === "\x1bOA" || data === "k") {
		return { type: "move", delta: -1 };
	}
	if (data === "\x1b[B" || data === "\x1bOB" || data === "j") {
		return { type: "move", delta: 1 };
	}
	if (data === "\x1b[5~") return { type: "move", delta: -8 };
	if (data === "\x1b[6~") return { type: "move", delta: 8 };
	return null;
}
