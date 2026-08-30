const ESCAPE = String.fromCharCode(27);
const CSI_ESCAPE_KEY = new RegExp(`${ESCAPE}\\[27(?:;1)?u`, "g");
const CSI_CTRL_KEY = new RegExp(`${ESCAPE}\\[(\\d+);([56])u`, "g");
const CSI_NAVIGATION_KEY = new RegExp(
	`${ESCAPE}\\[(?:5~|6~|1;5H|1;5F|H|F)`,
	"g",
);
const SGR_MOUSE_EVENT = new RegExp(
	`${ESCAPE}\\[<(\\d+);(\\d+);(\\d+)([Mm])`,
	"y",
);

export interface MousePosition {
	column: number;
	row: number;
}

export interface ParsedMouseInput {
	clicks: MousePosition[];
	consumedLength: number;
	wheel: MousePosition & { ticks: number };
}

/**
 * Translate Kitty CSI-u Ctrl+letter reports back to the C0 bytes consumed by
 * existing keybindings. Ctrl+I and Ctrl+M stay encoded so they remain
 * distinguishable from Tab and Enter and can reach their dedicated bindings.
 */
export function normalizeKeyboardInput(data: string): string {
	return data
		.replace(CSI_ESCAPE_KEY, ESCAPE)
		.replace(CSI_CTRL_KEY, (sequence, codepointText: string) => {
			const codepoint = Number(codepointText);
			const lowerCodepoint =
				codepoint >= 65 && codepoint <= 90 ? codepoint + 32 : codepoint;
			if (lowerCodepoint === 105 || lowerCodepoint === 109) return sequence;
			if (lowerCodepoint < 96 || lowerCodepoint > 127) return sequence;
			return String.fromCharCode(lowerCodepoint & 0x1f);
		});
}

/** Return repeated global-navigation sequences only when they fill the chunk. */
export function splitNavigationBatch(data: string): string[] | null {
	const matches = data.match(CSI_NAVIGATION_KEY);
	return matches && matches.length > 1 && matches.join("") === data
		? matches
		: null;
}

/** Decode the SGR mouse prefix of a terminal input chunk without routing it. */
export function parseSgrMouseInput(data: string): ParsedMouseInput | null {
	if (!data.startsWith(`${ESCAPE}[<`)) return null;
	const clicks: MousePosition[] = [];
	let consumedLength = 0;
	let ticks = 0;
	let column = 0;
	let row = 0;
	SGR_MOUSE_EVENT.lastIndex = 0;
	let match = SGR_MOUSE_EVENT.exec(data);
	while (match) {
		const button = Number(match[1]);
		const eventColumn = Number(match[2]) - 1;
		const eventRow = Number(match[3]) - 1;
		if (button === 64 || button === 65) {
			ticks += button === 64 ? -1 : 1;
			column = eventColumn;
			row = eventRow;
		} else if (button === 0 && match[4] === "M") {
			clicks.push({ column: eventColumn, row: eventRow });
		}
		consumedLength += match[0].length;
		SGR_MOUSE_EVENT.lastIndex = consumedLength;
		match = SGR_MOUSE_EVENT.exec(data);
	}
	return { clicks, consumedLength, wheel: { column, row, ticks } };
}
