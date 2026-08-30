import { appendFileSync } from "node:fs";

const ESCAPE = String.fromCharCode(27);
const KITTY_KEY = new RegExp(
	`${ESCAPE}\\[(\\d+)(?::\\d*)?(?::\\d+)?(?:;(\\d+)(?::(\\d+))?)?u`,
	"g",
);
const MODIFY_OTHER_KEYS_CTRL = new RegExp(`${ESCAPE}\\[27;([56]);(\\d+)~`, "g");
const CSI_NAVIGATION_KEY = new RegExp(
	`${ESCAPE}\\[(?:5~|6~|1;5H|1;5F|H|F)`,
	"g",
);
const SGR_MOUSE_EVENT = new RegExp(
	`${ESCAPE}\\[<(\\d+);(\\d+);(\\d+)([Mm])`,
	"y",
);

/** Opt-in byte-level trace for diagnosing terminal/proxy input differences. */
export function logInputTrace(
	stage: string,
	data: string,
	metadata: Record<string, unknown> = {},
): void {
	const path = process.env.LOGICIAN_TUI_INPUT_LOG;
	if (!path) return;
	try {
		appendFileSync(
			path,
			`${JSON.stringify({
				time: new Date().toISOString(),
				stage,
				hex: Buffer.from(data, "utf8").toString("hex"),
				length: Buffer.byteLength(data),
				...metadata,
			})}\n`,
		);
	} catch {
		// Diagnostics must never affect input handling.
	}
}

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
		.replace(
			KITTY_KEY,
			(
				sequence,
				codepointText: string,
				modifierText?: string,
				eventText?: string,
			) => {
				// Kitty event type 3 is a key release, not another key press.
				if (eventText === "3") return "";
				const codepoint = Number(codepointText);
				const modifiers = Math.max(0, Number(modifierText ?? "1") - 1);
				// Lock state occupies separate modifier bits. Test only the Ctrl bit,
				// matching pi (e.g. Ctrl+C with Num Lock is modifier 133, not 5).
				const ctrl = (modifiers & 4) !== 0;
				if (codepoint === 27 && !ctrl) return ESCAPE;
				if (!ctrl) return sequence;
				const lowerCodepoint =
					codepoint >= 65 && codepoint <= 90 ? codepoint + 32 : codepoint;
				// Preserve Ctrl+I/M encodings so Tab/Enter remain distinguishable.
				if (lowerCodepoint === 105 || lowerCodepoint === 109) return sequence;
				if (lowerCodepoint < 96 || lowerCodepoint > 127) return sequence;
				return String.fromCharCode(lowerCodepoint & 0x1f);
			},
		)
		.replace(
			MODIFY_OTHER_KEYS_CTRL,
			(sequence, _modifier: string, codepointText: string) => {
				const codepoint = Number(codepointText);
				if (codepoint === 27) return ESCAPE;
				const lowerCodepoint =
					codepoint >= 65 && codepoint <= 90 ? codepoint + 32 : codepoint;
				if (lowerCodepoint < 96 || lowerCodepoint > 127) return sequence;
				return String.fromCharCode(lowerCodepoint & 0x1f);
			},
		);
}

const BRACKETED_PASTE_START = `${ESCAPE}[200~`;
const BRACKETED_PASTE_END = `${ESCAPE}[201~`;

/** Reassembles fragmented terminal sequences and splits batched stdin reads. */
export class TerminalInputBuffer {
	private buffer = "";
	private escapeTimer: ReturnType<typeof setTimeout> | null = null;

	constructor(
		private readonly emit: (sequence: string) => void,
		private readonly escapeTimeoutMs = 25,
	) {}

	process(data: string): void {
		if (!data) return;
		this.clearEscapeTimer();
		this.buffer += data;
		this.drain();
	}

	destroy(): void {
		this.clearEscapeTimer();
		this.buffer = "";
	}

	private drain(): void {
		while (this.buffer.length > 0) {
			if (!this.buffer.startsWith(ESCAPE)) {
				const codepoint = this.buffer.codePointAt(0);
				if (codepoint === undefined) return;
				const character = String.fromCodePoint(codepoint);
				this.buffer = this.buffer.slice(character.length);
				this.emit(character);
				continue;
			}

			if (this.buffer.startsWith(BRACKETED_PASTE_START)) {
				const end = this.buffer.indexOf(BRACKETED_PASTE_END);
				if (end < 0) return;
				const length = end + BRACKETED_PASTE_END.length;
				this.emit(this.buffer.slice(0, length));
				this.buffer = this.buffer.slice(length);
				continue;
			}

			if (this.buffer.length === 1) {
				this.escapeTimer = setTimeout(() => {
					this.escapeTimer = null;
					if (this.buffer === ESCAPE) {
						this.buffer = "";
						this.emit(ESCAPE);
					}
				}, this.escapeTimeoutMs);
				return;
			}

			const length = this.escapeSequenceLength();
			if (length === 0) return;
			this.emit(this.buffer.slice(0, length));
			this.buffer = this.buffer.slice(length);
		}
	}

	private escapeSequenceLength(): number {
		const kind = this.buffer[1];
		if (kind === "[") {
			for (let index = 2; index < this.buffer.length; index++) {
				const code = this.buffer.charCodeAt(index);
				if (code >= 0x40 && code <= 0x7e) return index + 1;
			}
			return 0;
		}
		if (kind === "O") return this.buffer.length >= 3 ? 3 : 0;
		if (kind === "]") {
			const bel = this.buffer.indexOf("\x07", 2);
			const st = this.buffer.indexOf(`${ESCAPE}\\`, 2);
			if (bel >= 0 && (st < 0 || bel < st)) return bel + 1;
			return st >= 0 ? st + 2 : 0;
		}
		if (kind === "P" || kind === "_") {
			const st = this.buffer.indexOf(`${ESCAPE}\\`, 2);
			return st >= 0 ? st + 2 : 0;
		}
		// Legacy Alt+key sequence.
		return 2;
	}

	private clearEscapeTimer(): void {
		if (!this.escapeTimer) return;
		clearTimeout(this.escapeTimer);
		this.escapeTimer = null;
	}
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
