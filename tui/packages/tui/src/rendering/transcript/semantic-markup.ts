import type { InkTextRow, InkTextSpan } from "../../terminal/core.ts";
import {
	theme as terminalTheme,
	type ThemeBg,
	type ThemeColor,
} from "../../terminal/theme.ts";

export const RESET = "\ue000";
export const BOLD = "\ue001";
export const DIM = "\ue002";
export const UNDERLINE = "\ue003";
export const ITALIC = "\ue004";
export const BOLD_OFF = "\ue005";
export const FG_RESET = "\ue006";
export const BG_RESET = "\ue007";

const styleByToken = new Map<string, Omit<InkTextSpan, "text">>();
const tokenByStyle = new Map<string, string>();
let nextToken = 0xe100;

function styleToken(style: Omit<InkTextSpan, "text">): string {
	const key = JSON.stringify(style);
	const existing = tokenByStyle.get(key);
	if (existing) return existing;
	if (nextToken > 0xf8ff) throw new Error("Transcript semantic style token space exhausted");
	const token = String.fromCodePoint(nextToken++);
	tokenByStyle.set(key, token);
	styleByToken.set(token, style);
	return token;
}

export function isSemanticStyleToken(character: string): boolean {
	const code = character.codePointAt(0) ?? 0;
	return code >= 0xe000 && code <= 0xf8ff;
}

export function semanticMarkupToInkRow(value: string): InkTextRow {
	const spans: InkTextSpan[] = [];
	let style: Omit<InkTextSpan, "text"> = {};
	let text = "";
	const flush = (): void => {
		if (!text) return;
		spans.push({ text, ...style });
		text = "";
	};
	for (const character of value) {
		if (!isSemanticStyleToken(character)) {
			text += character;
			continue;
		}
		flush();
		if (character === RESET) style = {};
		else if (character === BOLD) style = { ...style, bold: true };
		else if (character === DIM) {
			style = { ...style, color: terminalTheme.inkColor("dim") };
		}
		else if (character === UNDERLINE) style = { ...style, underline: true };
		else if (character === ITALIC) style = { ...style, italic: true };
		else if (character === BOLD_OFF) style = { ...style, bold: false };
		else if (character === FG_RESET) {
			const { color: _color, ...rest } = style;
			style = rest;
		} else if (character === BG_RESET) {
			const { backgroundColor: _backgroundColor, ...rest } = style;
			style = rest;
		} else style = { ...style, ...styleByToken.get(character) };
	}
	flush();
	return spans;
}

export const theme = {
	fg(color: ThemeColor, text: string): string {
		return `${styleToken({ color: terminalTheme.inkColor(color) })}${text}${RESET}`;
	},
	fgRaw(color: ThemeColor): string {
		return styleToken({ color: terminalTheme.inkColor(color) });
	},
	bg(color: ThemeBg, text: string): string {
		return `${styleToken({ backgroundColor: terminalTheme.inkBackgroundColor(color) })}${text}${RESET}`;
	},
	fgAsBg(color: ThemeColor): string {
		return styleToken({ backgroundColor: terminalTheme.inkColor(color) });
	},
	inkColor(color: ThemeColor): string | undefined {
		return terminalTheme.inkColor(color);
	},
};
