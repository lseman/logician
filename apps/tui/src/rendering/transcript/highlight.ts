// Adapt the runtime highlighter's fixed ANSI sheet to the active TUI palette.
// Keep the runtime's parsed-code cache independent of theme changes.
import {
	highlight as runtimeHighlight,
	highlightAuto as runtimeHighlightAuto,
} from "@logician/log-runtime/formatting";
import { type ThemeColor, theme } from "../../terminal/theme.ts";

// Keys are the fixed 256-color codes emitted by DARK_SHEET in
// log-runtime's syntax-highlighter.ts — keep the two in sync.
const syntaxColors: Record<string, ThemeColor> = {
	141: "jsonKeyword", // keyword
	114: "jsonString", // string
	245: "dim", // comment, doctype, meta, shebang — quieter than punctuation
	179: "jsonNumber", // number, attribute, symbol
	111: "jsonKey", // function, property, title, link
	81: "accent", // class name, type, section
	220: "text", // built_in, params — plain, not number-colored
	203: "jsonKeyword", // literal (true/false/null), deletion — reads as a constant, not an error
	244: "muted", // punctuation — stays legible, distinct from dimmer comments
	147: "jsonPunctuation", // operator, template expression — structural, not keyword-weight
	208: "jsonNumber", // regex, subst
};

function themed(value: string): string {
	const plain = theme.fgRaw("mdCodeBlock");
	return value
		.split("\n")
		.map(
			line =>
				plain +
				line
					// biome-ignore lint/suspicious/noControlCharactersInRegex: Terminal rendering intentionally recognizes ANSI control bytes.
					.replace(/\x1b\[38;5;(\d+)m/g, (ansi, index: string) => {
						const token = syntaxColors[index];
						return token ? theme.fgRaw(token) : ansi;
					})
					// biome-ignore lint/suspicious/noControlCharactersInRegex: Terminal rendering intentionally recognizes ANSI control bytes.
					.replace(/\x1b\[39m/g, plain) +
				"\x1b[39m",
		)
		.join("\n");
}

export function highlight(code: string, language: string) {
	const result = runtimeHighlight(code, language);
	return { ...result, value: themed(result.value) };
}

export function highlightAuto(code: string) {
	const result = runtimeHighlightAuto(code);
	return { ...result, value: themed(result.value) };
}
