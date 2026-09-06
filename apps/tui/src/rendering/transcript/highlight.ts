// Adapt the runtime highlighter's fixed ANSI sheet to the active TUI palette.
// Keep the runtime's parsed-code cache independent of theme changes.
import {
	highlight as runtimeHighlight,
	highlightAuto as runtimeHighlightAuto,
} from "@logician/log-runtime/formatting";
import { type ThemeColor, theme } from "../../terminal/theme.ts";

const syntaxColors: Record<string, ThemeColor> = {
	141: "jsonKeyword",
	114: "jsonString",
	245: "muted",
	179: "jsonNumber",
	111: "jsonKey",
	81: "accent",
	220: "jsonNumber",
	203: "error",
	244: "jsonPunctuation",
	147: "jsonKeyword",
	208: "jsonNumber",
};

function themed(value: string): string {
	const plain = theme.fgRaw("mdCodeBlock");
	return value
		.split("\n")
		.map(
			line =>
				plain +
				line
					.replace(/\x1b\[38;5;(\d+)m/g, (ansi, index: string) => {
						const token = syntaxColors[index];
						return token ? theme.fgRaw(token) : ansi;
					})
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
