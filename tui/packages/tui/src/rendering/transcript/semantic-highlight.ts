import {
	highlightAutoWithSheet,
	highlightWithSheet,
	type HighlightResult,
} from "@logician/agent-core/tools/shared/syntax-highlighter.ts";
import { theme } from "./semantic-markup.ts";
import type { ThemeColor } from "../../terminal/theme.ts";

const colors: Record<string, ThemeColor> = {
	keyword: "jsonKey",
	string: "jsonString",
	comment: "dim",
	number: "jsonNumber",
	function: "active",
	"class name": "accent",
	built_in: "warning",
	literal: "jsonKeyword",
	punctuation: "jsonPunctuation",
	operator: "thinkingText",
	regex: "warning",
	"template expression": "thinkingText",
	attribute: "jsonNumber",
	doctype: "dim",
	"xml tag": "jsonKey",
	property: "active",
	meta: "dim",
	shebang: "dim",
	subst: "warning",
	symbol: "jsonNumber",
	type: "accent",
	params: "warning",
	title: "active",
	section: "accent",
	link: "active",
	code: "jsonString",
	addition: "diffAdded",
	deletion: "diffRemoved",
};

const SEMANTIC_SHEET = Object.fromEntries(
	Object.entries(colors).map(([token, color]) => [
		token,
		(text: string) => theme.fg(color, text),
	]),
);

export function highlight(code: string, language: string): HighlightResult {
	return highlightWithSheet(code, language, SEMANTIC_SHEET);
}

export function highlightAuto(code: string): HighlightResult {
	return highlightAutoWithSheet(code, SEMANTIC_SHEET);
}
