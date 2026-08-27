// ── Transcript text/markup helpers ────────────────────────────────────────────
// Pure text stripping, inline/block markdown rendering, JSON coloring, and
// arg-parsing helpers used by TranscriptDisplay. No instance state.

import { isAbsolute } from "node:path";
import { pathToFileURL } from "node:url";
import { stripAcceptanceReport, stripTextToolCalls } from "@logician/log-core";
import type { AssistantChunk } from "@logician/log-runtime/sessions";
import { BOLD, DIM, RESET } from "../../terminal/core.ts";
import { hyperlink, supportsHyperlinks } from "../../terminal/hyperlinks.ts";
import { theme } from "../../terminal/theme.ts";

const UNDERLINE = "\x1b[4m";

// Heading palette — distinct color + weight per level.
// NOTE: do NOT append RESET in `color` — it is placed around the text later.
const getHeadingStyles = (): Array<{ color: string; deco: string }> => [
	{ color: theme.fgRaw("mdHeading"), deco: BOLD + UNDERLINE },
	{ color: theme.fgRaw("accent"), deco: BOLD },
	{ color: theme.fgRaw("mdHeading"), deco: BOLD },
	{ color: theme.fgRaw("warning"), deco: "" },
	{ color: theme.fgRaw("muted"), deco: "" },
	{ color: theme.fgRaw("dim") + DIM, deco: DIM },
];

// ── Code fence language extraction ────────────────────────────────────────────

export function extractLangFromFence(line: string): string | null {
	const m = /^```(\w+)/.exec(line.trim());
	return m ? m[1].toLowerCase() : null;
}

// ── Embedded reasoning stripping ──────────────────────────────────────────────
export function stripThinkTags(text: string): string {
	if (!text?.includes("<think")) return text;
	return text
		.replace(/<think(?:ing)?>\s*[\s\S]*?<\/think(?:ing)?>\s*/gi, "")
		.replace(/<think(?:ing)?>\s*[\s\S]*$/i, "")
		.trimStart();
}

export function unwrapThinkingChannel(text: string): string {
	return text.replace(/<\/?think(?:ing)?>/gi, "").trim();
}

export function stripThinkingToolMarkup(text: string): string {
	if (!/<(?:tool\\?_call|function\s*=)/i.test(text)) return text;
	return stripTextToolCalls(text)
		.replace(
			/\n*\**\s*<(?:tool\\?_call|function\s*=\s*[a-zA-Z_][\w.-]*)[^>]*>[\s\S]*$/i,
			"",
		)
		.trimEnd();
}

export function stripAcceptanceForDisplay(text: string): string {
	const marker = text.indexOf("```acceptance-report");
	if (marker < 0) return text;
	const stripped = stripAcceptanceReport(text);
	// While the report is still streaming there is no closing fence to strip.
	// Hide the internal report from its opening marker onward.
	return stripped === text
		? text.slice(0, marker).trimEnd()
		: stripped.trimEnd();
}

export function stripInternalHookGuidance(
	text: string | undefined,
): string | undefined {
	if (!text?.includes("-hook>")) return text;
	const visible = text
		.replace(
			/\n*<(?:post-tool-use|pre-tool-use|stop)-hook>[\s\S]*?<\/(?:post-tool-use|pre-tool-use|stop)-hook>\n*/gi,
			"\n",
		)
		.replace(/\n{3,}/g, "\n\n")
		.trimEnd();
	return visible || undefined;
}

// ── Inline markdown renderer ──────────────────────────────────────────────────

interface ParsedPostEditDiagnostic {
	line: number;
	column: number;
	label?: string;
	message: string;
}

export interface PostEditDiagnosticBlock {
	file: string;
	diagnostics: ParsedPostEditDiagnostic[];
}

export function extractPostEditDiagnostics(text: string | undefined): {
	text: string | undefined;
	blocks: PostEditDiagnosticBlock[];
} {
	if (!text?.includes("<post_edit_diagnostics")) {
		return { text, blocks: [] };
	}
	const blocks: PostEditDiagnosticBlock[] = [];
	const cleaned = text.replace(
		/\n*<post_edit_diagnostics\s+file="([^"]*)">([\s\S]*?)<\/post_edit_diagnostics>\n*/gi,
		(_match, fileValue: string, body: string) => {
			const diagnostics = body
				.split("\n")
				.flatMap((line): ParsedPostEditDiagnostic[] => {
					const parsed = /^-\s+.*?:(\d+):(\d+)(?:\s+(.+?))?:\s+(.+)$/.exec(
						line.trim(),
					);
					if (!parsed) return [];
					return [
						{
							line: Number(parsed[1]),
							column: Number(parsed[2]),
							label: parsed[3]?.trim(),
							message: parsed[4].trim(),
						},
					];
				});
			blocks.push({ file: fileValue, diagnostics });
			return "\n";
		},
	);
	const visible = cleaned.replace(/\n{3,}/g, "\n\n").trimEnd();
	return { text: visible || undefined, blocks };
}

function matchTagAt(text: string, i: number): number {
	const m = /^<\/?[A-Za-z][\w-]*(?:\s[^<>]*)?\/?>/.exec(text.slice(i, i + 200));
	return m ? m[0].length : 0;
}

const MARKDOWN_LINK_RE = /^\[([^[\]]*)\]\((\S+?)\)/;

/** Matches a `[text](url)` markdown link at position `i`, if any. */
function matchMarkdownLinkAt(
	text: string,
	i: number,
): { length: number; linkText: string; url: string } | null {
	const m = MARKDOWN_LINK_RE.exec(text.slice(i, i + 2000));
	if (!m) return null;
	return { length: m[0].length, linkText: m[1], url: m[2] };
}

/** Renders a markdown link's visible text + styling, as a real OSC 8
 * hyperlink when the terminal supports it, or `text (url)` otherwise. */
function renderMarkdownLink(
	linkText: string,
	url: string,
	baseColor: string,
): string {
	const styledText =
		theme.fgRaw("mdLink") + UNDERLINE + renderInline(linkText, "") + RESET;
	if (supportsHyperlinks()) return hyperlink(styledText, url) + baseColor;
	if (linkText === url) return styledText + baseColor;
	return `${styledText} ${theme.fgRaw("dim")}(${url})${RESET}${baseColor}`;
}

/**
 * Wraps `displayText` (typically an already-styled rendering of `path`) in a
 * `file://` OSC 8 hyperlink when the terminal supports it and `path` is
 * absolute. Relative paths have no unambiguous filesystem root to resolve
 * against here, so they're left as plain text rather than guessed at.
 * Returns `displayText` unchanged when hyperlinking isn't applicable.
 */
export function hyperlinkedFilePath(path: string, displayText: string): string {
	if (!supportsHyperlinks() || !isAbsolute(path)) return displayText;
	return hyperlink(displayText, pathToFileURL(path).href);
}

export function renderInline(text: string, baseColor: string): string {
	text = decodeInlineDisplayEscapes(text);
	let out = baseColor;
	let i = 0;
	while (i < text.length) {
		if (text[i] === "<") {
			const tag = matchTagAt(text, i);
			if (tag) {
				out +=
					theme.fgRaw("warning") +
					BOLD +
					text.slice(i, i + tag) +
					RESET +
					baseColor;
				i += tag;
				continue;
			}
		}
		if (text.startsWith("```", i)) {
			const end = text.indexOf("```", i + 3);
			if (end !== -1) {
				out +=
					theme.fgRaw("mdCodeBlock") +
					"```" +
					text.slice(i + 3, end) +
					"```" +
					RESET +
					baseColor;
				i = end + 3;
				continue;
			}
		}
		if (text.startsWith("**", i)) {
			const end = text.indexOf("**", i + 2);
			if (end !== -1) {
				out += BOLD + text.slice(i + 2, end) + RESET + baseColor;
				i = end + 2;
				continue;
			}
		}
		if (text[i] === "`") {
			const end = text.indexOf("`", i + 1);
			if (end !== -1) {
				out +=
					theme.fgRaw("mdCode") +
					BOLD +
					"`" +
					text.slice(i + 1, end) +
					"`" +
					RESET +
					baseColor;
				i = end + 1;
				continue;
			}
		}
		if (text[i] === "*" && i + 1 < text.length && text[i + 1] !== "*") {
			const end = text.indexOf("*", i + 1);
			if (end !== -1 && end !== i + 1) {
				out += `\x1b[3m${text.slice(i + 1, end)}${RESET}${baseColor}`;
				i = end + 1;
				continue;
			}
		}
		if (text[i] === "[") {
			const link = matchMarkdownLinkAt(text, i);
			if (link) {
				out += renderMarkdownLink(link.linkText, link.url, baseColor);
				i += link.length;
				continue;
			}
		}
		out += text[i];
		i++;
	}
	return out + RESET;
}

function decodeInlineDisplayEscapes(text: string): string {
	return text
		.replace(
			/&(?:quot|apos|amp|lt|gt|#\d+|#x[\da-f]+);/gi,
			entity => decodeXmlEntity(entity.slice(1, -1)) ?? entity,
		)
		.replace(/\\([!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])/g, "$1");
}

function decodeXmlEntity(entity: string): string | null {
	const named: Record<string, string> = {
		quot: '"',
		apos: "'",
		amp: "&",
		lt: "<",
		gt: ">",
	};
	const normalized = entity.toLowerCase();
	if (normalized in named) return named[normalized];
	const radix = normalized.startsWith("#x") ? 16 : 10;
	const rawCodePoint = normalized.slice(radix === 16 ? 2 : 1);
	const codePoint = Number.parseInt(rawCodePoint, radix);
	if (!Number.isFinite(codePoint) || codePoint < 0 || codePoint > 0x10ffff)
		return null;
	return String.fromCodePoint(codePoint);
}

// ── Block-level markdown ──────────────────────────────────────────────────────

export function renderMarkdownLine(line: string, baseColor: string): string {
	// Headings
	const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
	if (heading) {
		const level = heading[1].length;
		const style = getHeadingStyles()[level - 1];
		const marker = level <= 2 ? "▌ " : "";
		return `${style.color}${marker}${style.deco}${style.color}${renderInlinePlain(heading[2])}${RESET}`;
	}

	// Horizontal rule
	if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
		return `${DIM}${theme.fgRaw("dim")}${"─".repeat(40)}${RESET}`;
	}

	// List items
	const listMatch = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
	if (listMatch) {
		const indent = listMatch[1];
		const lmarker = listMatch[2];
		const rest = listMatch[3];
		if (/^\d/.test(lmarker)) {
			return `${indent}${theme.fgRaw("mdListBullet")}${BOLD}${lmarker}${RESET} ${renderInline(rest, baseColor)}`;
		}
		const glyph = "•";
		return `${indent}${theme.fgRaw("mdListBullet")}${glyph}${RESET} ${renderInline(rest, baseColor)}`;
	}

	// Blockquote
	const quote = line.match(/^(\s*)>\s?(.*)$/);
	if (quote) {
		return `${quote[1]}${DIM}${theme.fgRaw("mdQuote")}▏ ${renderInlinePlain(quote[2])}${RESET}`;
	}

	return renderInline(line, baseColor);
}

// JSON syntax color helpers
const getJsonKeyCol = (): string => theme.fgRaw("jsonKey");
const getJsonStringCol = (): string => theme.fgRaw("jsonString");
const getJsonNumCol = (): string => theme.fgRaw("jsonNumber");
const getJsonKwCol = (): string => theme.fgRaw("jsonKeyword");
const getJsonPunctCol = (): string => theme.fgRaw("jsonPunctuation");

export function formatJsonLine(rawLine: string): string[] | null {
	const trimmed = rawLine.trim();
	if (trimmed.length < 2) return null;
	const first = trimmed[0];
	const last = trimmed[trimmed.length - 1];
	const looksJson =
		(first === "{" && last === "}") || (first === "[" && last === "]");
	if (!looksJson) return null;
	let parsed: unknown;
	try {
		parsed = JSON.parse(trimmed);
	} catch {
		return null;
	}
	if (parsed === null || typeof parsed !== "object") return null;
	if (
		Array.isArray(parsed)
			? parsed.length === 0
			: Object.keys(parsed).length === 0
	) {
		return null;
	}
	const pretty = JSON.stringify(parsed, null, 2);
	return pretty.split("\n").map(colorizeJsonRow);
}

function colorizeJsonRow(row: string): string {
	const indentMatch = row.match(/^(\s*)(.*)$/s);
	const indent = indentMatch?.[1] ?? "";
	let body = indentMatch?.[2] ?? row;

	const keyMatch = body.match(/^("(?:[^"\\]|\\.)*")(\s*:\s*)(.*)$/s);
	let prefix = "";
	if (keyMatch) {
		prefix = `${getJsonKeyCol()}${keyMatch[1]}${RESET}${getJsonPunctCol()}${keyMatch[2]}${RESET}`;
		body = keyMatch[3];
	}

	let trailing = "";
	const commaMatch = body.match(/^(.*?)(,)\s*$/s);
	if (commaMatch) {
		body = commaMatch[1];
		trailing = `${getJsonPunctCol()},${RESET}`;
	}

	let valued: string;
	if (/^".*"$/.test(body)) {
		valued = `${getJsonStringCol()}${body}${RESET}`;
	} else if (/^-?\d/.test(body)) {
		valued = `${getJsonNumCol()}${body}${RESET}`;
	} else if (body === "true" || body === "false" || body === "null") {
		valued = `${getJsonKwCol()}${body}${RESET}`;
	} else if (/^[{}[\]]+$/.test(body)) {
		valued = `${getJsonPunctCol()}${body}${RESET}`;
	} else {
		valued = body;
	}

	return `${indent}${prefix}${valued}${trailing}`;
}

export function stringArg(
	args: Record<string, unknown>,
	key: string,
): string | undefined {
	const value = args[key];
	return typeof value === "string" ? value : undefined;
}

/** Read an early string field from streamed JSON before the full args parse. */
export function streamedStringArg(
	json: string | undefined,
	key: string,
): string | undefined {
	if (!json) return undefined;
	const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	const match = new RegExp(
		`"${escapedKey}"\\s*:\\s*"((?:\\\\.|[^"\\\\])*)"`,
	).exec(json);
	if (!match) return undefined;
	try {
		return JSON.parse(`"${match[1]}"`) as string;
	} catch {
		return match[1];
	}
}

/** Decode a JSON string field even while its closing quote has not arrived yet. */
export function streamedStringArgLive(
	json: string | undefined,
	key: string,
): string | undefined {
	if (!json) return undefined;
	const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	const startMatch = new RegExp(`"${escapedKey}"\\s*:\\s*"`).exec(json);
	if (!startMatch) return undefined;

	const start = startMatch.index + startMatch[0].length;
	let encoded = "";
	let escaped = false;
	for (let index = start; index < json.length; index++) {
		const char = json[index];
		if (char === '"' && !escaped) break;
		encoded += char;
		if (char === "\\" && !escaped) escaped = true;
		else escaped = false;
	}
	if (encoded.endsWith("\\")) encoded = encoded.slice(0, -1);
	try {
		return JSON.parse(`"${encoded}"`) as string;
	} catch {
		return encoded
			.replace(/\\n/g, "\n")
			.replace(/\\r/g, "\r")
			.replace(/\\t/g, "\t")
			.replace(/\\"/g, '"')
			.replace(/\\\\/g, "\\");
	}
}

export function compactText(text: string): string {
	return text.replace(/\s+/g, " ").trim();
}

export function formatDurationMs(ms: number): string {
	return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

export function diffLineColor(line: string): string {
	if (line.startsWith("@@")) return theme.fgRaw("diffHunk");
	if (
		line.startsWith("diff --git") ||
		line.startsWith("index ") ||
		line.startsWith("---") ||
		line.startsWith("+++")
	) {
		return theme.fgRaw("diffMeta");
	}
	if (line.startsWith("+")) return theme.fgRaw("diffAdded");
	if (line.startsWith("-")) return theme.fgRaw("diffRemoved");
	return theme.fgRaw("mdCodeBlock");
}

export function parseJsonMaybe(value: string): unknown | null {
	const trimmed = value.trim();
	if (!trimmed || !/^[[{]/.test(trimmed)) return null;
	try {
		return JSON.parse(trimmed);
	} catch {
		return null;
	}
}

export function isPermissionRejection(value: string): boolean {
	const text = value.toLowerCase();
	return [
		"permission denied",
		"not granted",
		"requires permission",
		"outside allowed",
		"denied",
		"blocked",
		"rejected",
	].some(pattern => text.includes(pattern));
}

export function normalizeEditArgs(
	args: Record<string, unknown>,
): Array<{ oldText: string; newText: string }> {
	const edits: Array<{ oldText: string; newText: string }> = [];
	if (typeof args.old_text === "string" || typeof args.oldText === "string") {
		edits.push({
			oldText: String(args.old_text ?? args.oldText ?? ""),
			newText: String(args.new_text ?? args.newText ?? ""),
		});
	}

	let rawEdits = args.edits;
	if (typeof rawEdits === "string") {
		try {
			rawEdits = JSON.parse(rawEdits);
		} catch {
			rawEdits = undefined;
		}
	}
	if (Array.isArray(rawEdits)) {
		for (const item of rawEdits) {
			if (!item || typeof item !== "object") continue;
			const edit = item as Record<string, unknown>;
			edits.push({
				oldText: String(edit.old_text ?? edit.oldText ?? ""),
				newText: String(edit.new_text ?? edit.newText ?? ""),
			});
		}
	}
	return edits;
}

function renderInlinePlain(text: string): string {
	let out = "";
	let i = 0;
	while (i < text.length) {
		if (text.startsWith("**", i)) {
			const end = text.indexOf("**", i + 2);
			if (end !== -1) {
				out += `${BOLD + text.slice(i + 2, end)}\x1b[22m`;
				i = end + 2;
				continue;
			}
		}
		if (text[i] === "`") {
			const end = text.indexOf("`", i + 1);
			if (end !== -1) {
				out += text.slice(i, end + 1);
				i = end + 1;
				continue;
			}
		}
		out += text[i];
		i++;
	}
	return out;
}

export function escapeMarkdownTableCell(value: string): string {
	return value.replace(/\\/g, "\\\\").replace(/\|/g, "\\|");
}

export function hasStreamingChunk(chunks: AssistantChunk[]): boolean {
	return chunks.some(c => !c.isComplete);
}

export function revisionText(value: string | undefined): string {
	if (!value) return "0:";
	return `${value.length}:${value.slice(-48)}`;
}
