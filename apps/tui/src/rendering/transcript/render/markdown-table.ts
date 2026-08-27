// ── Transcript markdown/table rendering ─────────────────────────────────────
// Block-level markdown line rendering (code fences, tables, JSON) and the
// table layout helpers it depends on. No instance state.

import { highlight, highlightAuto } from "@logician/log-runtime/formatting";
import { BOLD, DIM, RESET, visibleWidth } from "../../../terminal/core.ts";
import { theme } from "../../../terminal/theme.ts";
import { wrapText } from "../layout.ts";
import {
	escapeMarkdownTableCell,
	extractLangFromFence,
	formatJsonLine,
	renderInline,
	renderMarkdownLine,
} from "../text-utils.ts";

/**
 * Cache for syntax-highlighted code blocks.
 *
 * Keyed by `${lang}|${hash(content)}` — during streaming the same code
 * blocks are re-highlighted every frame even though their content hasn't
 * changed. This avoids the expensive highlighter calls (which may fork a
 * tree-sitter binary) on unchanged blocks.
 *
 * Bounded at 512 entries (FIFO). Typical transcripts have <20 code blocks
 * so overflow is rare.
 */
const highlightCache = new Map<string, string>();
const highlightCacheOrder: string[] = [];
const HIGHLIGHT_CACHE_MAX = 512;

/** Simple djb2 hash for cache keys (fast, no crypto deps). */
function hashString(s: string): number {
	let hash = 5381;
	for (let i = 0; i < s.length; i++) {
		hash = ((hash << 5) + hash + s.charCodeAt(i)) | 0;
	}
	return hash;
}

function cachedHighlight(content: string, lang: string | null): string {
	const key = lang
		? `${lang}|${hashString(content)}`
		: `auto|${hashString(content)}`;
	const hit = highlightCache.get(key);
	if (hit !== undefined) return hit;

	let result = content;
	try {
		result = lang
			? highlight(content, lang).value
			: highlightAuto(content).value;
	} catch {
		/* keep raw content on failure */
	}

	/* Insert / update cache */
	if (highlightCache.has(key)) {
		/* key already exists — just in case the first attempt failed and now
		   succeeded, update the cached value. */
		highlightCache.set(key, result);
	} else {
		if (highlightCacheOrder.length >= HIGHLIGHT_CACHE_MAX) {
			const evicted = highlightCacheOrder.shift();
			if (evicted !== undefined) highlightCache.delete(evicted);
		}
		highlightCache.set(key, result);
		highlightCacheOrder.push(key);
	}
	return result;
}

export function renderMarkdownLines(
	text: string,
	maxLen: number,
	_streaming: boolean,
	baseColor = theme.fg("assistantText", ""),
	firstLinePrefix = "",
): string[] {
	const lines: string[] = [];
	const rawLines = text.split("\n");
	let inCodeBlock = false;
	let codeContent = "";
	let codeBlockLang: string | null = null;
	let prevEmptyLine = false;
	const bg = theme.bg("mdCodeBlockBg", "");
	const bgReset = RESET;
	for (let li = 0; li < rawLines.length; li++) {
		const rawLine = rawLines[li];

		if (rawLine.startsWith("```")) {
			if (inCodeBlock) {
				// Flush code block — use cached highlight when content unchanged
				const lang = codeBlockLang || null;
				const renderedCode = cachedHighlight(codeContent, lang);
				for (const cl of renderedCode.split("\n")) {
					lines.push(`${bg}  ${cl}${bgReset}`);
				}
				const count = codeContent.split("\n").length;
				lines.push(
					`${bg}${DIM}  └─ ${count} line${count === 1 ? "" : "s"}${bgReset}`,
				);
				codeContent = "";
				codeBlockLang = null;
				inCodeBlock = false;
			} else {
				inCodeBlock = true;
				codeBlockLang = extractLangFromFence(rawLine);
				lines.push(`${bg}${DIM}  ┌─ ${codeBlockLang || "code"}${bgReset}`);
			}
			continue;
		}

		if (inCodeBlock) {
			codeContent += (codeContent ? "\n" : "") + rawLine;
			continue;
		}

		if (isMemorySummaryRow(rawLine)) {
			const rows: string[][] = [];
			while (li < rawLines.length && isMemorySummaryRow(rawLines[li])) {
				const parsed = parseMemorySummaryRow(rawLines[li]);
				if (parsed) rows.push(parsed);
				li++;
			}
			li--;

			const renderedTable = renderTable(
				[
					"| ID | Time | Type | Title |",
					"| --- | --- | --- | --- |",
					...rows.map(
						row => `| ${row.map(escapeMarkdownTableCell).join(" | ")} |`,
					),
				],
				maxLen,
			);
			for (const line of renderedTable) lines.push(line);
			continue;
		}

		if (isTableStart(rawLines, li)) {
			const tableLines: string[] = [];
			tableLines.push(rawLines[li]);
			tableLines.push(rawLines[li + 1]);
			li += 2;
			while (li < rawLines.length && isTableRow(rawLines[li])) {
				tableLines.push(rawLines[li]);
				li++;
			}
			li--;

			const renderedTable = renderTable(tableLines, maxLen);
			for (let ti = 0; ti < renderedTable.length; ti++) {
				const line = renderedTable[ti];
				lines.push(line);
			}
			continue;
		}

		const jsonLines = formatJsonLine(rawLine);
		if (jsonLines) {
			for (let ji = 0; ji < jsonLines.length; ji++) {
				const line =
					ji === 0 ? firstLinePrefix + jsonLines[ji] : `  ${jsonLines[ji]}`;
				// No streaming cursor
				lines.push(line);
			}
			continue;
		}

		const effectiveColor = baseColor;
		const rendered = renderMarkdownLine(rawLine, effectiveColor);
		const wrapped = wrapText(rendered, maxLen);
		if (wrapped.length === 0 || (wrapped.length === 1 && wrapped[0] === "")) {
			if (!prevEmptyLine) {
				lines.push(`  ${effectiveColor} ${RESET}`);
			}
			prevEmptyLine = true;
			continue;
		}
		prevEmptyLine = false;
		for (let wi = 0; wi < wrapped.length; wi++) {
			const seg = `${effectiveColor}${wrapped[wi]}${RESET}`;
			const line =
				wi === 0
					? firstLinePrefix + seg
					: `  ${effectiveColor}${wrapped[wi]}${RESET}`;
			// No streaming cursor
			lines.push(line);
		}
	}

	if (inCodeBlock && codeContent) {
		// Streaming code block — also cached so repeated renders of the
		// same partial content skip re-highlighting.
		const renderedCode = cachedHighlight(codeContent, codeBlockLang);
		for (const cl of renderedCode.split("\n")) {
			lines.push(`${bg}  ${cl}${bgReset}`);
		}
		lines.push(`${bg}${DIM}  └─ streaming${bgReset}`);
	}

	return lines;
}

function isMemorySummaryRow(line: string): boolean {
	return parseMemorySummaryRow(line) !== null;
}

function parseMemorySummaryRow(line: string): string[] | null {
	const match = line.match(
		/^([A-Za-z]?\d+)\s+((?:\d{1,2}:\d{2}[ap])|")\s+(\S+)\s+(.+)$/,
	);
	if (match) return [match[1], match[2], match[3], match[4]];
	const sessionMatch = line.match(/^(S\d+)\s+(.+)$/);
	if (sessionMatch) return [sessionMatch[1], "", "", sessionMatch[2]];
	return null;
}

// Strip ANSI escape codes for plain-text analysis (table detection, etc.)
function stripAnsi(s: string): string {
	return s.replace(/\x1b\[[0-9;]*m/g, "");
}

function isTableStart(lines: string[], index: number): boolean {
	return (
		isTableRow(lines[index] || "") && isTableSeparator(lines[index + 1] || "")
	);
}

function isTableRow(line: string): boolean {
	const plain = stripAnsi(line);
	const trimmed = plain.trim();
	return trimmed.includes("|") && trimmed.split("|").length >= 3;
}

function isTableSeparator(line: string): boolean {
	const cells = splitTableRow(line);
	if (cells.length < 2) return false;
	return cells.every(cell => /^:?-+:?$/.test(stripAnsi(cell.trim())));
}

function splitTableRow(line: string): string[] {
	const plain = stripAnsi(line);
	const trimmed = plain.trim();
	const withoutEdges = trimmed.replace(/^\|/, "").replace(/\|$/, "");
	const cells: string[] = [];
	let current = "";
	let escaped = false;
	for (const ch of withoutEdges) {
		if (escaped) {
			current += ch === "|" ? "|" : `\\${ch}`;
			escaped = false;
			continue;
		}
		if (ch === "\\") {
			escaped = true;
			continue;
		}
		if (ch === "|") {
			cells.push(current.trim());
			current = "";
			continue;
		}
		current += ch;
	}
	if (escaped) current += "\\";
	cells.push(current.trim());
	return cells;
}

function renderTable(rawLines: string[], maxLen: number): string[] {
	if (rawLines.length < 2) return rawLines;

	const header = splitTableRow(rawLines[0]);
	const rows = rawLines.slice(2).map(line => splitTableRow(line));
	const columnCount = Math.max(header.length, ...rows.map(row => row.length));
	if (columnCount < 2) return rawLines;

	const normalizedRows = [header, ...rows].map(row => {
		const next = row.slice(0, columnCount);
		while (next.length < columnCount) next.push("");
		return next;
	});

	const minColumnWidth = 8;
	const gapWidth = 3 * (columnCount - 1) + 4;
	const usableWidth = Math.max(columnCount * minColumnWidth, maxLen - gapWidth);
	const naturalWidths = Array.from({ length: columnCount }, (_, col) => {
		return Math.max(
			minColumnWidth,
			...normalizedRows.map(row => visibleWidth(row[col] || "")),
		);
	});

	let widths = naturalWidths.slice();
	const total = widths.reduce((a, b) => a + b, 0);
	if (total > usableWidth) {
		const widest = widths.indexOf(Math.max(...widths));
		widths = widths.map((width, idx) => {
			if (idx === widest) return width;
			return Math.min(width, 24);
		});
		const reserved = widths.reduce(
			(sum, width, idx) => (idx === widest ? sum : sum + width),
			0,
		);
		widths[widest] = Math.max(minColumnWidth, usableWidth - reserved);
		while (widths.reduce((a, b) => a + b, 0) > usableWidth) {
			const shrinkIndex = widths.indexOf(Math.max(...widths));
			if (widths[shrinkIndex] <= minColumnWidth) break;
			widths[shrinkIndex]--;
		}
	}

	const borderColor = theme.fgRaw("borderMuted");
	const headerColor = theme.fgRaw("assistantText");
	const rowColor = theme.fgRaw("assistantText");
	const altRowColor = theme.fgRaw("dim");

	// Frame lines use full Unicode box-drawing, colored via a single ANSI code per line
	const frame = (left: string, fill: string, join: string, right: string) => {
		return (
			borderColor +
			left +
			widths.map(width => fill.repeat(width + 2)).join(join) +
			right +
			RESET
		);
	};

	const out: string[] = [];
	out.push(frame("┌", "─", "┬", "┐"));
	out.push(...renderTableRow(header, widths, headerColor, true, borderColor));
	out.push(frame("├", "─", "┼", "┤"));
	for (let ri = 0; ri < rows.length; ri++) {
		const row = rows[ri];
		const normalized = row.slice(0, columnCount);
		while (normalized.length < columnCount) normalized.push("");
		const rowColor_ = ri % 2 === 0 ? rowColor : altRowColor;
		out.push(
			...renderTableRow(normalized, widths, rowColor_, false, borderColor),
		);
	}
	out.push(frame("└", "─", "┴", "┘"));
	return out;
}

function renderTableRow(
	cells: string[],
	widths: number[],
	color: string,
	header: boolean,
	borderColor: string,
): string[] {
	const wrappedCells = cells.map((cell, idx) =>
		wrapPlainCell(cell, widths[idx]),
	);
	const rowHeight = Math.max(1, ...wrappedCells.map(cell => cell.length));
	const lines: string[] = [];
	const cellColor = header ? color + BOLD : color;
	const border = borderColor;

	for (let row = 0; row < rowHeight; row++) {
		const parts: string[] = [];
		for (let idx = 0; idx < widths.length; idx++) {
			const raw = wrappedCells[idx][row] || "";
			const styled = renderInline(raw, cellColor);
			const padding = " ".repeat(Math.max(0, widths[idx] - visibleWidth(raw)));
			parts.push(` ${styled}${padding} `);
		}
		// Unicode vertical bars to match the frame — gives continuous box borders.
		lines.push(`${border}│${parts.join("│")}${border}│`);
	}

	return lines;
}

function wrapPlainCell(text: string, width: number): string[] {
	if (!text) return [""];
	const words = text.split(/\s+/);
	const lines: string[] = [];
	let current = "";

	const pushHardWrapped = (word: string) => {
		let remaining = word;
		while (visibleWidth(remaining) > width) {
			lines.push(remaining.slice(0, width));
			remaining = remaining.slice(width);
		}
		current = remaining;
	};

	for (const word of words) {
		if (!current) {
			if (visibleWidth(word) > width) pushHardWrapped(word);
			else current = word;
		} else if (visibleWidth(current) + 1 + visibleWidth(word) <= width) {
			current += ` ${word}`;
		} else {
			lines.push(current);
			if (visibleWidth(word) > width) pushHardWrapped(word);
			else current = word;
		}
	}
	if (current) lines.push(current);
	return lines.length > 0 ? lines : [""];
}
