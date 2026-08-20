// ── Transcript generic content/text rendering ───────────────────────────────
// Pi-style line-numbered content rendering, syntax-highlighted file content,
// and the shared truncation helpers used throughout the transcript renderers.

import { highlight, highlightAuto } from "@logician/agent-core/formatting";
import { DIM, RESET, visibleWidth } from "../../../terminal/core.ts";
import { theme } from "../../../terminal/theme.ts";
import { wrapText } from "../layout.ts";

// ── Pi-style line-numbered content rendering ────────────────────────────
// Shows content with line numbers, collapsed to a preview when expanded,
// with a line-number gutter like Pi's write tool.

export function renderPiContent(
	text: string,
	width: number,
	totalLines: number,
): string[] {
	const lines: string[] = [];
	const bg = theme.bg("mdCodeBlockBg", "");
	const bgReset = RESET;
	const gutterColor = theme.fgRaw("dim");
	const contentColor = theme.fgRaw("assistantText");

	// Determine how many lines to show in collapsed mode
	const collapsedPreviewLines = 8;
	const showAll = totalLines <= collapsedPreviewLines;

	const rawLines = text.split("\n");
	const displayLines = showAll
		? rawLines
		: rawLines.slice(0, collapsedPreviewLines);

	// Calculate gutter width (line number column)
	const gutterWidth = String(totalLines).length + 1;
	const availableContentWidth = Math.max(20, width - gutterWidth - 2);

	for (let i = 0; i < displayLines.length; i++) {
		const lineNum = i + 1;
		const rawLine = displayLines[i];
		const formatted = rawLine.length ? rawLine.replace(/\t/g, "    ") : " ";

		// Truncate line to fit available width
		const displayContent =
			visibleWidth(formatted) > availableContentWidth
				? wrapText(formatted, availableContentWidth)
				: [formatted];

		for (let wi = 0; wi < displayContent.length; wi++) {
			const content = displayContent[wi];
			const numStr = String(lineNum + (wi > 0 ? 0 : 0)).padStart(
				gutterWidth - 1,
				" ",
			);
			lines.push(
				`${bg}${gutterColor}${numStr}│${RESET}${bg}${contentColor}${content}${bgReset}`,
			);
		}
	}

	// Add truncation hint if collapsed
	if (!showAll) {
		const remaining = totalLines - collapsedPreviewLines;
		lines.push(
			`${bg}${DIM}  └─ ${remaining} more lines · ctrl+o to expand${RESET}`,
		);
	}

	return lines;
}

/** Render file content with syntax highlighting and line numbers (Pi-style). */
export function renderFileContent(
	text: string,
	width: number,
	totalLines: number,
	language: string | undefined,
	expanded: boolean,
): string[] {
	const lines: string[] = [];
	const bg = theme.bg("mdCodeBlockBg", "");
	const bgReset = RESET;
	const gutterColor = theme.fgRaw("dim");
	const plainColor = theme.fgRaw("assistantText");

	const collapsedPreviewLines = 8;
	const showAll = expanded || totalLines <= collapsedPreviewLines;

	const rawLines = text.split("\n");
	const displayLines = showAll
		? rawLines
		: rawLines.slice(0, collapsedPreviewLines);

	// Calculate gutter width
	const gutterWidth = String(totalLines).length + 1;
	const availableContentWidth = Math.max(20, width - gutterWidth - 2);

	if (language) {
		// Highlighted rendering: split each line by ANSI sequences, apply line numbers
		const highlighted = language
			? highlight(text, language)
			: highlightAuto(text);

		// Parse highlighted output into lines, each line may have ANSI color spans
		const hlLines = highlighted.value.split("\n");
		for (let i = 0; i < displayLines.length; i++) {
			const lineNum = i + 1;
			const hlLine = hlLines[i] || "";

			// If highlighted output is empty for this line, fall back to plain
			const content = hlLine.replace(/\x1b\[[\d;]*m/g, "");
			const displayContent =
				visibleWidth(content) > availableContentWidth
					? wrapText(content, availableContentWidth)
					: [content];

			for (let wi = 0; wi < displayContent.length; wi++) {
				const displayLine = displayContent[wi];
				const numStr = String(lineNum).padStart(gutterWidth - 1, " ");
				// Extract ANSI spans from highlighted line at the same wrap position
				let hlContent = extractHlSpan(hlLine, displayLine);
				if (!hlContent) hlContent = plainColor + displayLine + plainColor;
				lines.push(
					`${bg}${gutterColor}${numStr}│${RESET}${bg}${hlContent}${bgReset}`,
				);
			}
		}

		// Add language label in the gutter area if collapsed
		if (!showAll) {
			const remaining = totalLines - collapsedPreviewLines;
			lines.push(
				`${bg}${DIM}  └─ ${remaining} more lines · ctrl+o to expand${RESET}`,
			);
		}
	} else {
		// No language detection — plain text with line numbers
		for (let i = 0; i < displayLines.length; i++) {
			const lineNum = i + 1;
			const rawLine = displayLines[i];
			const formatted = rawLine.length ? rawLine.replace(/\t/g, "    ") : " ";

			const displayContent =
				visibleWidth(formatted) > availableContentWidth
					? wrapText(formatted, availableContentWidth)
					: [formatted];

			for (let wi = 0; wi < displayContent.length; wi++) {
				const content = displayContent[wi];
				const numStr = String(lineNum).padStart(gutterWidth - 1, " ");
				lines.push(
					`${bg}${gutterColor}${numStr}│${RESET}${bg}${plainColor}${content}${bgReset}`,
				);
			}
		}
		if (!showAll) {
			const remaining = totalLines - collapsedPreviewLines;
			lines.push(
				`${bg}${DIM}  └─ ${remaining} more lines · ctrl+o to expand${RESET}`,
			);
		}
	}

	return lines;
}

/** Extract the portion of a highlighted line corresponding to a display line. */
export function extractHlSpan(
	hlLine: string,
	displayLine: string,
): string | null {
	if (!hlLine || hlLine.trim().length === 0) return null;
	// If the plain text of the hl line matches the display line, use it directly
	const stripped = hlLine.replace(/\x1b\[[\d;]*m/g, "");
	if (stripped === displayLine) return hlLine;
	// Otherwise approximate: if displayLine is shorter (wrapped), take first N chars of hl
	if (displayLine.length < hlLine.length) {
		// Count visible chars needed
		const visLen = displayLine.length;
		let idx = 0;
		let visible = 0;
		let inSeq = false;
		while (idx < hlLine.length && visible < visLen) {
			if (hlLine[idx] === "\x1b") {
				inSeq = true;
			}
			if (inSeq && hlLine[idx] === "m") {
				inSeq = false;
			} else if (!inSeq) {
				visible++;
			}
			idx++;
		}
		return hlLine.slice(0, idx);
	}
	return hlLine;
}

export function truncateText(text: string, maxMessageLength: number): string {
	if (text.length <= maxMessageLength) return text;
	return withTruncationMarker(text.slice(0, maxMessageLength));
}

/** Appends the shared "content cut off" marker used by every truncation path. */
export function withTruncationMarker(text: string): string {
	return `${text}\n\n${DIM}[truncated]${RESET}`;
}
