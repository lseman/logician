import { RESET, visibleWidth } from "../../terminal/core.ts";

/**
 * Parse one ANSI SGR/OSC/APC escape sequence starting at `index`, or return
 * null if there isn't one there. Used to walk text char-by-char while still
 * treating each escape sequence as a single atomic unit.
 */
function scanAnsiSequence(text: string, index: number): string | null {
	if (text[index] !== "\x1b") return null;
	const start = index;
	let i = index + 1;
	const next = text[i];
	if (next === "[") {
		i++;
		while (i < text.length) {
			const finalCode = text.charCodeAt(i);
			i++;
			if (finalCode >= 0x40 && finalCode <= 0x7e) break;
		}
	} else if (next === "]" || next === "_") {
		i++;
		while (i < text.length) {
			if (text[i] === "\x07") {
				i++;
				break;
			}
			if (text[i] === "\x1b" && text[i + 1] === "\\") {
				i += 2;
				break;
			}
			i++;
		}
	} else {
		i++;
	}
	return text.slice(start, i);
}

/**
 * A wrappable unit: either a run of non-whitespace characters (a "word") or
 * a single hard-wrapped slice of a word too long to fit any line. `text` is
 * exactly the original characters/escape-sequences for this unit — no codes
 * are added or removed. `trailingCodes` is whatever SGR codes are open
 * immediately after it, so a break inserted right after this unit knows what
 * to reopen for whatever comes next.
 */
interface Unit {
	text: string;
	width: number;
	trailingCodes: string;
}

/** Split a line into whitespace-delimited units, tracking active SGR codes
 * through escape sequences and whitespace alike (colors span across spaces
 * even though spaces themselves aren't part of any unit's text). */
function scanUnits(rawLine: string, initialCodes: string): Unit[] {
	const units: Unit[] = [];
	let activeCodes = initialCodes;
	let current = "";
	let currentWidth = 0;
	let inWord = false;
	let index = 0;

	const flush = () => {
		if (!inWord) return;
		units.push({
			text: current,
			width: currentWidth,
			trailingCodes: activeCodes,
		});
		current = "";
		currentWidth = 0;
		inWord = false;
	};

	while (index < rawLine.length) {
		const seq = scanAnsiSequence(rawLine, index);
		if (seq) {
			current += seq;
			inWord = true;
			if (/^\x1b\[0m$/.test(seq)) activeCodes = "";
			else if (seq.startsWith("\x1b[")) activeCodes += seq;
			index += seq.length;
			continue;
		}
		const codePoint = rawLine.codePointAt(index);
		const character =
			codePoint === undefined
				? rawLine[index]
				: String.fromCodePoint(codePoint);
		if (/\s/.test(character)) {
			flush();
			index += character.length;
			continue;
		}
		inWord = true;
		current += character;
		currentWidth += visibleWidth(character);
		index += character.length;
	}
	flush();
	return units;
}

/**
 * Hard-wrap a single unit (no internal whitespace) that alone exceeds
 * `width`, splitting on visible-width boundaries. `leadingCodes` are
 * reopened as an explicit prefix on every slice after the first — never
 * relying on the original text to already contain them, since after the
 * first slice it never does.
 */
function hardWrapUnit(unit: Unit, leadingCodes: string, width: number): Unit[] {
	const slices: Unit[] = [];
	// activeCodes tracks what's open for bookkeeping (trailingCodes), but is
	// deliberately never written into `slice` itself — the caller (wrapText)
	// is the single place that reopens a carried-over color, by prefixing
	// currentOpenCodes onto the first unit of a new line. A slice's own text
	// only ever contains codes that change *within* it, same as an ordinary
	// (non-hard-wrapped) unit's text does.
	let slice = "";
	let sliceWidth = 0;
	let activeCodes = leadingCodes;
	let index = 0;
	while (index < unit.text.length) {
		const seq = scanAnsiSequence(unit.text, index);
		if (seq) {
			slice += seq;
			if (/^\x1b\[0m$/.test(seq)) activeCodes = "";
			else if (seq.startsWith("\x1b[")) activeCodes += seq;
			index += seq.length;
			continue;
		}
		const codePoint = unit.text.codePointAt(index);
		const character =
			codePoint === undefined
				? unit.text[index]
				: String.fromCodePoint(codePoint);
		const characterWidth = visibleWidth(character);
		if (slice && sliceWidth + characterWidth > width) {
			slices.push({
				text: slice,
				width: sliceWidth,
				trailingCodes: activeCodes,
			});
			slice = "";
			sliceWidth = 0;
		}
		slice += character;
		sliceWidth += characterWidth;
		index += character.length;
	}
	if (slice)
		slices.push({ text: slice, width: sliceWidth, trailingCodes: activeCodes });
	return slices;
}

/**
 * Word-wrap text to a max visible width, one visual line per array entry.
 * Preserves ANSI color across every line break — word-boundary wraps and,
 * when a single word alone exceeds the width, mid-word hard wraps too — by
 * explicitly reopening whatever SGR codes were active at each break as a
 * prefix on the next line, and closing an still-open color with RESET at
 * the end of a line, since each produced line is written as an independent
 * terminal row and colors don't otherwise persist across separate rows.
 */
export function wrapText(text: string, maxLineLength: number): string[] {
	const width = Math.max(1, Math.floor(maxLineLength));
	const lines: string[] = [];

	let carryCodes = "";
	for (const rawLine of text.split("\n")) {
		if (visibleWidth(rawLine) <= width) {
			lines.push(rawLine);
			const units = scanUnits(rawLine, carryCodes);
			if (units.length) carryCodes = units[units.length - 1].trailingCodes;
			continue;
		}

		const rawUnits = scanUnits(rawLine, carryCodes);
		let leadingForNext = carryCodes;
		const units: Unit[] = [];
		for (const unit of rawUnits) {
			if (unit.width > width) {
				const slices = hardWrapUnit(unit, leadingForNext, width);
				units.push(...slices);
			} else {
				units.push(unit);
			}
			leadingForNext = unit.trailingCodes;
		}

		let current = "";
		let currentWidth = 0;
		let currentOpenCodes = carryCodes;
		let hasContent = false;

		for (const unit of units) {
			if (hasContent && currentWidth + 1 + unit.width > width) {
				lines.push(currentOpenCodes ? current + RESET : current);
				current = "";
				currentWidth = 0;
				hasContent = false;
			}
			if (!hasContent && currentOpenCodes) {
				current += currentOpenCodes;
			}
			current += (hasContent ? " " : "") + unit.text;
			currentWidth += (hasContent ? 1 : 0) + unit.width;
			currentOpenCodes = unit.trailingCodes;
			hasContent = true;
		}
		if (hasContent) {
			lines.push(currentOpenCodes ? current + RESET : current);
		}
		carryCodes = currentOpenCodes;
	}

	return lines;
}
