// ── Edit matching (three-tier fuzzy ladder) ──────────────────────────────
// Applies exact text replacements to file content. Matching runs a
// three-tier ladder: exact → whitespace/punctuation-normalized (position-
// mapped back to the original, so untouched regions are never rewritten) →
// line-trimmed with indentation re-application. Pure text transform, no I/O —
// ported verbatim (algorithm unchanged) from coding-agent's tools/edit-file.ts.

export interface Edit {
	oldText: string;
	newText: string;
	replaceAll?: boolean;
}

export interface ApplyEditsResult {
	baseContent: string;
	newContent: string;
}

/** Map smart quotes/dashes and uncommon Unicode spaces to their ASCII forms. */
function normalizeChar(ch: string): string {
	if (/[‘’‚‛]/.test(ch)) return "'";
	if (/[“”„‟]/.test(ch)) return '"';
	if (/[‐‑‒–—―−]/.test(ch)) return "-";
	if (/[  -   　]/.test(ch)) return " ";
	return ch;
}

/**
 * Normalize text for fuzzy matching: trailing whitespace per line is ignored,
 * smart quotes/dashes are normalized, and uncommon Unicode spaces become
 * regular spaces. Character-for-character (no NFKC) so positions can be mapped
 * back to the original content.
 */
export function normalizeForFuzzyMatch(text: string): string {
	return text
		.split("\n")
		.map(line => {
			const trimmed = line.trimEnd();
			let out = "";
			for (const ch of trimmed) out += normalizeChar(ch);
			return out;
		})
		.join("\n");
}

interface NormalizedContent {
	norm: string;
	/** map[i] = index in the original content of norm[i]. */
	map: number[];
}

/** Build the fuzzy-normalized content plus a normalized→original index map. */
function buildNormalizedWithMap(content: string): NormalizedContent {
	const norm: string[] = [];
	const map: number[] = [];
	const lines = content.split("\n");
	let offset = 0;
	for (let li = 0; li < lines.length; li++) {
		const line = lines[li] ?? "";
		const trimmedLength = line.trimEnd().length;
		for (let i = 0; i < trimmedLength; i++) {
			norm.push(normalizeChar(line[i] ?? ""));
			map.push(offset + i);
		}
		if (li < lines.length - 1) {
			norm.push("\n");
			map.push(offset + line.length);
		}
		offset += line.length + 1;
	}
	return { norm: norm.join(""), map };
}

function indexOfAll(haystack: string, needle: string): number[] {
	const out: number[] = [];
	let i = haystack.indexOf(needle);
	while (i !== -1) {
		out.push(i);
		i = haystack.indexOf(needle, i + needle.length);
	}
	return out;
}

function leadingWhitespace(line: string): string {
	return line.slice(0, line.length - line.trimStart().length);
}

/**
 * Shift newText from the indentation the model wrote to the file's actual
 * indentation. Lines nested deeper than the matched block's first line keep
 * their *relative* depth, scaled by the ratio between the model's indent
 * width and the file's — so a 2-space search indent mapping to a 1-tab file
 * indent turns a 4-space nested line into two tabs, not "tab + 2 leftover
 * spaces". Falls back to a flat prefix swap when depth can't be inferred
 * (searchIndent is empty, so there's no unit to scale by).
 */
function reindent(
	newText: string,
	searchIndent: string,
	origIndent: string,
): string {
	if (searchIndent === origIndent) return newText;
	const ratio =
		searchIndent.length > 0 ? origIndent.length / searchIndent.length : 1;
	return newText
		.split("\n")
		.map(line => {
			if (line.trim() === "") return line;
			if (!line.startsWith(searchIndent)) return line;
			const extra = leadingWhitespace(line).slice(searchIndent.length);
			const rest = line.slice(searchIndent.length + extra.length);
			if (extra.length === 0 || searchIndent.length === 0) {
				return origIndent + line.slice(searchIndent.length);
			}
			const extraUnitChar =
				origIndent.length > 0 ? origIndent[origIndent.length - 1] : extra[0];
			const scaledExtraLength = Math.max(0, Math.round(extra.length * ratio));
			return (
				origIndent + (extraUnitChar ?? "").repeat(scaledExtraLength) + rest
			);
		})
		.join("\n");
}

interface ResolvedSpan {
	start: number;
	end: number;
	newText: string;
	editIndex: number;
}

/**
 * Tier 3: match oldText against the file line by line, comparing trimmed lines.
 * Tolerates the model getting indentation wrong; newText is re-indented to the
 * file's actual indentation on match.
 */
function lineTrimmedMatches(
	content: string,
	oldText: string,
	newText: string,
): Array<{ start: number; end: number; newText: string }> {
	const searchLines = oldText.split("\n");
	let trailingNewline = false;
	if (searchLines.length > 1 && searchLines[searchLines.length - 1] === "") {
		searchLines.pop();
		trailingNewline = true;
	}
	const searchTrimmed = searchLines.map(l => normalizeForFuzzyMatch(l.trim()));
	if (searchTrimmed.every(l => l === "")) return [];

	const lines = content.split("\n");
	const lineStarts: number[] = [];
	let offset = 0;
	for (const line of lines) {
		lineStarts.push(offset);
		offset += line.length + 1;
	}

	const matches: Array<{ start: number; end: number; newText: string }> = [];
	outer: for (let i = 0; i + searchTrimmed.length <= lines.length; i++) {
		for (let j = 0; j < searchTrimmed.length; j++) {
			if (
				normalizeForFuzzyMatch((lines[i + j] ?? "").trim()) !== searchTrimmed[j]
			) {
				continue outer;
			}
		}
		const lastLine = i + searchTrimmed.length - 1;
		const lineEnd =
			(lineStarts[lastLine] ?? 0) + (lines[lastLine] ?? "").length;
		const end =
			trailingNewline && lastLine < lines.length - 1 ? lineEnd + 1 : lineEnd;
		matches.push({
			start: lineStarts[i] ?? 0,
			end,
			newText: reindent(
				newText,
				leadingWhitespace(searchLines[0] ?? ""),
				leadingWhitespace(lines[i] ?? ""),
			),
		});
	}
	return matches;
}

interface FuzzyMatchResult {
	found: boolean;
	index: number;
	matchLength: number;
	usedFuzzyMatch: boolean;
}

/** Compatibility helper: locate oldText in content, exact first then fuzzy. */
export function fuzzyFindText(
	content: string,
	oldText: string,
): FuzzyMatchResult {
	const exactIndex = content.indexOf(oldText);
	if (exactIndex !== -1) {
		return {
			found: true,
			index: exactIndex,
			matchLength: oldText.length,
			usedFuzzyMatch: false,
		};
	}
	const fuzzyContent = normalizeForFuzzyMatch(content);
	const fuzzyOldText = normalizeForFuzzyMatch(oldText);
	const fuzzyIndex = fuzzyOldText ? fuzzyContent.indexOf(fuzzyOldText) : -1;
	if (fuzzyIndex === -1) {
		return { found: false, index: -1, matchLength: 0, usedFuzzyMatch: false };
	}
	return {
		found: true,
		index: fuzzyIndex,
		matchLength: fuzzyOldText.length,
		usedFuzzyMatch: true,
	};
}

// ── Errors ────────────────────────────────────────────────────────────────

function lineNumberAt(content: string, offset: number): number {
	let line = 1;
	for (let i = 0; i < offset; i++) {
		if (content[i] === "\n") line++;
	}
	return line;
}

/**
 * Best-effort hint pointing at where the first line of a failed oldText
 * appears, and — if a block starting there almost matches — which line
 * inside the block is the first to diverge.
 */
function closestLineHint(content: string, oldText: string): string {
	const searchLines = oldText
		.split("\n")
		.filter((_, i, arr) => !(i === arr.length - 1 && arr[i] === ""));
	const firstLine = searchLines.find(l => l.trim() !== "");
	if (!firstLine) return "";
	const needle = normalizeForFuzzyMatch(firstLine.trim());
	const lines = content.split("\n");
	const hits: number[] = [];
	for (let i = 0; i < lines.length && hits.length < 3; i++) {
		if (normalizeForFuzzyMatch((lines[i] ?? "").trim()) === needle)
			hits.push(i + 1);
	}
	if (hits.length === 0) return "";

	// For the first hit, check how far a line-by-line match extends before
	// diverging — tells the caller which line of oldText is actually wrong,
	// rather than just "later lines likely differ".
	if (searchLines.length > 1) {
		const start = (hits[0] ?? 1) - 1;
		let matched = 0;
		while (
			matched < searchLines.length &&
			start + matched < lines.length &&
			normalizeForFuzzyMatch((lines[start + matched] ?? "").trim()) ===
				normalizeForFuzzyMatch((searchLines[matched] ?? "").trim())
		) {
			matched++;
		}
		if (matched < searchLines.length) {
			return ` oldText's first ${matched} line(s) match starting at line ${hits[0]}, but oldText line ${matched + 1} ("${(searchLines[matched] ?? "").trim().slice(0, 80)}") does not match file line ${start + matched + 1} ("${(lines[start + matched] ?? "").trim().slice(0, 80)}"). Re-read that region.`;
		}
	}

	return ` The first line of oldText matches line${hits.length > 1 ? "s" : ""} ${hits.join(", ")} — later lines likely differ; re-read that region.`;
}

function editLabel(editIndex: number, totalEdits: number): string {
	return totalEdits === 1 ? "the exact text" : `edits[${editIndex}]`;
}

function getNotFoundError(
	path: string,
	editIndex: number,
	totalEdits: number,
	hint: string,
): Error {
	return new Error(
		`Could not find ${editLabel(editIndex, totalEdits)} in ${path}. The oldText must match the file content exactly, including whitespace and newlines. Read the file first to get the exact content, or provide more surrounding context to make it unique.${hint}`,
	);
}

function getDuplicateError(
	path: string,
	editIndex: number,
	totalEdits: number,
	occurrences: number,
	lineNumbers: number[],
): Error {
	const lines = lineNumbers.slice(0, 5).join(", ");
	const suffix = lineNumbers.length > 5 ? ", …" : "";
	return new Error(
		`Found ${occurrences} occurrences of ${editLabel(editIndex, totalEdits)} in ${path} (lines ${lines}${suffix}). Each oldText must uniquely identify a single location. Include 3-5 unchanged lines before and after the target text to make it unique, or set replaceAll: true to replace every occurrence.`,
	);
}

function getEmptyOldTextError(
	path: string,
	editIndex: number,
	totalEdits: number,
): Error {
	if (totalEdits === 1) {
		return new Error(
			`oldText must not be empty in ${path}. Provide text to find and replace.`,
		);
	}
	return new Error(
		`edits[${editIndex}].oldText must not be empty in ${path}. Provide text to find and replace.`,
	);
}

function getNoChangeError(path: string, totalEdits: number): Error {
	if (totalEdits === 1) {
		return new Error(
			`No changes made to ${path}. The replacement produced identical content. Verify that oldText and newText are different, and that oldText exists in the file.`,
		);
	}
	return new Error(
		`No changes made to ${path}. The replacements produced identical content.`,
	);
}

// ── Edit application ──────────────────────────────────────────────────────

function normalizeToLF(content: string): string {
	return content.replace(/\r\n/g, "\n");
}

export function applyEditsToNormalizedContent(
	normalizedContent: string,
	edits: Edit[],
	filePath: string,
): ApplyEditsResult {
	const normalizedEdits = edits.map(edit => ({
		oldText: normalizeToLF(edit.oldText),
		newText: normalizeToLF(edit.newText),
		replaceAll: edit.replaceAll === true,
	}));

	for (let i = 0; i < normalizedEdits.length; i++) {
		if (normalizedEdits[i]?.oldText.length === 0) {
			throw getEmptyOldTextError(filePath, i, normalizedEdits.length);
		}
	}

	const normCache = buildNormalizedWithMap(normalizedContent);
	const spans: ResolvedSpan[] = [];

	for (let i = 0; i < normalizedEdits.length; i++) {
		const edit = normalizedEdits[i];
		if (!edit) continue;

		// Tier 1: exact match.
		let matches: Array<{ start: number; end: number; newText: string }> =
			indexOfAll(normalizedContent, edit.oldText).map(start => ({
				start,
				end: start + edit.oldText.length,
				newText: edit.newText,
			}));

		// Tier 2: normalized match, positions mapped back to the original content.
		if (matches.length === 0) {
			const normOld = normalizeForFuzzyMatch(edit.oldText);
			if (normOld.trim() !== "") {
				matches = indexOfAll(normCache.norm, normOld).map(start => ({
					start: normCache.map[start] ?? 0,
					end: (normCache.map[start + normOld.length - 1] ?? 0) + 1,
					newText: edit.newText,
				}));
			}
		}

		// Tier 3: line-trimmed match with indentation re-application.
		if (matches.length === 0) {
			matches = lineTrimmedMatches(
				normalizedContent,
				edit.oldText,
				edit.newText,
			);
		}

		if (matches.length === 0) {
			throw getNotFoundError(
				filePath,
				i,
				normalizedEdits.length,
				closestLineHint(normalizedContent, edit.oldText),
			);
		}
		if (matches.length > 1 && !edit.replaceAll) {
			throw getDuplicateError(
				filePath,
				i,
				normalizedEdits.length,
				matches.length,
				matches.map(m => lineNumberAt(normalizedContent, m.start)),
			);
		}

		for (const match of matches) {
			spans.push({ ...match, editIndex: i });
		}
	}

	spans.sort((a, b) => a.start - b.start);
	for (let i = 1; i < spans.length; i++) {
		const previous = spans[i - 1];
		const current = spans[i];
		if (previous && current && previous.end > current.start) {
			throw new Error(
				`edits[${previous.editIndex}] and edits[${current.editIndex}] overlap in ${filePath}. Merge them into one edit or target disjoint regions.`,
			);
		}
	}

	let newContent = normalizedContent;
	for (let i = spans.length - 1; i >= 0; i--) {
		const span = spans[i];
		if (!span) continue;
		newContent =
			newContent.substring(0, span.start) +
			span.newText +
			newContent.substring(span.end);
	}

	if (normalizedContent === newContent) {
		throw getNoChangeError(filePath, normalizedEdits.length);
	}

	return { baseContent: normalizedContent, newContent };
}
