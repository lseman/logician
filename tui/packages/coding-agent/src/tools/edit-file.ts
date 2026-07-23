// ── edit_file tool ────────────────────────────────────────────────────────────────
// Edit file contents with exact text replacement. Supports both single old_text/new_text
// and multi-edit arrays, plus replaceAll for renames. Matching runs a three-tier ladder:
// exact → whitespace/punctuation-normalized (position-mapped back to the original, so
// untouched regions are never rewritten) → line-trimmed with indentation re-application.
// BOM handling and line-ending preservation ported from pi's edit tool.

import type { Tool, ToolResult } from "@logician/agent-core/core/types.ts";
import { ensureInsideCwd, resolveReadPath } from "@logician/agent-core/tools/shared/path-utils.ts";
import {
	detectLineEnding,
	normalizeToLF,
	restoreLineEndings,
	stripBom,
	defaultEditOperations,
} from "./helpers.ts";
import { withFileMutationQueue } from "./shared/file-mutation-queue.ts";
import { atomicWriteFile } from "./shared/atomic-write.ts";
import { hasBeenRead, isStaleSinceRead, refreshAfterWrite } from "./read-tracker.ts";
import { generateDiffString, generateUnifiedPatch } from "./diff-utils.ts";

// ============================================================================
// Fuzzy matching
// ============================================================================

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
	if (/[\u2018\u2019\u201A\u201B]/.test(ch)) return "'";
	if (/[\u201C\u201D\u201E\u201F]/.test(ch)) return "\"";
	if (/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]/.test(ch)) return "-";
	if (/[\u00A0\u2002-\u200A\u202F\u205F\u3000]/.test(ch)) return " ";
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
		.map((line) => {
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
		const line = lines[li];
		const trimmedLength = line.trimEnd().length;
		for (let i = 0; i < trimmedLength; i++) {
			norm.push(normalizeChar(line[i]));
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

/** Shift newText from the indentation the model wrote to the file's actual indentation. */
function reindent(newText: string, searchIndent: string, origIndent: string): string {
	if (searchIndent === origIndent) return newText;
	return newText
		.split("\n")
		.map((line) => {
			if (line.trim() === "") return line;
			if (line.startsWith(searchIndent)) {
				return origIndent + line.slice(searchIndent.length);
			}
			return line;
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
	const searchTrimmed = searchLines.map((l) => normalizeForFuzzyMatch(l.trim()));
	if (searchTrimmed.every((l) => l === "")) return [];

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
			if (normalizeForFuzzyMatch(lines[i + j].trim()) !== searchTrimmed[j]) {
				continue outer;
			}
		}
		const lastLine = i + searchTrimmed.length - 1;
		const lineEnd = lineStarts[lastLine] + lines[lastLine].length;
		const end =
			trailingNewline && lastLine < lines.length - 1 ? lineEnd + 1 : lineEnd;
		matches.push({
			start: lineStarts[i],
			end,
			newText: reindent(
				newText,
				leadingWhitespace(searchLines[0]),
				leadingWhitespace(lines[i]),
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
export function fuzzyFindText(content: string, oldText: string): FuzzyMatchResult {
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

// ============================================================================
// Errors
// ============================================================================

function lineNumberAt(content: string, offset: number): number {
	let line = 1;
	for (let i = 0; i < offset; i++) {
		if (content[i] === "\n") line++;
	}
	return line;
}

/** Best-effort hint pointing at where the first line of a failed oldText appears. */
function closestLineHint(content: string, oldText: string): string {
	const firstLine = oldText.split("\n").find((l) => l.trim() !== "");
	if (!firstLine) return "";
	const needle = normalizeForFuzzyMatch(firstLine.trim());
	const hits: number[] = [];
	const lines = content.split("\n");
	for (let i = 0; i < lines.length && hits.length < 3; i++) {
		if (normalizeForFuzzyMatch(lines[i].trim()) === needle) hits.push(i + 1);
	}
	if (hits.length === 0) return "";
	return (
		` The first line of oldText matches line${hits.length > 1 ? "s" : ""} ` +
		`${hits.join(", ")} — later lines likely differ; re-read that region.`
	);
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
		`Could not find ${editLabel(editIndex, totalEdits)} in ${path}. ` +
		"The oldText must match the file content exactly, including whitespace and newlines. " +
		"Read the file first to get the exact content, or provide more surrounding context to make it unique." +
		hint,
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
		`Found ${occurrences} occurrences of ${editLabel(editIndex, totalEdits)} in ${path} ` +
		`(lines ${lines}${suffix}). ` +
		"Each oldText must uniquely identify a single location. " +
		"Include 3-5 unchanged lines before and after the target text to make it unique, " +
		"or set replaceAll: true to replace every occurrence.",
	);
}

function getEmptyOldTextError(path: string, editIndex: number, totalEdits: number): Error {
	if (totalEdits === 1) {
		return new Error(`oldText must not be empty in ${path}. Provide text to find and replace.`);
	}
	return new Error(`edits[${editIndex}].oldText must not be empty in ${path}. Provide text to find and replace.`);
}

function getNoChangeError(path: string, totalEdits: number): Error {
	if (totalEdits === 1) {
		return new Error(
			`No changes made to ${path}. ` +
			"The replacement produced identical content. " +
			"Verify that oldText and newText are different, and that oldText exists in the file.",
		);
	}
	return new Error(`No changes made to ${path}. The replacements produced identical content.`);
}

// ============================================================================
// Edit application
// ============================================================================

export function applyEditsToNormalizedContent(
	normalizedContent: string,
	edits: Edit[],
	filePath: string,
): ApplyEditsResult {
	const normalizedEdits = edits.map((edit) => ({
		oldText: normalizeToLF(edit.oldText),
		newText: normalizeToLF(edit.newText),
		replaceAll: edit.replaceAll === true,
	}));

	for (let i = 0; i < normalizedEdits.length; i++) {
		if (normalizedEdits[i].oldText.length === 0) {
			throw getEmptyOldTextError(filePath, i, normalizedEdits.length);
		}
	}

	const normCache = buildNormalizedWithMap(normalizedContent);
	const spans: ResolvedSpan[] = [];

	for (let i = 0; i < normalizedEdits.length; i++) {
		const edit = normalizedEdits[i];

		// Tier 1: exact match.
		let matches: Array<{ start: number; end: number; newText: string }> =
			indexOfAll(normalizedContent, edit.oldText).map((start) => ({
				start,
				end: start + edit.oldText.length,
				newText: edit.newText,
			}));

		// Tier 2: normalized match, positions mapped back to the original content.
		if (matches.length === 0) {
			const normOld = normalizeForFuzzyMatch(edit.oldText);
			if (normOld.trim() !== "") {
				matches = indexOfAll(normCache.norm, normOld).map((start) => ({
					start: normCache.map[start],
					end: normCache.map[start + normOld.length - 1] + 1,
					newText: edit.newText,
				}));
			}
		}

		// Tier 3: line-trimmed match with indentation re-application.
		if (matches.length === 0) {
			matches = lineTrimmedMatches(normalizedContent, edit.oldText, edit.newText);
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
				matches.map((m) => lineNumberAt(normalizedContent, m.start)),
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
		if (previous.end > current.start) {
			throw new Error(
				`edits[${previous.editIndex}] and edits[${current.editIndex}] ` +
				`overlap in ${filePath}. ` +
				"Merge them into one edit or target disjoint regions.",
			);
		}
	}

	let newContent = normalizedContent;
	for (let i = spans.length - 1; i >= 0; i--) {
		const span = spans[i];
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

// ============================================================================
// Schema & argument normalization
// ============================================================================

const editSchema = {
	type: "object",
	properties: {
		path: {
			type: "string",
			description: "File path to edit (relative or absolute)",
		},
		edits: {
			type: "array",
			description:
				"Exact text replacements. Each must match a unique, " +
				"non-overlapping region of the original file. " +
				"If two changes touch the same block, merge them into one edit.",
			items: {
				type: "object",
				properties: {
					oldText: {
						type: "string",
						description: "Exact text to find and replace",
					},
					newText: { type: "string", description: "Replacement text" },
					replaceAll: {
						type: "boolean",
						description:
							"Replace every occurrence of oldText instead of requiring uniqueness. " +
							"Use for renaming a symbol throughout the file.",
					},
				},
			},
		},
	},
	required: ["path"],
} as const;

function prepareArguments(raw: unknown): Record<string, unknown> {
	if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
	const args = raw as Record<string, unknown>;

	const path =
		(typeof args.path === "string" && args.path) ||
		(typeof args.file_path === "string" && args.file_path) ||
		"";

	const oldText =
		(typeof args.oldText === "string" && args.oldText) ||
		(typeof args.old_text === "string" && args.old_text) ||
		(typeof args.oldString === "string" && args.oldString) ||
		"";
	const newText =
		(typeof args.newText === "string" && args.newText) ||
		(typeof args.new_text === "string" && args.new_text) ||
		(typeof args.newString === "string" && args.newString) ||
		"";
	const topReplaceAll = args.replaceAll === true || args.replace_all === true;

	let rawEdits = args.edits;
	if (typeof rawEdits === "string") {
		try {
			const parsed = JSON.parse(rawEdits);
			if (Array.isArray(parsed)) rawEdits = parsed;
		} catch (e: unknown) {
			// Leave as-is
		}
	}

	const edits: Edit[] = [];
	if (oldText) {
		edits.push({ oldText, newText, replaceAll: topReplaceAll });
	}
	if (Array.isArray(rawEdits)) {
		for (const item of rawEdits) {
			if (!item || typeof item !== "object") continue;
			const e = item as Record<string, unknown>;
			const eOld =
				(typeof e.oldText === "string" && e.oldText) ||
				(typeof e.old_text === "string" && e.old_text) ||
				"";
			const eNew =
				(typeof e.newText === "string" && e.newText) ||
				(typeof e.new_text === "string" && e.new_text) ||
				"";
			if (eOld) {
				edits.push({
					oldText: eOld,
					newText: eNew,
					replaceAll: e.replaceAll === true || e.replace_all === true,
				});
			}
		}
	}

	return { path, edits };
}

// ============================================================================
// Tool definition
// ============================================================================

export const edit_file: Tool = {
	name: "edit_file",
	executionMode: "parallel",
	label: "Edit File",
	hookAliases: ["Edit"],
	description:
		"Edit a single file using exact text replacement. " +
			"Every edits[].oldText must match a unique, " +
			"non-overlapping region of the original file " +
			"(or set replaceAll: true on an edit to replace every occurrence). " +
			"The file must have been read with read_file first. " +
			"Supports BOM handling and line-ending preservation.",
	promptSnippet: "Edit files using exact text replacement with precise matching",
	promptGuidelines: [
		"Use edit_file for surgical edits; keep oldText unique in the file, or set replaceAll for renames",
	],
	parameters: editSchema,
	prepareArguments,
	execute: async (
		args: Record<string, unknown>,
		ctx,
	): Promise<string | ToolResult> => {
		const path = String(args.path ?? "");
		const edits = (args.edits as Edit[]) || [];

		if (!path) {
			return "Error: edit_file requires a path.";
		}
		if (edits.length === 0) {
			return "Error: Provide oldText/newText or edits[].";
		}

		const resolved = resolveReadPath(path, ctx.cwd || process.cwd());
		ensureInsideCwd(ctx.cwd, resolved, undefined, ctx.allowAllPaths);

		try {
			await defaultEditOperations.access(resolved);
		} catch (error: unknown) {
			const errorMessage =
				error instanceof Error && "code" in error
					? `Error code: ${(error as { code: string }).code}`
					: String(error);
			return `Could not edit file: ${path}. ${errorMessage}.`;
		}

		if (!hasBeenRead(resolved)) {
			return `${resolved} has not been read yet. ` +
				"Read it with read_file before editing.";
		}
		return withFileMutationQueue(resolved, async () => {
			if (isStaleSinceRead(resolved)) {
				return `${resolved} has been modified since it was last read. ` +
					"Read it again before editing.";
			}
			const buffer = await defaultEditOperations.readFile(resolved);
			const rawContent = buffer.toString("utf-8");
			const { bom, text: content } = stripBom(rawContent);
			const lineEnding = detectLineEnding(content);
			const normalizedContent = normalizeToLF(content);

			const { baseContent, newContent } = applyEditsToNormalizedContent(
				normalizedContent,
				edits,
				path,
			);

			const finalContent = bom + restoreLineEndings(newContent, lineEnding);

			await atomicWriteFile(resolved, finalContent, {
				expectedContent: rawContent,
			});
			refreshAfterWrite(resolved);

			const { diff, firstChangedLine } = generateDiffString(
				baseContent,
				newContent,
			);
			const patch = generateUnifiedPatch(path, baseContent, newContent);

			return {
				content:
					`Successfully replaced ${edits.length} block(s) in ${path}.\n` +
					`\nDiff:\n${diff}`,
				details: {
					diff,
					patch,
					firstChangedLine,
				},
			};
		});
	},
};
