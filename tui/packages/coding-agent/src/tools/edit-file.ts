// ── edit_file tool ────────────────────────────────────────────────────────────────
// Edit file contents with exact text replacement. Supports both single old_text/new_text
// and multi-edit arrays. Uses fuzzy whitespace matching for robustness, BOM handling,
// and line-ending preservation — all ported from pi's edit tool.
//
// Fuzzy edit logic (buildPosMapping, searchForText, applyEditsToNormalizedContent,
// normalizeForFuzzyMatch) is included inline to avoid a circular dependency with
// helpers.ts (which used to re-export these).

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
import { isStaleSinceRead, refreshAfterWrite } from "./read-tracker.ts";
import { generateDiffString, generateUnifiedPatch } from "./diff-utils.ts";

// ============================================================================
// Fuzzy edit logic (merged from fuzzy-edit.ts)
// ============================================================================

export interface Edit {
	oldText: string;
	newText: string;
}

export interface ApplyEditsResult {
	baseContent: string;
	newContent: string;
}

/**
 * Normalize text for fuzzy matching. This mirrors Pi's edit tool:
 * trailing whitespace is ignored, smart quotes/dashes are normalized, and
 * uncommon Unicode spaces become regular spaces.
 */
export function normalizeForFuzzyMatch(text: string): string {
	return text
		.normalize("NFKC")
		.split("\n")
		.map((line) => line.trimEnd())
		.join("\n")
		.replace(/[\u2018\u2019\u201A\u201B]/g, "'")
		.replace(/[\u201C\u201D\u201E\u201F]/g, "\"")
		.replace(/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]/g, "-")
		.replace(/[\u00A0\u2002-\u200A\u202F\u205F\u3000]/g, " ");
}

interface FuzzyMatchResult {
	found: boolean;
	index: number;
	matchLength: number;
	usedFuzzyMatch: boolean;
}

interface MatchedEdit {
	editIndex: number;
	matchIndex: number;
	matchLength: number;
	newText: string;
}

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
	const fuzzyIndex = fuzzyContent.indexOf(fuzzyOldText);
	if (fuzzyIndex === -1) {
		return {
			found: false,
			index: -1,
			matchLength: 0,
			usedFuzzyMatch: false,
		};
	}

	return {
		found: true,
		index: fuzzyIndex,
		matchLength: fuzzyOldText.length,
		usedFuzzyMatch: true,
	};
}

function countOccurrences(content: string, oldText: string): number {
	const fuzzyContent = normalizeForFuzzyMatch(content);
	const fuzzyOldText = normalizeForFuzzyMatch(oldText);
	return fuzzyContent.split(fuzzyOldText).length - 1;
}

function getNotFoundError(path: string, editIndex: number, totalEdits: number): Error {
	if (totalEdits === 1) {
		return new Error(
			`Could not find the exact text in ${path}. ` +
			"The old text must match exactly including all whitespace and newlines.",
		);
	}
	return new Error(
		`Could not find edits[${editIndex}] in ${path}. ` +
		"The oldText must match exactly including all whitespace and newlines.",
	);
}

function getDuplicateError(
	path: string,
	editIndex: number,
	totalEdits: number,
	occurrences: number,
): Error {
	if (totalEdits === 1) {
		return new Error(
			`Found ${occurrences} occurrences of the text in ${path}. ` +
			"The text must be unique. " +
			"Please provide more context to make it unique.",
		);
	}
	return new Error(
		`Found ${occurrences} occurrences of edits[${editIndex}] in ${path}. ` +
		"Each oldText must be unique. " +
		"Please provide more context to make it unique.",
	);
}

function getEmptyOldTextError(path: string, editIndex: number, totalEdits: number): Error {
	if (totalEdits === 1) {
		return new Error(`oldText must not be empty in ${path}.`);
	}
	return new Error(`edits[${editIndex}].oldText must not be empty in ${path}.`);
}

function getNoChangeError(path: string, totalEdits: number): Error {
	if (totalEdits === 1) {
		return new Error(
			`No changes made to ${path}. ` +
			"The replacement produced identical content. " +
			"This might indicate an issue with special characters " +
			"or the text not existing as expected.",
		);
	}
	return new Error(`No changes made to ${path}. The replacements produced identical content.`);
}

export function applyEditsToNormalizedContent(
	normalizedContent: string,
	edits: Edit[],
	filePath: string,
): ApplyEditsResult {
	const normalizedEdits = edits.map((edit) => ({
		oldText: normalizeToLF(edit.oldText),
		newText: normalizeToLF(edit.newText),
	}));

	for (let i = 0; i < normalizedEdits.length; i++) {
		if (normalizedEdits[i].oldText.length === 0) {
			throw getEmptyOldTextError(filePath, i, normalizedEdits.length);
		}
	}

	const initialMatches = normalizedEdits.map((edit) =>
		fuzzyFindText(normalizedContent, edit.oldText),
	);
	const baseContent = initialMatches.some((match) => match.usedFuzzyMatch)
		? normalizeForFuzzyMatch(normalizedContent)
		: normalizedContent;

	const matchedEdits: MatchedEdit[] = [];
	for (let i = 0; i < normalizedEdits.length; i++) {
		const edit = normalizedEdits[i];
		const matchResult = fuzzyFindText(baseContent, edit.oldText);
		if (!matchResult.found) {
			throw getNotFoundError(filePath, i, normalizedEdits.length);
		}

		const occurrences = countOccurrences(baseContent, edit.oldText);
		if (occurrences > 1) {
			throw getDuplicateError(filePath, i, normalizedEdits.length, occurrences);
		}

		matchedEdits.push({
			editIndex: i,
			matchIndex: matchResult.index,
			matchLength: matchResult.matchLength,
			newText: edit.newText,
		});
	}

	matchedEdits.sort((a, b) => a.matchIndex - b.matchIndex);
	for (let i = 1; i < matchedEdits.length; i++) {
		const previous = matchedEdits[i - 1];
		const current = matchedEdits[i];
		if (previous.matchIndex + previous.matchLength > current.matchIndex) {
			throw new Error(
				`edits[${previous.editIndex}] and edits[${current.editIndex}] ` +
				`overlap in ${filePath}. ` +
				"Merge them into one edit or target disjoint regions.",
			);
		}
	}

	let newContent = baseContent;
	for (let i = matchedEdits.length - 1; i >= 0; i--) {
		const edit = matchedEdits[i];
		newContent =
			newContent.substring(0, edit.matchIndex) +
			edit.newText +
			newContent.substring(edit.matchIndex + edit.matchLength);
	}

	if (baseContent === newContent) {
		throw getNoChangeError(filePath, normalizedEdits.length);
	}

	return { baseContent, newContent };
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

	let rawEdits = args.edits;
	if (typeof rawEdits === "string") {
		try {
			const parsed = JSON.parse(rawEdits);
			if (Array.isArray(parsed)) rawEdits = parsed;
		} catch {
			// Leave as-is
		}
	}

	const edits: Array<{ oldText: string; newText: string }> = [];
	if (oldText || newText) {
		edits.push({ oldText, newText });
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
				edits.push({ oldText: eOld, newText: eNew });
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
	label: "Edit File",
	hookAliases: ["Edit"],
	description:
		"Edit a single file using exact text replacement. " +
			"Every edits[].oldText must match a unique, " +
			"non-overlapping region of the original file. " +
			"Supports BOM handling and line-ending preservation.",
	promptSnippet: "Edit files using exact text replacement with precise matching",
	promptGuidelines: ["Use edit_file for surgical edits; keep oldText unique in the file"],
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
		ensureInsideCwd(ctx.cwd, resolved);

		try {
			await defaultEditOperations.access(resolved);
		} catch (error: unknown) {
			const errorMessage =
				error instanceof Error && "code" in error
					? `Error code: ${(error as { code: string }).code}`
					: String(error);
			return `Could not edit file: ${path}. ${errorMessage}.`;
		}

		if (isStaleSinceRead(resolved)) {
			return `${resolved} has been modified since it was last read. ` +
				"Read it again before editing.";
		}

		return withFileMutationQueue(resolved, async () => {
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

			await defaultEditOperations.writeFile(resolved, finalContent);
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
