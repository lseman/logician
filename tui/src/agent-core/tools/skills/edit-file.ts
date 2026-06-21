// ── edit_file tool ────────────────────────────────────────────────────────────────
// Edit file contents with exact text replacement. Supports both single old_text/new_text
// and multi-edit arrays. Uses fuzzy whitespace matching for robustness, BOM handling,
// and line-ending preservation — all ported from pi's edit tool.
//
// Fuzzy edit logic (buildPosMapping, searchForText, applyEditsToNormalizedContent,
// normalizeForFuzzyMatch) is included inline to avoid a circular dependency with
// helpers.ts (which used to re-export these).

import type { Tool, ToolResult } from "../../core/types.ts";
import { ensureInsideCwd, resolvePath } from "../shared/path-utils.ts";
import { generateDiffString, generateUnifiedPatch } from "./diff-utils.ts";
import {
	detectLineEnding,
	normalizeToLF,
	stripBom,
	defaultEditOperations,
} from "../shared/helpers.ts";
import { withFileMutationQueue } from "../shared/file-mutation-queue.ts";
import { isStaleSinceRead, refreshAfterWrite } from "./read-tracker.ts";

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

/** Normalize whitespace for fuzzy matching. */
export function normalizeForFuzzyMatch(text: string): string {
	return text.replace(/[\s]+/g, " ").trim();
}

/** Map positions from fuzzy-normalized content back to actual content positions. */
function buildPosMapping(actual: string, fuzzy: string): number[] {
	const mapping: number[] = [];
	let actualPos = 0;
	let fuzzyPos = 0;

	while (fuzzyPos < fuzzy.length) {
		const fuzzyChar = fuzzy[fuzzyPos];
		if (actualPos >= actual.length) {
			mapping[fuzzyPos] = actualPos;
			fuzzyPos++;
			continue;
		}
		const actualChar = actual[actualPos];
		if (fuzzyChar === " ") {
			if (actualChar === " " || actualChar === "\t" || actualChar === "\n") {
				mapping[fuzzyPos] = actualPos;
				actualPos++;
			} else {
				mapping[fuzzyPos] = actualPos;
				fuzzyPos++;
			}
		} else {
			while (
				actualPos < actual.length &&
				(actual[actualPos] === " " || actual[actualPos] === "\t" || actual[actualPos] === "\n")
			) {
				actualPos++;
			}
			mapping[fuzzyPos] = actualPos;
			actualPos++;
			fuzzyPos++;
		}
	}
	return mapping;
}

export function applyEditsToNormalizedContent(
	normalizedContent: string,
	edits: Edit[],
	_filePath: string,
): ApplyEditsResult {
	const fuzzyNormalized = normalizeForFuzzyMatch(normalizedContent);
	const sortedEdits = edits
		.map((edit, i) => ({ ...edit, originalIndex: i }))
		.sort((a, b) => a.oldText.length - b.oldText.length);

	const editPositions: Array<{
		start: number;
		end: number;
		oldText: string;
		newText: string;
	}> = [];

	for (const edit of sortedEdits) {
		if (!edit.oldText) continue;
		const fuzzyOldText = normalizeForFuzzyMatch(edit.oldText);
		const fuzzyMatchPos = fuzzyNormalized.indexOf(fuzzyOldText);

		if (fuzzyMatchPos !== -1) {
			const fuzzyMapping = buildPosMapping(normalizedContent, fuzzyNormalized);
			const actualStart = fuzzyMapping[fuzzyMatchPos] ?? fuzzyMatchPos;
			const actualEnd =
				fuzzyMapping[fuzzyMatchPos + fuzzyOldText.length - 1] ??
				fuzzyMatchPos + fuzzyOldText.length;
			editPositions.push({
				start: actualStart,
				end: actualEnd,
				oldText: edit.oldText,
				newText: edit.newText,
			});
		} else {
			const exactPos = normalizedContent.indexOf(edit.oldText);
			if (exactPos !== -1) {
				editPositions.push({
					start: exactPos,
					end: exactPos + edit.oldText.length,
					oldText: edit.oldText,
					newText: edit.newText,
				});
			}
		}
	}

	let newContent = normalizedContent;
	for (let i = editPositions.length - 1; i >= 0; i--) {
		const { start, end, newText } = editPositions[i];
		newContent = newContent.slice(0, start) + newText + newContent.slice(end);
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
				"Exact text replacements. Each must match a unique, non-overlapping region of the original file. If two changes touch the same block, merge them into one edit.",
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
		"Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. Supports BOM handling and line-ending preservation.",
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

		const resolved = resolvePath(ctx.cwd, path);
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
			return `${resolved} has been modified since it was last read. Read it again before editing.`;
		}

		return withFileMutationQueue(resolved, async () => {
			const buffer = await defaultEditOperations.readFile(resolved);
			const rawContent = buffer.toString("utf-8");
			const { text: content } = stripBom(rawContent);
			const lineEnding = detectLineEnding(content);
			const normalizedContent = normalizeToLF(content);

			const { baseContent, newContent } = applyEditsToNormalizedContent(
				normalizedContent,
				edits,
				path,
			);

			const finalContent =
				(content.startsWith("\uFEFF") ? "\uFEFF" : "") +
				(lineEnding === "\r\n"
					? newContent.replace(/\n/g, "\r\n")
					: newContent);

			await defaultEditOperations.writeFile(resolved, finalContent);
			refreshAfterWrite(resolved);

			const { diff, firstChangedLine } = generateDiffString(
				baseContent,
				finalContent,
			);
			const patch = generateUnifiedPatch(path, baseContent, finalContent);

			return {
				content: `Successfully replaced ${edits.length} block(s) in ${path}.`,
				details: {
					diff,
					patch,
					firstChangedLine,
				},
			};
		});
	},
};
