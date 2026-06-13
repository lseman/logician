// ── edit_file tool ────────────────────────────────────────────────────────────────
// Edit file contents with exact text replacement. Supports both single old_text/new_text
// and multi-edit arrays. Uses fuzzy whitespace matching for robustness, BOM handling,
// and line-ending preservation — all ported from pi's edit tool.

import * as fs from "node:fs";
import type { Tool, ToolResult } from "../types.ts";
import {
	applyEditsToNormalizedContent,
	detectLineEnding,
	Edit,
	generateDiffString,
	generateUnifiedPatch,
	normalizeToLF,
	stripBom,
	type EditOperations,
	defaultEditOperations,
} from "./helpers.ts";
import { withFileMutationQueue } from "./file-mutation-queue.ts";
import { ensureInsideCwd, resolvePath } from "./helpers.ts";
import { isStaleSinceRead, refreshAfterWrite } from "./read-tracker.ts";

// ============================================================================
// Schema & argument normalization
// ============================================================================

const editSchema = {
	type: "object",
	properties: {
		path: { type: "string", description: "File path to edit (relative or absolute)" },
		edits: {
			type: "array",
			description:
				"Exact text replacements. Each must match a unique, non-overlapping region. If two changes touch the same block, merge them into one edit.",
			items: {
				type: "object",
				properties: {
					oldText: { type: "string", description: "Exact text to find and replace" },
					newText: { type: "string", description: "Replacement text" },
				},
			},
		},
	},
	required: ["path"],
} as const;

type EditToolArgs = {
	path?: string;
	edits?: Array<{ oldText?: string; newText?: string }>;
	// Legacy single-edit fields (supported for compatibility)
	oldText?: string;
	newText?: string;
	file_path?: string;
	old_text?: string;
	new_text?: string;
	oldString?: string;
	newString?: string;
};

function prepareArguments(raw: unknown): Record<string, unknown> {
	if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
	const args = raw as Record<string, unknown>;

	// Normalize path aliases
	const path =
		(typeof args.path === "string" && args.path) ||
		(typeof args.file_path === "string" && args.file_path) ||
		"";

	// Normalize single-edit field aliases
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

	// Some models (Opus 4.6, GLM-5.1) send edits as a JSON string instead of an array
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

	// Collect single-edit fields if present
	if (oldText || newText) {
		edits.push({ oldText, newText });
	}

	// Collect edits[] entries
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

	return {
		path,
		edits,
	};
}

// ============================================================================
// Tool definition
// ============================================================================

export const edit_file: Tool = {
	name: "edit_file",
	hookAliases: ["Edit"],
	description:
		"Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. Supports BOM handling and line-ending preservation.",
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

		// Check file existence and permissions
		try {
			await defaultEditOperations.access(resolved);
		} catch (error: unknown) {
			const errorMessage =
				error instanceof Error && "code" in error
					? `Error code: ${(error as { code: string }).code}`
					: String(error);
			return `Could not edit file: ${path}. ${errorMessage}.`;
		}

		// Check read-tracker staleness
		if (isStaleSinceRead(resolved)) {
			return `${resolved} has been modified since it was last read. Read it again before editing.`;
		}

		return withFileMutationQueue(resolved, async () => {
			// Read file
			const buffer = await defaultEditOperations.readFile(resolved);
			const rawContent = buffer.toString("utf-8");

			// Strip BOM before matching (LLM won't include invisible BOM in oldText)
			const { text: content } = stripBom(rawContent);
			const lineEnding = detectLineEnding(content);
			const normalizedContent = normalizeToLF(content);

			// Apply edits with fuzzy whitespace matching
			const { baseContent, newContent } = applyEditsToNormalizedContent(
				normalizedContent,
				edits,
				path,
			);

			// Restore original line endings
			const finalContent =
				(content.startsWith("\uFEFF") ? "\uFEFF" : "") +
				(lineEnding === "\r\n" ? newContent.replace(/\n/g, "\r\n") : newContent);

			// Write back
			await defaultEditOperations.writeFile(resolved, finalContent);
			refreshAfterWrite(resolved);

			// Generate diff and patch
			const { diff, firstChangedLine } = generateDiffString(baseContent, finalContent);
			const patch = generateUnifiedPatch(path, baseContent, finalContent);

			// Return structured result
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
