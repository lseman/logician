// ── edit tool ─────────────────────────────────────────────────────────────
// Edit file contents with exact text replacement, via the three-tier fuzzy
// matching ladder in edit-matching.ts. Supports both single oldText/newText
// and multi-edit arrays, plus replaceAll for renames. BOM handling and
// line-ending preservation. Ported from coding-agent's tools/edit-file.ts,
// rewritten against ExecutionEnv.

import { type Static, Type } from "@sinclair/typebox";
import type { AgentTool } from "../../agent/types.ts";
import type { ExecutionEnv } from "../../env/execution-env.ts";
import { atomicWriteFile } from "./atomic-write.ts";
import { generateDiffString, generateUnifiedPatch } from "./diff-utils.ts";
import { applyEditsToNormalizedContent, type Edit } from "./edit-matching.ts";
import { withFileMutationQueue } from "./file-mutation-queue.ts";
import { ensureInsideCwd, resolveReadPath } from "./path-utils.ts";
import {
	hasBeenRead,
	isStaleSinceRead,
	refreshAfterWrite,
} from "./read-tracker.ts";
import {
	detectLineEnding,
	normalizeToLF,
	restoreLineEndings,
	stripBom,
} from "./text-helpers.ts";

const editSchema = Type.Object({
	path: Type.String({
		description: "File path to edit (relative or absolute)",
	}),
	edits: Type.Array(
		Type.Object({
			oldText: Type.String({ description: "Exact text to find and replace" }),
			newText: Type.String({ description: "Replacement text" }),
			replaceAll: Type.Optional(
				Type.Boolean({
					description:
						"Replace every occurrence of oldText instead of requiring uniqueness. Use for renaming a symbol throughout the file.",
				}),
			),
		}),
		{
			description:
				"Exact text replacements. Each must match a unique, non-overlapping region of the original file. If two changes touch the same block, merge them into one edit.",
		},
	),
});

export interface EditToolDetails {
	diff: string;
	patch: string;
	firstChangedLine: number | undefined;
}

export interface EditToolOptions {
	env: ExecutionEnv;
	allowedPaths?: string[];
	allowAllPaths?: boolean;
}

function prepareEditArguments(raw: unknown): unknown {
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
		} catch {
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

export function createEditTool(
	options: EditToolOptions,
): AgentTool<typeof editSchema, EditToolDetails | undefined> {
	const { env, allowedPaths, allowAllPaths } = options;
	return {
		name: "edit",
		label: "Edit File",
		executionMode: "parallel",
		description:
			"Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file (or set replaceAll: true on an edit to replace every occurrence). The file must have been read with read first. Supports BOM handling and line-ending preservation.",
		parameters: editSchema,
		prepareArguments: prepareEditArguments,
		async execute(_toolCallId, rawParams) {
			const { path, edits } = rawParams as Static<typeof editSchema>;
			if (!path) throw new Error("edit requires a path.");
			if (edits.length === 0)
				throw new Error("Provide oldText/newText or edits[].");

			const resolved = await resolveReadPath(env, path);
			await ensureInsideCwd(env, resolved, allowedPaths, allowAllPaths);

			const exists = await env.exists(resolved);
			if (!exists.ok)
				throw new Error(
					`Could not edit file: ${path}. ${exists.error.message}`,
				);
			if (!exists.value)
				throw new Error(`Could not edit file: ${path}. File not found.`);

			if (!(await hasBeenRead(env, resolved))) {
				throw new Error(
					`${resolved} has not been read yet. Read it with read before editing.`,
				);
			}

			return withFileMutationQueue(env, resolved, async () => {
				if (await isStaleSinceRead(env, resolved)) {
					throw new Error(
						`${resolved} has been modified since it was last read. Read it again before editing.`,
					);
				}
				const rawResult = await env.readTextFile(resolved);
				if (!rawResult.ok)
					throw new Error(
						`Could not read file: ${path}. ${rawResult.error.message}`,
					);
				const rawContent = rawResult.value;
				const { bom, text: content } = stripBom(rawContent);
				const lineEnding = detectLineEnding(content);
				const normalizedContent = normalizeToLF(content);

				const { baseContent, newContent } = applyEditsToNormalizedContent(
					normalizedContent,
					edits as Edit[],
					path,
				);

				const finalContent = bom + restoreLineEndings(newContent, lineEnding);

				await atomicWriteFile(env, resolved, finalContent, {
					expectedContent: rawContent,
				});
				await refreshAfterWrite(env, resolved);

				const { diff, firstChangedLine } = generateDiffString(
					baseContent,
					newContent,
				);
				const patch = generateUnifiedPatch(path, baseContent, newContent);

				return {
					content: [
						{
							type: "text" as const,
							text: `Successfully replaced ${edits.length} block(s) in ${path}.\n\nDiff:\n${diff}`,
						},
					],
					details: { diff, patch, firstChangedLine },
				};
			});
		},
	};
}
