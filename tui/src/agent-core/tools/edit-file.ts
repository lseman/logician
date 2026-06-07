// ── edit_file tool ────────────────────────────────────────────────────────────────
// Edit file contents with exact text replacement.

import * as fs from "node:fs";
import type { Tool } from "../types.ts";
import { withFileMutationQueue } from "./file-mutation-queue.ts";
import { ensureInsideCwd, mutationSummary, resolvePath } from "./helpers.ts";
import { isStaleSinceRead, refreshAfterWrite } from "./read-tracker.ts";

export const edit_file: Tool = {
	name: "edit_file",
	description:
		"Replace exact text in a file and return a unified diff. Supports either old_text/new_text or edits[].",
	parameters: {
		type: "object",
		properties: {
			path: { type: "string", description: "File path to edit" },
			old_text: {
				type: "string",
				description: "Exact text to find and replace",
			},
			new_text: { type: "string", description: "Replacement text" },
			edits: {
				type: "array",
				description:
					"Multiple exact replacements, each with old_text/new_text or oldText/newText",
				items: {
					type: "object",
					properties: {
						old_text: { type: "string" },
						new_text: { type: "string" },
						oldText: { type: "string" },
						newText: { type: "string" },
					},
				},
			},
		},
		required: ["path"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
		const args = raw as Record<string, unknown>;
		return {
			...args,
			path: args.path ?? args.file_path ?? args.filename,
			old_text: args.old_text ?? args.oldString ?? args.old_string,
			new_text: args.new_text ?? args.newString ?? args.new_string,
		};
	},
	execute: async (args, ctx): Promise<string> => {
		const filePath = String(args.path);
		const resolved = resolvePath(ctx.cwd, filePath);
		ensureInsideCwd(ctx.cwd, resolved);

		if (!fs.existsSync(resolved)) {
			return `Error: File not found: ${resolved}`;
		}

		const edits = normalizeEdits(args);
		if (edits.length === 0) {
			return "Error: Provide old_text/new_text or edits[].";
		}

		return withFileMutationQueue(resolved, async () => {
			if (isStaleSinceRead(resolved)) {
				return `Error: ${resolved} has been modified since it was last read. Read it again before editing.`;
			}
			const before = fs.readFileSync(resolved, "utf-8");
			let content = before;
			for (const [idx, edit] of edits.entries()) {
				if (!edit.oldText) {
					return `Error: Edit ${idx + 1} has empty old_text.`;
				}
				const count = countOccurrences(content, edit.oldText);
				if (count === 0) {
					return `Error: Text not found for edit ${idx + 1}. Ensure exact match including whitespace.`;
				}
				if (count > 1) {
					return `Error: Text for edit ${idx + 1} is not unique (${count} matches). Include more context.`;
				}
				content = content.replace(edit.oldText, edit.newText);
			}

			if (content === before) {
				return `No changes made: ${resolved}`;
			}

			fs.writeFileSync(resolved, content, "utf-8");
			refreshAfterWrite(resolved);
			const diff = await mutationSummary(ctx.cwd, resolved, before, content);

			return `Edited ${resolved}\n\nDiff:\n${diff}`;
		});
	},
};

function normalizeEdits(
	args: Record<string, unknown>,
): Array<{ oldText: string; newText: string }> {
	const out: Array<{ oldText: string; newText: string }> = [];
	if (typeof args.old_text === "string" || typeof args.oldText === "string") {
		out.push({
			oldText: String(args.old_text ?? args.oldText ?? ""),
			newText: String(args.new_text ?? args.newText ?? ""),
		});
	}
	// Some models (Opus, GLM) send edits as a JSON string instead of an array.
	let rawEdits = args.edits;
	if (typeof rawEdits === "string") {
		try {
			const parsed = JSON.parse(rawEdits);
			if (Array.isArray(parsed)) rawEdits = parsed;
		} catch {
			// Leave as-is; falls through to the non-array branch below.
		}
	}
	if (Array.isArray(rawEdits)) {
		for (const item of rawEdits) {
			if (!item || typeof item !== "object") continue;
			const e = item as Record<string, unknown>;
			out.push({
				oldText: String(e.old_text ?? e.oldText ?? ""),
				newText: String(e.new_text ?? e.newText ?? ""),
			});
		}
	}
	return out;
}

function countOccurrences(text: string, needle: string): number {
	if (!needle) return 0;
	let count = 0;
	let idx = 0;
	while ((idx = text.indexOf(needle, idx)) !== -1) {
		count++;
		idx += needle.length;
	}
	return count;
}
