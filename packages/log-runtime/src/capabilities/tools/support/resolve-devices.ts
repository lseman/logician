// ── xd://resolve and xd://reject devices ──────────────────────────────────────
// These devices handle the accept/reject flow for staged (previewed) edits
// left by tools like ast_edit and ast_grep. Dispatched via:
//   write path="xd://resolve" content='{"reason": "..."}'  → apply to disk
//   write path="xd://reject"  content='{"reason": "..."}'  → discard
//
// `files` normally comes from the staged-edits singleton (set by the
// previewing tool); a caller may pass its own `files` to bypass that.

import * as fs from "node:fs";
import * as path from "node:path";
import type { Tool } from "@logician/log-core";
import type { EditStore } from "./edit-store.js";
import { clearStagedEdit, getStagedEdit } from "./staged-edits.js";
import { atomicWriteFile } from "./utils/atomic-write.js";

// ── Resolution device result ──────────────────────────────────────────────────

/** Result of applying or rejecting a staged edit. */
export interface ResolutionResult {
	/** Whether the operation was successful. */
	success: boolean;
	/** Human-readable message. */
	message: string;
	/** Number of files affected. */
	filesAffected?: number;
	/** Number of lines changed. */
	linesChanged?: number;
}

// ── Resolution arguments ──────────────────────────────────────────────────────

/** Arguments for xd://resolve and xd://reject. */
export interface ResolutionArgs {
	/** Reason for the accept/reject decision. */
	reason?: string;
	/** Display label for the staged edit, used in the reject message. */
	label?: string;
	/** Files to apply (for resolve). Falls back to staged edits singleton. */
	files?: Array<{
		path: string;
		content: string;
	}>;
}

// ── Resolve device ─────────────────────────────────────────────────────────────

/**
 * Apply a staged edit. This is the "resolve" device handler.
 * Writes the proposed content to disk for all files in the staged edit.
 */
export async function handleResolve(
	args: ResolutionArgs,
	cwd: string,
	store?: EditStore,
): Promise<ResolutionResult> {
	let files = args.files;

	// Fall back to staged edits singleton if no files provided
	if (!files || files.length === 0) {
		const staged = getStagedEdit();
		if (!staged?.files?.length) {
			return { success: false, message: "No staged edits to apply." };
		}
		files = staged.files;
	}

	let filesAffected = 0;
	let linesChanged = 0;

	for (const file of files) {
		const fullPath = path.resolve(cwd, file.path);
		try {
			// Create parent directories if needed
			const dir = path.dirname(fullPath);
			if (!fs.existsSync(dir)) {
				fs.mkdirSync(dir, { recursive: true });
			}
			// Skip writing if the file is already identical.
			if (fs.existsSync(fullPath)) {
				const current = fs.readFileSync(fullPath, "utf8");
				if (current === file.content) {
					continue;
				}
			}
			await atomicWriteFile(fullPath, file.content);
			filesAffected++;
			linesChanged += file.content.split("\n").length;
		} catch (e) {
			return {
				success: false,
				message: `Failed to write ${file.path}: ${e instanceof Error ? e.message : String(e)}`,
			};
		}
	}

	if (store) {
		store.clearStagedEdits();
	}
	clearStagedEdit();

	return {
		success: true,
		message: `Applied ${linesChanged} line(s) across ${filesAffected} file(s).`,
		filesAffected,
		linesChanged,
	};
}

// ── Reject device ──────────────────────────────────────────────────────────────

/**
 * Discard a staged edit. This is the "reject" device handler.
 * No disk changes — just clears the staging state.
 */
async function handleReject(
	args: ResolutionArgs,
	store?: EditStore,
): Promise<ResolutionResult> {
	if (store) {
		store.clearStagedEdits();
	}
	clearStagedEdit();

	return {
		success: true,
		message: `Rejected ${args.label || "staged edit"}. Changes discarded.`,
	};
}

// ── Device registry ────────────────────────────────────────────────────────────

/**
 * Device names for xd:// dispatch.
 */
const RESOLVE_DEVICE_NAME = "resolve";
const REJECT_DEVICE_NAME = "reject";

// ── xd:// device wiring ─────────────────────────────────────────────────────
// Mounted as ordinary discoverable tools (see default-tools.ts) so `write
// path="xd://resolve"` reaches them through the normal ToolRegistry dispatch.
// The `write` tool's resolveCall already parses the JSON body, so execute()
// receives args directly — no re-parsing needed here.

function resolutionTool(
	name: typeof RESOLVE_DEVICE_NAME | typeof REJECT_DEVICE_NAME,
): Tool {
	const applying = name === RESOLVE_DEVICE_NAME;
	return {
		name,
		label: applying ? "Resolve" : "Reject",
		description: applying
			? "Apply the most recently staged edit preview (from ast_edit, ast_grep, or a hashline edit) to disk."
			: "Discard the most recently staged edit preview without writing to disk.",
		promptSnippet: applying
			? "Apply a staged edit preview"
			: "Discard a staged edit preview",
		parameters: {
			type: "object",
			properties: {
				reason: {
					type: "string",
					description: "One-sentence reason for the decision.",
				},
			},
			required: ["reason"],
		},
		execute: async (args, ctx) => {
			const resolutionArgs = args as ResolutionArgs;
			const result = applying
				? await handleResolve(resolutionArgs, ctx.cwd || process.cwd())
				: await handleReject(resolutionArgs);
			return { content: result.message, isError: !result.success };
		},
	};
}

export const resolve: Tool = resolutionTool(RESOLVE_DEVICE_NAME);
export const reject: Tool = resolutionTool(REJECT_DEVICE_NAME);
