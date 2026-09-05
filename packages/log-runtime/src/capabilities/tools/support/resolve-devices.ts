// ── xd://resolve and xd://reject devices ──────────────────────────────────────
// These devices handle the accept/reject flow for staged (previewed) edits.
// They are dispatched via: write xd://resolve {reason, sourceToolName, label, ...}
//                         write xd://reject  {reason, sourceToolName, label, ...}
//
// The system prompt describes them as:
//   write xd://resolve: one-line reason → disk move happens (atomic, all or nothing)
//   write xd://reject: discard the staged changes

import * as fs from "node:fs";
import * as path from "node:path";
import type { EditStore } from "./edit-store.js";
import { createEditStore } from "./edit-store.js";
import { atomicWriteFile } from "./utils/atomic-write.js";
import { getStagedEdit, clearStagedEdit } from "./staged-edits.js";

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
	reason: string;
	/** Name of the tool that created the staged edit. */
	sourceToolName: string;
	/** Display label for the staged edit. */
	label: string;
	/** The staged edit ID (optional — if omitted, uses the most recent). */
	editId?: string;
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
		if (!staged || !staged.files?.length) {
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
export async function handleReject(
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
export const RESOLVE_DEVICE_NAME = "resolve";
export const REJECT_DEVICE_NAME = "reject";

/**
 * Check if a device name is a resolution device.
 */
export function isResolutionDeviceName(name: string): boolean {
	return name === RESOLVE_DEVICE_NAME || name === REJECT_DEVICE_NAME;
}

/**
 * Execute a resolution device call.
 */
export async function executeResolutionDevice(
	deviceName: string,
	content: string,
	cwd: string,
	store?: EditStore,
): Promise<ResolutionResult> {
	let args: ResolutionArgs;
	try {
		args = JSON.parse(content) as ResolutionArgs;
	} catch {
		return { success: false, message: `Invalid JSON: ${content.slice(0, 100)}...` };
	}

	if (deviceName === RESOLVE_DEVICE_NAME) {
		return handleResolve(args, cwd, store);
	} else if (deviceName === REJECT_DEVICE_NAME) {
		return handleReject(args, store);
	}

	return { success: false, message: `Unknown resolution device: ${deviceName}` };
}

/**
 * Create a default edit store for simple usage.
 */
export function createDefaultResolutionStore(): EditStore {
	return createEditStore();
}
