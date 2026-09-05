// ── MutationSession — session-owned file mutation module ───────────────────────
// A single, session-scoped module that unifies all file mutation paths (exact-text,
// hashline, AST, write-file). It provides versioned mutation handles, stale
// rejection, path policy, atomic writes, and diagnostic events.
//
// Format parsers (edit-file.ts, hashline.ts, ast-edit.ts) produce MutationProposal
// objects; the session owns the lifecycle: begin → apply/preview → commit.

import * as fs from "node:fs";
import * as path from "node:path";
import { createHash } from "node:crypto";
import type { EditStore } from "../support/edit-store.js";
import { atomicWriteFile, AtomicWriteOptions } from "../support/utils/atomic-write.js";
import { generateEditDiffs } from "../support/utils/diff-utils.js";
import { ensureInsideCwd } from "../support/utils/path-utils.js";

// ── Core types ────────────────────────────────────────────────────────────────

export interface MutationProposal {
	/** Absolute file path (already resolved through path policy). */
	path: string;
	/** Byte-identical content observed before the mutation. */
	before: string;
	/** SHA-256 hash of `before`. */
	beforeHash: string;
	/** New content after applying the edit. */
	after: string;
	/** Whether this is a new file being created. */
	newFile?: boolean;
}

export interface MutationResult {
	/** Whether content was written to disk. */
	applied: boolean;
	/** Whether `after` differs from `before`. */
	changed: boolean;
	/** Absolute path of the affected file. */
	path: string;
	/** SHA-256 hash of `before`. */
	beforeHash: string;
	/** SHA-256 hash of `after`. */
	afterHash: string;
	/** Number of lines changed (heuristic: max of before/after line counts). */
	linesChanged: number;
	/** Number of files affected (always 1 for single-file proposals). */
	filesAffected: number;
	/** Unified diff string. */
	diff: string;
	/** Error message if the mutation failed (applied may still be true for partial failures). */
	error?: string;
}

export interface MutationHandle {
	/** Monotonically increasing version for this path. */
	version: number;
	/** Returns true if a newer handle has superseded this one. */
	isStale(): boolean;
}

export interface MutationPathPolicy {
	/** Allowed absolute paths. */
	allowedPaths?: string[];
	/** Allow all paths regardless of allowedPaths. */
	allowAllPaths?: boolean;
}

export interface MutationDiagnosticEvent {
	/** Absolute file path. */
	path: string;
	/** Summary text. */
	summary: string;
	/** Diagnostic messages (LSP format). */
	messages: string[];
	/** Whether the diagnostics errored. */
	errored: boolean;
	/** Staleness check — returns true if this event belongs to an older mutation. */
	isStale: () => boolean;
}

// ── MutationSession ───────────────────────────────────────────────────────────

/**
 * Session-scoped mutation coordinator. All file mutations flow through here.
 */
export class MutationSession {
	readonly #store: EditStore;
	readonly #cwd: string;
	readonly #policy: MutationPathPolicy;

	/** Version counters per file path. */
	readonly #versions = new Map<string, number>();
	/** Latest version for each path at time of last mutation. */
	readonly #mutationVersions = new Map<string, number>();
	/** Pending diagnostic callbacks. */
	readonly #pendingDiagnostics = new Map<string, MutationDiagnosticEvent>();

	constructor(
		store: EditStore,
		cwd: string,
		policy: MutationPathPolicy = {},
	) {
		this.#store = store;
		this.#cwd = cwd;
		this.#policy = policy;
	}

	/**
	 * Begin a mutation on a file. Returns a handle for version tracking.
	 * The handle's version increments on each subsequent begin for the same path.
	 */
	begin(filePath: string): MutationHandle {
		const key = this.#normalizePath(filePath);
		const version = (this.#versions.get(key) ?? 0) + 1;
		this.#versions.set(key, version);

		return {
			version,
			isStale: () => this.#mutationVersions.get(key) !== version,
		};
	}

	/**
	 * Preview a mutation without writing to disk. Returns a result with
	 * changed = true/false, but applied is always false.
	 */
	async preview(proposal: MutationProposal): Promise<MutationResult> {
		const beforeHash = this.#hash(proposal.before);
		const afterHash = this.#hash(proposal.after);
		const changed = beforeHash !== afterHash;
		const linesChanged = changed
			? Math.max(
					proposal.after.split("\n").length,
					proposal.before.split("\n").length,
				)
			: 0;
		const diff = changed
			? generateEditDiffs(proposal.path, proposal.before, proposal.after).diff
			: "";

		return {
			applied: false,
			changed,
			path: proposal.path,
			beforeHash,
			afterHash,
			linesChanged,
			filesAffected: 1,
			diff,
		};
	}

	/**
	 * Apply a mutation proposal. Validates path policy, stale detection,
	 * and writes atomically. Returns the result with applied/changed status.
	 */
	async apply(
		proposal: MutationProposal,
		writeOptions?: AtomicWriteOptions,
	): Promise<MutationResult> {
		// ── Path policy validation ───────────────────────────────────────────
		const resolved = proposal.path.startsWith("/")
			? proposal.path
			: path.resolve(this.#cwd, proposal.path);
		ensureInsideCwd(this.#cwd, resolved, this.#policy.allowedPaths, this.#policy.allowAllPaths);

		const beforeHash = this.#hash(proposal.before);
		const afterHash = this.#hash(proposal.after);
		const changed = beforeHash !== afterHash;
		const linesChanged = changed
			? Math.max(
					proposal.after.split("\n").length,
					proposal.before.split("\n").length,
				)
			: 0;

		// ── Skip if no actual change ─────────────────────────────────────────
		if (!changed) {
			return {
				applied: false,
				changed: false,
				path: resolved,
				beforeHash,
				afterHash,
				linesChanged: 0,
				filesAffected: 0,
				diff: "",
			};
		}

		// ── Stale detection ──────────────────────────────────────────────────
		// Re-read the file to verify it hasn't changed since we observed `before`.
		// For new files, the file must not exist.
		if (proposal.newFile) {
			if (fs.existsSync(resolved)) {
				return {
					applied: false,
					changed: true,
					path: resolved,
					beforeHash,
					afterHash,
					linesChanged,
					filesAffected: 0,
					diff: generateEditDiffs(resolved, proposal.before, proposal.after).diff,
					error: `${resolved} already exists. Cannot create new file.`,
				};
			}
		} else {
			const current = fs.readFileSync(resolved, "utf8");
			const currentHash = this.#hash(current);
			if (currentHash !== beforeHash) {
				return {
					applied: false,
					changed: true,
					path: resolved,
					beforeHash,
					afterHash,
					linesChanged,
					filesAffected: 0,
					diff: generateEditDiffs(resolved, proposal.before, proposal.after).diff,
					error: `${resolved} has been modified since it was last read. Read it again before editing.`,
				};
			}
		}

		// ── Atomic write ─────────────────────────────────────────────────────
		try {
			const opts = {
				...writeOptions,
				expectedContent: changed ? proposal.before : undefined,
			};
			await atomicWriteFile(resolved, proposal.after, opts);

			// Clear the snapshot in the edit store so stale detection is reset.
			this.#store.clearSnapshot(resolved);

			// Update the mutation version for this path.
			this.#mutationVersions.set(this.#normalizePath(resolved), (this.#mutationVersions.get(this.#normalizePath(resolved)) ?? 0) + 1);

			const diff = generateEditDiffs(resolved, proposal.before, proposal.after).diff;

			return {
				applied: true,
				changed: true,
				path: resolved,
				beforeHash,
				afterHash,
				linesChanged,
				filesAffected: 1,
				diff,
			};
		} catch (error: unknown) {
			const errorMessage =
				error instanceof Error ? error.message : String(error);
			return {
				applied: false,
				changed: true,
				path: resolved,
				beforeHash,
				afterHash,
				linesChanged,
				filesAffected: 0,
				diff: generateEditDiffs(resolved, proposal.before, proposal.after).diff,
				error: `Failed to write ${resolved}: ${errorMessage}`,
			};
		}
	}

	/**
	 * Register a deferred diagnostic event for later injection into the LSP flow.
	 * Returns a handle that can be used to check staleness.
	 */
	registerDiagnostic(
		path: string,
		summary: string,
		messages: string[],
		errored: boolean,
	): MutationDiagnosticEvent {
		const version = (this.#mutationVersions.get(this.#normalizePath(path)) ?? 0) + 1;
		this.#mutationVersions.set(this.#normalizePath(path), version);

		const entry: MutationDiagnosticEvent = {
			path,
			summary,
			messages,
			errored,
			isStale: () => this.#mutationVersions.get(this.#normalizePath(path)) !== version,
		};

		this.#pendingDiagnostics.set(path, entry);
		return entry;
	}

	/**
	 * Get the pending diagnostic event for a path, if one exists.
	 */
	getDiagnostic(path: string): MutationDiagnosticEvent | undefined {
		return this.#pendingDiagnostics.get(path);
	}

	/**
	 * Clear all pending diagnostics for a path (consumes the event).
	 */
	clearDiagnostic(path: string): void {
		this.#pendingDiagnostics.delete(path);
	}

	/**
	 * Record a snapshot for a file (used after read operations).
	 */
	recordSnapshot(filePath: string, content: string): void {
		this.#store.recordSnapshot(filePath, content);
	}

	/**
	 * Check if a file is stale since last snapshot.
	 */
	checkStale(filePath: string): string | undefined {
		return this.#store.checkStale(filePath);
	}

	// ── Internal helpers ───────────────────────────────────────────────────────

	#hash(content: string): string {
		return createHash("sha256").update(content, "utf-8").digest("hex");
	}

	#normalizePath(filePath: string): string {
		try {
			return fs.realpathSync(filePath);
		} catch {
			return filePath;
		}
	}
}

// ── Factory ────────────────────────────────────────────────────────────────────

/**
 * Create a new mutation session.
 */
export function createMutationSession(
	store: EditStore,
	cwd: string,
	policy?: MutationPathPolicy,
): MutationSession {
	return new MutationSession(store, cwd, policy);
}
