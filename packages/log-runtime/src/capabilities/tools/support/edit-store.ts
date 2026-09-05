// ── Edit store ────────────────────────────────────────────────────────────────
// Per-session store for edit state: file snapshots (byte-identical), hashline
// tags, clipboard registers, and the no-op loop guard. One store instance per
// edit session; every edit_file call that reads a file shares it.

import * as fs from "node:fs";
import { createHash } from "node:crypto";
import { quickFileFingerprint } from "./hashline";

/**
 * A byte-identical snapshot of a file at read time.
 * Used for stale detection and hashline anchor validation.
 */
export interface FileSnapshot {
	/** Absolute path to the file. */
	path: string;
	/** Byte-identical content at snapshot time (Buffer for exact comparison). */
	content: Buffer;
	/** SHA-256 hash of the full content. */
	hash: string;
	/** 4-char hashline tag. */
	tag: string;
	/** Fingerprint for quick stale check. */
	fingerprint: string;
	/** Timestamp when the snapshot was taken. */
	timestamp: number;
}

// ── Register (clipboard) ───────────────────────────────────────────────────────

/**
 * A named register for CUT/PUT operations.
 */
export interface Register {
	/** Lines captured by CUT. */
	lines: string[];
	/** Block text for block CUT operations. */
	block?: string;
}

// ── Staged edit ────────────────────────────────────────────────────────────────

/**
 * A staged (previewed but not yet applied) edit proposal.
 */
export interface StagedEdit {
	/** Unique proposal ID. */
	id: string;
	/** Files to be modified. */
	entries: Array<{
		path: string;
		oldContent: string;
		newContent: string;
		diff: string;
	}>;
	/** Total files to be changed. */
	filesCount: number;
	/** Total replacements across all files. */
	linesChanged: number;
	/** Created timestamp. */
	createdAt: number;
	/** Whether the edit was already applied or rejected. */
	resolved: boolean;
}

// ── Edit store ─────────────────────────────────────────────────────────────────

/**
 * Per-session edit store.
 */
export class EditStore {
	/** File snapshots: path → snapshot. */
	readonly #snapshots = new Map<string, FileSnapshot>();
	/** Clipboard registers: name → register. */
	readonly #registers = new Map<string, Register>();
	/** Staged edits: id → staged edit. */
	readonly #staged = new Map<string, StagedEdit>();
	/** Next staged edit ID counter. */
	#stagedCounter = 0;

	// ── File snapshots ─────────────────────────────────────────────────────────

	/**
	 * Record a file snapshot after reading it.
	 * Stores byte-identical content for stale detection.
	 */
	recordSnapshot(path: string, content: string): FileSnapshot {
		const raw = Buffer.from(content, "utf-8");
		const hash = createHash("sha256")
			.update(raw)
			.digest("hex");
		const tag = hash.slice(0, 4);
		const fingerprint = quickFileFingerprint(content);

		const snapshot: FileSnapshot = {
			path,
			content: raw,
			hash,
			tag,
			fingerprint,
			timestamp: Date.now(),
		};
		this.#snapshots.set(path, snapshot);
		return snapshot;
	}

	/**
	 * Get the snapshot for a file, if one exists.
	 */
	getSnapshot(path: string): FileSnapshot | undefined {
		return this.#snapshots.get(path);
	}

	/**
	 * Check if a file is stale (modified since last snapshot).
	 * Returns the stale reason, or undefined if the file is fresh.
	 */
	checkStale(path: string): string | undefined {
		const snapshot = this.#snapshots.get(path);
		if (!snapshot) return undefined;

		try {
			const currentRaw = fs.readFileSync(path);
			// Fast path: compare hashes
			const currentHash = createHash("sha256")
				.update(currentRaw)
				.digest("hex");
			if (currentHash === snapshot.hash) {
				return undefined; // Not stale
			}
			// Slow path: the file has changed since snapshot
			return `${path} has been modified since it was last read. The content-hash anchor is stale. Read it again before editing.`;
		} catch {
			// File doesn't exist
			return `${path} does not exist. Read it again before editing.`;
		}
	}

	/**
	 * Remove snapshot for a file (e.g., after successful edit).
	 */
	clearSnapshot(path: string): void {
		this.#snapshots.delete(path);
	}

	/**
	 * Clear all snapshots (e.g., after compaction).
	 */
	clearAll(): void {
		this.#snapshots.clear();
		this.#registers.clear();
	}

	// ── Registers ──────────────────────────────────────────────────────────────

	/**
	 * Store lines in a named register (for CUT/PUT).
	 */
	setRegister(name: string, lines: string[]): void {
		this.#registers.set(name, { lines });
	}

	/**
	 * Get a register's lines.
	 */
	getRegister(name: string): Register | undefined {
		return this.#registers.get(name);
	}

	// ── Staged edits ───────────────────────────────────────────────────────────

	/**
	 * Create a new staged edit proposal.
	 */
	stageEdit(entries: StagedEdit["entries"]): StagedEdit {
		const id = `edit-${++this.#stagedCounter}`;
		const totalLines = entries.reduce((sum, e) => sum + e.newContent.split("\n").length, 0);
		const staged: StagedEdit = {
			id,
			entries,
			filesCount: entries.length,
			linesChanged: totalLines,
			createdAt: Date.now(),
			resolved: false,
		};
		this.#staged.set(id, staged);
		return staged;
	}

	/**
	 * Get a staged edit by ID.
	 */
	getStagedEdit(id: string): StagedEdit | undefined {
		return this.#staged.get(id);
	}

	/**
	 * Resolve (apply or reject) a staged edit.
	 */
	resolveStagedEdit(id: string, applied: boolean): StagedEdit | undefined {
		const staged = this.#staged.get(id);
		if (!staged) return undefined;
		staged.resolved = true;
		if (applied) {
			this.#staged.delete(id);
		}
		return staged;
	}

	/**
	 * List all unresolved staged edits.
	 */
	listStagedEdits(): StagedEdit[] {
		return Array.from(this.#staged.values()).filter((s) => !s.resolved);
	}

	/**
	 * Clear all staged edits.
	 */
	clearStagedEdits(): void {
		this.#staged.clear();
	}
}

// ── Singleton store per session ────────────────────────────────────────────────

/**
 * Create a new edit store instance.
 * In practice, this would be instantiated per session/turn in the agent loop.
 */
export function createEditStore(): EditStore {
	return new EditStore();
}
