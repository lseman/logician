// ── Session Abstraction ─────────────────────────────────────────────────
// JSONL-based session persistence for the agent harness.
//
// A session is a named JSONL file where each line is a persisted message
// (role, content, timestamp). The harness writes a line after each message
// event (message_end), so the file is always up-to-date.
//
// Key operations:
// - append(msg): persist one message to the current session file
// - load(): return all messages from the session file
// - checkpoint(): write a snapshot marker for undo/restore
// - listSessions(): return all known session IDs
// - clear(): remove the session file and all checkpoints
//
// Design: minimal, file-based. No database. Each session is one JSONL file
// plus optional checkpoint markers. Sessions are stored under
// <cwd>/.logician/sessions/<sessionId>/

import {
	appendFileSync,
	existsSync,
	mkdirSync,
	readFileSync,
	readdirSync,
	rmSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { randomUUID } from "node:crypto";
import { dirname, join } from "node:path";

// ── Types ───────────────────────────────────────────────────────────────

/** A persisted session message. Mirrors the Message type for persistence. */
export interface SessionMessage {
	role: string;
	content: string | null;
	tool_call_id?: string;
	tool_calls?: Array<{
		id: string;
		name: string;
		arguments: string;
	}>;
	name?: string;
	timestamp: number;
	/** UUID for this message entry (tree-based tracking). */
	entryId?: string;
	/** UUID of the parent entry (enables tree traversal). */
	parentId?: string;
}

/** Checkpoint metadata written to a separate file. */
export interface SessionCheckpoint {
	timestamp: number;
	messageCount: number;
	/** File path for the checkpoint data (the full message array serialized). */
	dataFile: string;
}

/** Session metadata (name, creation time, message count). */
export interface SessionMeta {
	id: string;
	createdAt: number;
	messageCount: number;
	lastActivity: number;
	name?: string;
	/** UUID of the parent session (for forked sessions). */
	parentId?: string;
	/** Session format version (for migration). */
	version?: number;
}

/**
 * Configuration for session persistence.
 */
export interface SessionConfig {
	/** Base directory for session files (default: <cwd>/.logician/sessions). */
	baseDir?: string;
	/** Whether to enable session persistence (default: false). */
	enabled?: boolean;
	/** Maximum number of checkpoints to keep per session (default: 10). */
	maxCheckpoints?: number;
	/** UUID of parent session (for forked sessions). */
	parentId?: string;
	/** Session format version (auto-upgraded on load). */
	version?: number;
}

const DEFAULT_BASE_DIR = ".logician/sessions";
const DEFAULT_MAX_CHECKPOINTS = 10;
const SESSIONS_DIR = "sessions";
const CHECKPOINT_DIR = "checkpoints";
const META_FILE = "meta.json";

// ── Session class ───────────────────────────────────────────────────────

export class Session {
	private readonly dir: string;
	private readonly filePath: string;
	private readonly metaPath: string;
	private readonly checkpointDir: string;
	private maxCheckpoints: number;
	private messageCount = 0;
	private readonly createdAt: number;
	private lastActivity: number;
	private name?: string;
	private parentId?: string;
	private version = 2;

	constructor(
		private readonly sessionId: string,
		config?: SessionConfig,
	) {
		this.dir = join(
			config?.baseDir ?? DEFAULT_BASE_DIR,
			SESSIONS_DIR,
			sessionId,
		);
		this.filePath = join(this.dir, "messages.jsonl");
		this.metaPath = join(this.dir, META_FILE);
		this.checkpointDir = join(this.dir, CHECKPOINT_DIR);
		this.maxCheckpoints = config?.maxCheckpoints ?? DEFAULT_MAX_CHECKPOINTS;
		this.parentId = config?.parentId;
		this.createdAt = Date.now();
		this.lastActivity = this.createdAt;

		// Load existing version for migration
		const existingMeta = this.getMetaSilent();
		if (existingMeta) {
			this.version = (config?.version ?? existingMeta.version ?? 1) as number;
			this.migrateVersion();
		} else {
			this.version = config?.version ?? 2;
		}

		// Initialize session directory and meta
		this.init();
	}

	/** Create the session directory and meta file if they don't exist. */
	private init(): void {
		mkdirSync(this.dir, { recursive: true });
		mkdirSync(this.checkpointDir, { recursive: true });

		if (!existsSync(this.metaPath)) {
			writeFileSync(
				this.metaPath,
				JSON.stringify({
					id: this.sessionId,
					createdAt: this.createdAt,
					messageCount: 0,
					lastActivity: this.lastActivity,
					name: this.name,
				}),
				"utf8",
			);
		}
	}

	// ── Core operations ─────────────────────────────────────────────────

	/** Persist a message to the session file. */
	append(msg: SessionMessage): void {
		mkdirSync(dirname(this.filePath), { recursive: true });
		// Assign entryId if not present
		if (!msg.entryId) {
			msg.entryId = randomUUID();
		}
		// Inherit parentId from parent message if not set
		if (!msg.parentId) {
			const messages = this.load();
			if (messages.length > 0) {
				msg.parentId = messages[messages.length - 1].entryId;
			}
		}
		appendFileSync(this.filePath, `${JSON.stringify(msg)}\n`, "utf8");
		this.messageCount++;
		this.lastActivity = Date.now();
		this.updateMeta();
	}

	/** Load all messages from the session file. */
	load(): SessionMessage[] {
		if (!existsSync(this.filePath)) return [];
		const content = readFileSync(this.filePath, "utf8");
		const lines = content.trim().split("\n").filter(Boolean);
		return lines.map((line) => JSON.parse(line));
	}

	/** Save a checkpoint of the current state. Returns the checkpoint path. */
	saveCheckpoint(): string {
		const messages = this.load();
		const checkpoint: SessionCheckpoint = {
			timestamp: Date.now(),
			messageCount: messages.length,
			dataFile: join(this.checkpointDir, `checkpoint_${messages.length}.json`),
		};

		// Serialize full state for restore
		writeFileSync(checkpoint.dataFile, JSON.stringify(messages), "utf8");
		this.lastActivity = Date.now();
		this.updateMeta();

		// Prune old checkpoints
		this.pruneCheckpoints();

		return checkpoint.dataFile;
	}

	/** List available checkpoints for this session. */
	listCheckpoints(): Array<{
		dataFile: string;
		messageCount: number;
		timestamp: number;
	}> {
		if (!existsSync(this.checkpointDir)) return [];
		return readdirSync(this.checkpointDir)
			.filter((f) => f.startsWith("checkpoint_") && f.endsWith(".json"))
			.map((f) => {
				const content = JSON.parse(
					readFileSync(join(this.checkpointDir, f), "utf8"),
				);
				return {
					dataFile: join(this.checkpointDir, f),
					messageCount: content.length,
					timestamp: statSync(join(this.checkpointDir, f)).mtimeMs,
				};
			})
			.sort((a, b) => a.timestamp - b.timestamp);
	}

	/** Load a checkpoint by message count. Returns null if not found. */
	loadCheckpoint(messageCount: number): SessionMessage[] | null {
		const checkpoint = this.listCheckpoints().find(
			(c) => c.messageCount === messageCount,
		);
		if (!checkpoint) return null;
		return JSON.parse(readFileSync(checkpoint.dataFile, "utf8"));
	}

	// ── Metadata ────────────────────────────────────────────────────────

	getMeta(): SessionMeta {
		if (!existsSync(this.metaPath)) {
			this.init();
		}
		return JSON.parse(readFileSync(this.metaPath, "utf8"));
	}

	setName(name: string): void {
		this.name = name;
		this.updateMeta();
	}

	get dirPath(): string {
		return this.dir;
	}

	// ── Cleanup ─────────────────────────────────────────────────────────

	/** Remove the session directory and all checkpoints. */
	clear(): void {
		if (existsSync(this.dir)) {
			rmSync(this.dir, { recursive: true, force: true });
		}
	}

	/** Get the path of entries from root to the last message (tree traversal). */
	getPathToRoot(): SessionMessage[] {
		const messages = this.load();
		if (messages.length === 0) return [];

		// Build index by entryId
		const byId = new Map<string, SessionMessage>();
		for (const msg of messages) {
			byId.set(msg.entryId || msg.timestamp.toString(), msg);
		}

		// Walk from last message back to root
		const path: SessionMessage[] = [];
		let currentId: string | undefined = messages[messages.length - 1].entryId;
		let seen = new Set<string>();

		while (currentId && !seen.has(currentId)) {
			seen.add(currentId);
			const entry = byId.get(currentId);
			if (!entry) break;
			path.unshift(entry);
			currentId = entry.parentId;
		}

		return path;
	}

	/** Get last entryId in the session. */
	getLeafEntryId(): string | undefined {
		const messages = this.load();
		return messages.length > 0 ? messages[messages.length - 1].entryId : undefined;
	}

	/** Truncate the session file (keep only recent messages). */
	truncate(keepLast: number): void {
		const messages = this.load();
		const truncated = messages.slice(-keepLast);
		this.messageCount = truncated.length;
		writeFileSync(
			this.filePath,
			truncated.map((m) => JSON.stringify(m)).join("\n") + "\n",
			"utf8",
		);
		this.lastActivity = Date.now();
		this.updateMeta();
	}

	// ── Internals ───────────────────────────────────────────────────────

	private updateMeta(): void {
		writeFileSync(
			this.metaPath,
			JSON.stringify({
				id: this.sessionId,
				createdAt: this.createdAt,
				messageCount: this.messageCount,
				lastActivity: this.lastActivity,
				name: this.name,
			}),
			"utf8",
		);
	}

	private pruneCheckpoints(): void {
		const checkpoints = this.listCheckpoints();
		if (checkpoints.length <= this.maxCheckpoints) return;

		// Remove oldest checkpoints beyond the limit
		const toRemove = checkpoints.slice(
			0,
			checkpoints.length - this.maxCheckpoints,
		);
		for (const cp of toRemove) {
			try {
				rmSync(cp.dataFile, { force: true });
			} catch {
				// Best-effort cleanup
			}
		}
	}

	private getMetaSilent(): SessionMeta | null {
		if (!existsSync(this.metaPath)) return null;
		try {
			return JSON.parse(readFileSync(this.metaPath, "utf8")) as SessionMeta;
		} catch {
			return null;
		}
	}

	private migrateVersion(): void {
		// v1 → v2: add parentId to meta for fork support
		if (this.version < 2) {
			const meta = this.getMetaSilent();
			if (meta && !meta.parentId) {
				meta.parentId = undefined;
				writeFileSync(this.metaPath, JSON.stringify(meta, null, 2), "utf8");
			}
			this.version = 2;
		}
	}
}

// ── Session Manager ─────────────────────────────────────────────────────
// Manages multiple sessions: listing, loading, creating, forking.

export class SessionManager {
	private baseDir: string;

	constructor(config?: SessionConfig) {
		this.baseDir = join(config?.baseDir ?? DEFAULT_BASE_DIR, SESSIONS_DIR);
	}

	/** List all sessions with their metadata. */
	listSessions(): SessionMeta[] {
		if (!existsSync(this.baseDir)) return [];
		return readdirSync(this.baseDir)
			.filter((id) => {
				const dir = join(this.baseDir, id);
				return statSync(dir).isDirectory();
			})
			.map((id) => {
				const metaPath = join(this.baseDir, id, META_FILE);
				if (!existsSync(metaPath)) return null;
				return JSON.parse(readFileSync(metaPath, "utf8")) as SessionMeta;
			})
			.filter((meta): meta is SessionMeta => meta !== null)
			.sort((a, b) => b.lastActivity - a.lastActivity);
	}

	/** Open an existing session by ID. Returns null if not found. */
	getSession(sessionId: string): Session | null {
		const metaPath = join(this.baseDir, sessionId, META_FILE);
		if (!existsSync(metaPath)) return null;
		return new Session(sessionId, {
			baseDir: join(this.baseDir, ".."),
		});
	}

	/** Create a new session with a unique ID. */
	createSession(name?: string): Session {
		const sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
		const session = new Session(sessionId, {
			baseDir: join(this.baseDir, ".."),
		});
		if (name) session.setName(name);
		return session;
	}

	/**
	 * Fork an existing session, creating a child session that shares the
	 * parent's message tree but diverges from this point.
	 * The child inherits the parent's entryId as its parentId.
	 */
	forkSession(parentId: string, name?: string): Session | null {
		const parent = this.getSession(parentId);
		if (!parent) return null;

		const childId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
		const child = new Session(childId, {
			baseDir: join(this.baseDir, ".."),
			parentId,
			version: 2,
		});
		if (name) child.setName(name);
		return child;
	}

	/**
	 * Get the full session tree rooted at a given session ID.
	 * Returns an array of session IDs from root to leaf.
	 */
	getSessionTree(sessionId: string): SessionMeta[] {
		const tree: SessionMeta[] = [];
		let currentId: string | undefined = sessionId;

		while (currentId) {
			const session = this.getSession(currentId);
			if (!session) break;
			const meta = session.getMeta();
			tree.push(meta);
			currentId = meta.parentId;
		}

		return tree.reverse();
	}

	/** Delete a session by ID. */
	deleteSession(sessionId: string): void {
		const session = this.getSession(sessionId);
		session?.clear();
	}
}
