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
// - listSessions(): return all known session IDs
// - checkout(entryId): select the active leaf in the conversation tree
// - clear(): remove the session directory
//
// Design: minimal, file-based. No database. Each session is an append-only
// JSONL conversation tree plus metadata naming its active leaf. Sessions live at
// <cwd>/.logician/sessions/<sessionId>/

import { randomUUID } from "node:crypto";
import {
	appendFileSync,
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	rmSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { markPathIgnoredByCloudSync } from "../../tools/shared/path-utils.ts";

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

export interface MessageSessionEntry {
	type: "message";
	id: string;
	parentId?: string;
	timestamp: number;
	message: SessionMessage;
}

export interface ModelChangeSessionEntry {
	type: "model_change";
	id: string;
	parentId?: string;
	timestamp: number;
	model: string;
}

export interface ThinkingLevelSessionEntry {
	type: "thinking_level_changed";
	id: string;
	parentId?: string;
	timestamp: number;
	thinkingLevel: string;
}

export interface ActiveToolsSessionEntry {
	type: "active_tools_change";
	id: string;
	parentId?: string;
	timestamp: number;
	activeToolNames: string[];
}

export interface SettingsChangeSessionEntry {
	type: "settings_change";
	id: string;
	parentId?: string;
	timestamp: number;
	key: string;
	value: string | null;
	previousValue?: string | null;
}

export interface CompactionSessionEntry {
	type: "compaction";
	id: string;
	parentId?: string;
	timestamp: number;
	summary: string;
	firstKeptEntryId?: string;
	tokensBefore: number;
}

export interface BranchSummarySessionEntry {
	type: "branch_summary";
	id: string;
	parentId?: string;
	timestamp: number;
	fromId?: string;
	summary: string;
}

export interface LabelSessionEntry {
	type: "label";
	id: string;
	parentId?: string;
	timestamp: number;
	targetId: string;
	label?: string;
}

/**
 * Opaque app-defined data, keyed by `customType`. Lets a consumer (e.g. the
 * TUI) persist its own render-time shape in the same tree without agent-core
 * needing to know what it is. Never participates in `buildContext()`.
 */
export interface CustomSessionEntry<T = unknown> {
	type: "custom";
	id: string;
	parentId?: string;
	timestamp: number;
	customType: string;
	data: T;
}

export type SessionEntry =
	| MessageSessionEntry
	| ModelChangeSessionEntry
	| ThinkingLevelSessionEntry
	| ActiveToolsSessionEntry
	| SettingsChangeSessionEntry
	| CompactionSessionEntry
	| BranchSummarySessionEntry
	| LabelSessionEntry
	| CustomSessionEntry;

export interface SessionContext {
	messages: SessionMessage[];
	model?: string;
	thinkingLevel?: string;
	activeToolNames?: string[];
	settings: Map<string, string | null>;
	labels: Map<string, string>;
}

/** Session metadata (name, creation time, message count). */
export interface SessionMeta {
	id: string;
	createdAt: number;
	messageCount: number;
	lastActivity: number;
	name?: string;
	/** Working directory the session was started in — scopes browser listing to a project. */
	cwd?: string;
	/** UUID of the parent session (for forked sessions). */
	parentId?: string;
	/** Selected entry leaf for durable branch checkout. */
	activeLeafId?: string;
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
	/** Working directory the session was started in — scopes browser listing to a project. */
	cwd?: string;
	/** UUID of the parent session (for forked sessions). */
	parentId?: string;
	/** Session format version (auto-upgraded on load). */
	version?: number;
}

/** Session listing entry for the browser UI — metadata plus a text preview. */
export interface SessionInfo extends SessionMeta {
	/** First user message text, for browser preview. */
	preview: string;
}

const DEFAULT_BASE_DIR = ".logician/sessions";
const SESSIONS_DIR = "sessions";
const META_FILE = "meta.json";

// ── Session class ───────────────────────────────────────────────────────
// Append-only JSONL conversation tree for one harness session. Execution state
// belongs exclusively to the Run Kernel. Also the sole persistence layer for
// TUI conversation history and its session browser — see SessionManager below.

export class Session {
	private readonly dir: string;
	private readonly filePath: string;
	private readonly metaPath: string;
	private messageCount = 0;
	private readonly createdAt: number;
	private lastActivity: number;
	private name?: string;
	private cwd?: string;
	private parentId?: string;
	private activeLeafId?: string;
	private version = 3;

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
		this.parentId = config?.parentId;
		this.cwd = config?.cwd;
		this.createdAt = Date.now();
		this.lastActivity = this.createdAt;

		// Load existing version for migration
		const existingMeta = this.getMetaSilent();
		if (existingMeta) {
			this.createdAt = existingMeta.createdAt ?? this.createdAt;
			this.messageCount = existingMeta.messageCount ?? this.messageCount;
			this.lastActivity = existingMeta.lastActivity ?? this.lastActivity;
			this.name = existingMeta.name;
			this.cwd = config?.cwd ?? existingMeta.cwd;
			this.parentId = config?.parentId ?? existingMeta.parentId;
			this.activeLeafId = existingMeta.activeLeafId;
			this.version = (config?.version ?? existingMeta.version ?? 1) as number;
			this.migrateVersion();
		} else {
			this.version = config?.version ?? 3;
		}

		// Initialize session directory and meta
		this.init();
	}

	/** Create the session directory and meta file if they don't exist. */
	private init(): void {
		mkdirSync(this.dir, { recursive: true });
		markPathIgnoredByCloudSync(this.dir);

		if (!existsSync(this.metaPath)) {
			writeFileSync(
				this.metaPath,
				JSON.stringify({
					id: this.sessionId,
					createdAt: this.createdAt,
					messageCount: 0,
					lastActivity: this.lastActivity,
					cwd: this.cwd,
					name: this.name,
					parentId: this.parentId,
					version: this.version,
				}),
				"utf8",
			);
		}
	}

	// ── Core operations ─────────────────────────────────────────────────

	/** Persist a message to the session file. */
	append(msg: SessionMessage): void {
		this.appendMessageEntry(msg);
	}

	appendMessageEntry(msg: SessionMessage): void {
		this.appendEntry({
			type: "message",
			id: msg.entryId ?? randomUUID(),
			parentId: msg.parentId,
			timestamp: msg.timestamp ?? Date.now(),
			message: msg,
		});
	}

	appendModelChange(model: string): void {
		this.appendEntry({
			type: "model_change",
			id: randomUUID(),
			timestamp: Date.now(),
			model,
		});
		this.appendSettingsChange("model", model);
	}

	appendThinkingLevelChange(thinkingLevel: string): void {
		this.appendEntry({
			type: "thinking_level_changed",
			id: randomUUID(),
			timestamp: Date.now(),
			thinkingLevel,
		});
		this.appendSettingsChange("thinking_level", thinkingLevel);
	}

	appendSettingsChange(
		key: string,
		value: string | null,
		previousValue?: string | null,
	): void {
		this.appendEntry({
			type: "settings_change",
			id: randomUUID(),
			timestamp: Date.now(),
			key,
			value,
			previousValue,
		});
	}

	appendActiveToolsChange(activeToolNames: string[]): void {
		this.appendEntry({
			type: "active_tools_change",
			id: randomUUID(),
			timestamp: Date.now(),
			activeToolNames,
		});
	}

	appendCompaction(
		summary: string,
		tokensBefore: number,
		firstKeptEntryId?: string,
	): void {
		this.appendEntry({
			type: "compaction",
			id: randomUUID(),
			timestamp: Date.now(),
			summary,
			firstKeptEntryId,
			tokensBefore,
		});
	}

	appendBranchSummary(summary: string, fromId?: string): void {
		this.appendEntry({
			type: "branch_summary",
			id: randomUUID(),
			timestamp: Date.now(),
			summary,
			fromId,
		});
	}

	appendLabel(targetId: string, label?: string): void {
		this.appendEntry({
			type: "label",
			id: randomUUID(),
			timestamp: Date.now(),
			targetId,
			label,
		});
	}

	appendCustom<T>(customType: string, data: T): void {
		this.appendEntry({
			type: "custom",
			id: randomUUID(),
			timestamp: Date.now(),
			customType,
			data,
		});
	}

	appendEntry(entry: SessionEntry): void {
		mkdirSync(dirname(this.filePath), { recursive: true });
		if (!entry.parentId) {
			entry.parentId = this.activeLeafId;
		}
		if (entry.type === "message") {
			entry.message.entryId = entry.id;
			entry.message.parentId = entry.parentId;
		}
		appendFileSync(this.filePath, `${JSON.stringify(entry)}\n`, "utf8");
		this.activeLeafId = entry.id;
		this.messageCount++;
		this.lastActivity = Date.now();
		this.updateMeta();
	}

	/** Load all messages from the session file. */
	load(): SessionMessage[] {
		return this.getPathToRootEntries()
			.filter((entry): entry is MessageSessionEntry => entry.type === "message")
			.map(entry => entry.message);
	}

	loadEntries(): SessionEntry[] {
		if (!existsSync(this.filePath)) return [];
		const content = readFileSync(this.filePath, "utf8");
		const lines = content.trim().split("\n").filter(Boolean);
		return lines.map(line => this.parseEntryLine(line));
	}

	buildContext(): SessionContext {
		const entries = this.getPathToRootEntries();
		const labels = new Map<string, string>();
		const settings = new Map<string, string | null>();
		const context: SessionContext = { messages: [], labels, settings };
		let lastCompaction: CompactionSessionEntry | undefined;
		for (const entry of entries) {
			if (entry.type === "model_change") {
				context.model = entry.model;
				settings.set("model", entry.model);
			} else if (entry.type === "thinking_level_changed") {
				context.thinkingLevel = entry.thinkingLevel;
				settings.set("thinking_level", entry.thinkingLevel);
			} else if (entry.type === "settings_change") {
				settings.set(entry.key, entry.value);
				if (entry.key === "model" && entry.value) context.model = entry.value;
				if (entry.key === "thinking_level" && entry.value) {
					context.thinkingLevel = entry.value;
				}
			} else if (entry.type === "active_tools_change") {
				context.activeToolNames = [...entry.activeToolNames];
			} else if (entry.type === "label") {
				if (entry.label) labels.set(entry.targetId, entry.label);
				else labels.delete(entry.targetId);
			} else if (entry.type === "compaction") {
				lastCompaction = entry;
			}
		}

		if (lastCompaction) {
			context.messages.push({
				role: "system",
				content: `<compaction_summary>${lastCompaction.summary}</compaction_summary>`,
				timestamp: lastCompaction.timestamp,
				entryId: lastCompaction.id,
				parentId: lastCompaction.parentId,
			});
		}
		const compactionIndex = lastCompaction
			? entries.indexOf(lastCompaction)
			: -1;
		const keptStartIndex = lastCompaction?.firstKeptEntryId
			? entries.findIndex(entry => entry.id === lastCompaction.firstKeptEntryId)
			: -1;
		const contextEntries = lastCompaction
			? [
					...(keptStartIndex >= 0
						? entries.slice(keptStartIndex, compactionIndex)
						: []),
					...entries.slice(compactionIndex + 1),
				]
			: entries;
		for (const entry of contextEntries) {
			if (entry.type === "message") {
				context.messages.push(entry.message);
			} else if (entry.type === "branch_summary") {
				context.messages.push({
					role: "assistant",
					content: `Branch summary: ${entry.summary}`,
					timestamp: entry.timestamp,
					entryId: entry.id,
					parentId: entry.parentId,
				});
			}
		}
		return context;
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

	getCwd(): string | undefined {
		return this.cwd;
	}

	// ── Cleanup ─────────────────────────────────────────────────────────

	/** Remove the session directory and its conversation data. */
	clear(): void {
		if (existsSync(this.dir)) {
			rmSync(this.dir, { recursive: true, force: true });
		}
	}

	/** Get the path of entries from root to the last message (tree traversal). */
	getPathToRoot(): SessionMessage[] {
		return this.getPathToRootEntries()
			.filter((entry): entry is MessageSessionEntry => entry.type === "message")
			.map(entry => entry.message);
	}

	getPathToRootEntries(): SessionEntry[] {
		const entries = this.loadEntries();
		if (entries.length === 0) return [];

		const byId = new Map<string, SessionEntry>();
		for (const entry of entries) {
			byId.set(entry.id, entry);
		}

		const path: SessionEntry[] = [];
		let currentId: string | undefined =
			this.activeLeafId ?? entries[entries.length - 1].id;
		const seen = new Set<string>();

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
		return this.activeLeafId;
	}

	checkout(entryId?: string): void {
		if (entryId && !this.loadEntries().some(entry => entry.id === entryId))
			throw new Error(`Cannot checkout unknown session entry ${entryId}`);
		this.activeLeafId = entryId;
		this.lastActivity = Date.now();
		this.updateMeta();
	}

	/** Truncate the session file (keep only recent messages). */
	truncate(keepLast: number): void {
		const messages = this.load();
		const truncated = messages.slice(-keepLast);
		const entries: MessageSessionEntry[] = [];
		for (const message of truncated) {
			const id = message.entryId ?? randomUUID();
			const parentId = entries.at(-1)?.id;
			entries.push({
				type: "message",
				id,
				parentId,
				timestamp: message.timestamp,
				message: { ...message, entryId: id, parentId },
			});
		}
		this.messageCount = truncated.length;
		writeFileSync(
			this.filePath,
			`${entries
				.map(entry => JSON.stringify(entry satisfies MessageSessionEntry))
				.join("\n")}\n`,
			"utf8",
		);
		this.activeLeafId = entries.at(-1)?.id;
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
				cwd: this.cwd,
				parentId: this.parentId,
				activeLeafId: this.activeLeafId,
				version: this.version,
			}),
			"utf8",
		);
	}

	private parseEntryLine(line: string): SessionEntry {
		const parsed = JSON.parse(line);
		if (
			parsed &&
			typeof parsed === "object" &&
			typeof parsed.type === "string"
		) {
			return parsed as SessionEntry;
		}
		const message = parsed as SessionMessage;
		const id = message.entryId ?? `${message.timestamp}`;
		return {
			type: "message",
			id,
			parentId: message.parentId,
			timestamp: message.timestamp ?? Date.now(),
			message: { ...message, entryId: id },
		};
	}

	private getMetaSilent(): SessionMeta | null {
		if (!existsSync(this.metaPath)) return null;
		try {
			return JSON.parse(readFileSync(this.metaPath, "utf8")) as SessionMeta;
		} catch (_e: unknown) {
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
		if (this.version < 3) {
			const meta = this.getMetaSilent();
			this.activeLeafId = this.loadEntries().at(-1)?.id;
			if (meta) {
				meta.activeLeafId = this.activeLeafId;
				meta.version = 3;
				writeFileSync(this.metaPath, JSON.stringify(meta, null, 2), "utf8");
			}
			this.version = 3;
		}
	}
}

// ── Session Manager ─────────────────────────────────────────────────────
// Manages multiple JSONL Session journals (above): listing, loading,
// creating, forking, renaming, deleting — the sole backing store for the
// TUI's session browser (project-scoped by cwd).

export class SessionManager {
	private baseDir: string;

	constructor(config?: SessionConfig) {
		this.baseDir = join(config?.baseDir ?? DEFAULT_BASE_DIR, SESSIONS_DIR);
	}

	private allMeta(): SessionMeta[] {
		if (!existsSync(this.baseDir)) return [];
		return readdirSync(this.baseDir)
			.filter(id => {
				const dir = join(this.baseDir, id);
				return statSync(dir).isDirectory();
			})
			.map(id => {
				const metaPath = join(this.baseDir, id, META_FILE);
				if (!existsSync(metaPath)) return null;
				return JSON.parse(readFileSync(metaPath, "utf8")) as SessionMeta;
			})
			.filter((meta): meta is SessionMeta => meta !== null);
	}

	/** List all sessions with their metadata, newest-active first. */
	listSessions(): SessionMeta[] {
		return this.allMeta().sort((a, b) => b.lastActivity - a.lastActivity);
	}

	/**
	 * List sessions scoped to a working directory, each with a text preview
	 * (first user message) for the session browser. Sessions predating cwd
	 * scoping (no `cwd` recorded) are excluded — they can still be opened by id.
	 */
	listSessionInfos(cwd: string): SessionInfo[] {
		const resolvedCwd = resolve(cwd);
		return this.allMeta()
			.filter(meta => meta.cwd && resolve(meta.cwd) === resolvedCwd)
			.sort((a, b) => b.lastActivity - a.lastActivity)
			.map(meta => ({ ...meta, preview: this.previewFor(meta.id) }));
	}

	private previewFor(sessionId: string): string {
		const session = this.getSession(sessionId);
		if (!session) return "";
		const firstUserMessage = session
			.getPathToRootEntries()
			.find(
				(entry): entry is MessageSessionEntry =>
					entry.type === "message" && entry.message.role === "user",
			);
		const content = firstUserMessage?.message.content;
		return typeof content === "string" ? content : "";
	}

	/** Open an existing session by ID. Returns null if not found. */
	getSession(sessionId: string): Session | null {
		const metaPath = join(this.baseDir, sessionId, META_FILE);
		if (!existsSync(metaPath)) return null;
		return new Session(sessionId, {
			baseDir: join(this.baseDir, ".."),
		});
	}

	/** Create a new session with a unique ID, scoped to a working directory. */
	createSession(cwd: string, name?: string): Session {
		const sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
		const session = new Session(sessionId, {
			baseDir: join(this.baseDir, ".."),
			cwd,
		});
		if (name) session.setName(name);
		return session;
	}

	/** Rename a session. Returns false if the session doesn't exist. */
	renameSession(sessionId: string, name: string): boolean {
		const session = this.getSession(sessionId);
		if (!session) return false;
		session.setName(name);
		return true;
	}

	/** Delete a session by ID. Returns false if it didn't exist. */
	deleteSession(sessionId: string): boolean {
		const session = this.getSession(sessionId);
		if (!session) return false;
		session.clear();
		return true;
	}

	/** Delete all but the `keep` most recently active sessions for a cwd. Returns count deleted. */
	keepRecent(cwd: string, keep: number): number {
		const infos = this.listSessionInfos(cwd);
		const toDelete = infos.slice(keep);
		for (const info of toDelete) this.deleteSession(info.id);
		return toDelete.length;
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
}
