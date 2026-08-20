import { randomUUID } from "node:crypto";
import {
	appendFileSync,
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { markPathIgnoredByCloudSync } from "../tools/path-utils.ts";

export interface SessionMessage {
	role: string;
	content: string | null;
	tool_call_id?: string;
	tool_calls?: Array<{ id: string; name: string; arguments: string }>;
	name?: string;
	timestamp: number;
	entryId?: string;
	parentId?: string;
}

export interface SessionEntry {
	type: string;
	id: string;
	parentId?: string;
	timestamp: number;
	[key: string]: unknown;
}

export interface SessionConfig {
	baseDir?: string;
	enabled?: boolean;
	cwd?: string;
	parentId?: string;
	version?: number;
}

export interface SessionMeta {
	id: string;
	createdAt: number;
	messageCount: number;
	lastActivity: number;
	name?: string;
	cwd?: string;
	parentId?: string;
	activeLeafId?: string;
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

		this.init();
	}

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

	append(msg: SessionMessage): void {
		this.appendEntry({
			type: "message",
			id: msg.entryId ?? randomUUID(),
			parentId: msg.parentId,
			timestamp: msg.timestamp ?? Date.now(),
			message: msg,
		} as SessionEntry);
	}

	load(): SessionMessage[] {
		if (!existsSync(this.filePath)) return [];
		const content = readFileSync(this.filePath, "utf8");
		const lines = content.trim().split("\n").filter(Boolean);
		return lines
			.map(line => {
				try {
					const entry = JSON.parse(line) as {
						type: string;
						message?: SessionMessage;
					};
					return entry.type === "message" ? entry.message : null;
				} catch {
					return null;
				}
			})
			.filter((m): m is SessionMessage => m !== null);
	}

	/** Persist one opaque app-owned entry, tagged by `customType`, to the same JSONL file `load()`/`append()` use. */
	appendCustom<T>(customType: string, data: T): void {
		this.appendEntry({
			type: "custom",
			id: randomUUID(),
			timestamp: Date.now(),
			customType,
			data,
		} as SessionEntry);
	}

	/** Reload every custom entry of the given type, in append order. */
	loadCustom<T>(customType: string): T[] {
		if (!existsSync(this.filePath)) return [];
		const content = readFileSync(this.filePath, "utf8");
		const lines = content.trim().split("\n").filter(Boolean);
		return lines
			.map(line => {
				try {
					const entry = JSON.parse(line) as {
						type: string;
						customType?: string;
						data?: T;
					};
					return entry.type === "custom" && entry.customType === customType
						? (entry.data as T)
						: null;
				} catch {
					return null;
				}
			})
			.filter((data): data is T => data !== null);
	}

	updateMeta(): void {
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

	getMetaSilent(): SessionMeta | null {
		if (!existsSync(this.metaPath)) return null;
		try {
			return JSON.parse(readFileSync(this.metaPath, "utf8")) as SessionMeta;
		} catch {
			return null;
		}
	}

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

	/** Remove the session directory and its conversation data. */
	clear(): void {
		if (existsSync(this.dir)) {
			rmSync(this.dir, { recursive: true, force: true });
		}
	}

	private appendEntry(entry: SessionEntry): void {
		mkdirSync(dirname(this.filePath), { recursive: true });
		if (!entry.parentId) entry.parentId = this.activeLeafId;
		appendFileSync(this.filePath, `${JSON.stringify(entry)}\n`, "utf8");
		this.activeLeafId = entry.id;
		this.messageCount++;
		this.lastActivity = Date.now();
		this.updateMeta();
	}

	private migrateVersion(): void {
		// Placeholder for version migration logic
	}

	// SessionManager for listing sessions
	static listSessions(baseDir?: string): Array<{
		id: string;
		name?: string;
		messageCount: number;
		lastActivity: number;
	}> {
		const dir = join(baseDir ?? DEFAULT_BASE_DIR, SESSIONS_DIR);
		if (!existsSync(dir)) return [];
		try {
			return readdirSync(dir)
				.filter(d => !d.startsWith("."))
				.map(id => {
					const metaPath = join(dir, id, META_FILE);
					const meta = existsSync(metaPath)
						? (JSON.parse(readFileSync(metaPath, "utf8")) as SessionMeta)
						: null;
					return {
						id,
						name: meta?.name,
						messageCount: meta?.messageCount ?? 0,
						lastActivity: meta?.lastActivity ?? 0,
					};
				});
		} catch {
			return [];
		}
	}
}

export class SessionManager {
	constructor(private readonly baseDir?: string) {}

	listSessions(): Array<{
		id: string;
		name?: string;
		messageCount: number;
		lastActivity: number;
	}> {
		return Session.listSessions(this.baseDir);
	}

	private allMeta(): SessionMeta[] {
		const dir = join(this.baseDir ?? DEFAULT_BASE_DIR, SESSIONS_DIR);
		if (!existsSync(dir)) return [];
		return readdirSync(dir)
			.filter(id => !id.startsWith("."))
			.map(id => {
				const metaPath = join(dir, id, META_FILE);
				if (!existsSync(metaPath)) return null;
				try {
					return JSON.parse(readFileSync(metaPath, "utf8")) as SessionMeta;
				} catch {
					return null;
				}
			})
			.filter((meta): meta is SessionMeta => meta !== null);
	}

	private previewFor(sessionId: string): string {
		const session = this.getSession(sessionId);
		if (!session) return "";
		const firstUserMessage = session
			.load()
			.find(m => m.role === "user");
		const content = firstUserMessage?.content;
		return typeof content === "string" ? content : "";
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

	/** Open an existing session by ID. Returns null if not found. */
	getSession(sessionId: string): Session | null {
		const dir = join(this.baseDir ?? DEFAULT_BASE_DIR, SESSIONS_DIR);
		const metaPath = join(dir, sessionId, META_FILE);
		if (!existsSync(metaPath)) return null;
		return new Session(sessionId, { baseDir: this.baseDir });
	}

	/** Create a new session with a unique ID, scoped to a working directory. */
	createSession(cwd: string, name?: string): Session {
		const sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
		const session = new Session(sessionId, { baseDir: this.baseDir, cwd });
		if (name) session.setName(name);
		return session;
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
}
