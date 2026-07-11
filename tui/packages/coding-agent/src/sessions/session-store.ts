// ── Session Store ──────────────────────────────────────────────────────────────
// Persistent session storage using bun:sqlite.
// One SQLite DB per project workspace in ~/.logician/tui/sessions/.
// Auto-save on turn_end; crash-safe via WAL mode.

import { Database } from "bun:sqlite";
import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";
import type { Turn } from "./transcript.ts";
import { markPathIgnoredByCloudSync } from "@logician/agent-core/tools/shared/path-utils.ts";

const SCHEMA_VERSION = 2;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SessionRow {
	id: string;
	title: string;
	name: string | null;
	cwd: string;
	model: string;
	created_at: string;
	updated_at: string;
}

export interface LabelRow {
	id: string;
	session_id: string;
	turn_id: string | null;
	label: string;
	note: string | null;
	created_at: string;
}

export interface SettingChangeRow {
	id: string;
	session_id: string;
	key: string;
	value: string | null;
	previous_value: string | null;
	created_at: string;
}

export interface TurnRow {
	id: string;
	session_id: string;
	user_content: string;
	assistant_content: string | null;
	thinking_content: string | null;
	tool_executions: string | null; // JSON array of tool names
	turn_number: number;
	created_at: string;
	is_complete: number;
}

export interface SessionSummary {
	id: string;
	title: string;
	name: string | null;
	preview: string;
	lastUpdated: string;
	messageCount: number;
	model: string;
	cwd: string;
	created_at: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const SESSIONS_SUBDIR = "tui/sessions";

/** Hash a project directory to a stable 8-char prefix for DB filename. */
function hashProjectDir(projectDir: string): string {
	return createHash("sha256")
		.update(projectDir.toLowerCase())
		.digest("hex")
		.slice(0, 8);
}

function resolveSessionDbPath(projectDir: string): string {
	const storageRoot = process.env.XDG_DATA_HOME
		? join(process.env.XDG_DATA_HOME, SESSIONS_SUBDIR)
		: join(homedir(), ".local", "share", SESSIONS_SUBDIR);
	return join(storageRoot, `${hashProjectDir(projectDir)}.db`);
}

/** Truncate content to a short preview for session listings. */
function toPreview(text: string | null, maxLen = 80): string {
	if (!text) return "";
	const cleaned = text.replace(/\n+/g, " ").trim();
	return cleaned.length > maxLen ? cleaned.slice(0, maxLen) + "…" : cleaned;
}

/** Serialize tool executions to JSON string. */
function serializeTools(
	toolExecs: Array<{
		tool: string;
		tool_name: string;
		args?: Record<string, unknown>;
		isError: boolean;
		isComplete: boolean;
	}>,
): string {
	return JSON.stringify(
		toolExecs.map((t) => ({
			tool: t.tool,
			tool_name: t.tool_name,
			isError: t.isError,
			isComplete: t.isComplete,
		})),
	);
}

// ── SessionStore ──────────────────────────────────────────────────────────────

export class SessionStore {
	private db: Database;
	private statements: Record<
		string,
		{
			run: (...args: unknown[]) => unknown;
			get: (...args: unknown[]) => unknown;
			all: (...args: unknown[]) => unknown;
		}
	> = {};
	private projectDir: string;
	private currentSessionId: string | null = null;

	constructor(projectDir: string) {
		this.projectDir = projectDir;
		const dbPath = resolveSessionDbPath(projectDir);

		// Ensure directory exists
		const dir = dirname(dbPath);
		if (!existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
		}
		markPathIgnoredByCloudSync(dir);

		this.db = new Database(dbPath);
		this.db.exec("PRAGMA journal_mode = WAL");
		this.db.exec("PRAGMA synchronous = normal");
		this.db.exec("PRAGMA foreign_keys = true");

		this.initSchema();
		this.runMigrations();
		this.prepareStatements();
	}

	// ── Schema ────────────────────────────────────────────────────────────

	private runMigrations(): void {
		const current = (
			this.db.prepare("PRAGMA user_version").get() as { user_version: number }
		).user_version;
		if (current >= SCHEMA_VERSION) return;

		this.db.exec("BEGIN");
		try {
			// v0 -> v1: named sessions and labels/bookmarks.
			if (current < 1) {
				if (!this.hasColumn("sessions", "name")) {
					this.db.exec("ALTER TABLE sessions ADD COLUMN name TEXT");
				}
				this.db.exec(`
					CREATE TABLE IF NOT EXISTS labels (
						id TEXT PRIMARY KEY,
						session_id TEXT NOT NULL,
						turn_id TEXT,
						label TEXT NOT NULL,
						note TEXT,
						created_at TEXT NOT NULL DEFAULT (datetime('now')),
						FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
						FOREIGN KEY (turn_id) REFERENCES turns(id) ON DELETE SET NULL
					);
					CREATE INDEX IF NOT EXISTS idx_labels_session ON labels(session_id);
					PRAGMA user_version = 1;
				`);
			}

			// v1 -> v2: settings change log for model/thinking persistence.
			if (current < 2) {
				this.db.exec(`
					CREATE TABLE IF NOT EXISTS settings_changes (
						id TEXT PRIMARY KEY,
						session_id TEXT NOT NULL,
						key TEXT NOT NULL,
						value TEXT,
						previous_value TEXT,
						created_at TEXT NOT NULL DEFAULT (datetime('now')),
						FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
					);
					CREATE INDEX IF NOT EXISTS idx_settings_changes_session
						ON settings_changes(session_id, created_at);
					PRAGMA user_version = 2;
				`);
			}
			this.db.exec("COMMIT");
		} catch (error) {
			this.db.exec("ROLLBACK");
			throw error;
		}
	}

	private initSchema(): void {
		this.db.exec(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'Untitled Session',
        name TEXT,
        cwd TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
      );

      CREATE TABLE IF NOT EXISTS turns (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        user_content TEXT NOT NULL DEFAULT '',
        assistant_content TEXT,
        thinking_content TEXT,
        tool_executions TEXT,
        turn_number INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        is_complete INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
      );

      CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
      CREATE INDEX IF NOT EXISTS idx_turns_number ON turns(session_id, turn_number);
      CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);

      CREATE TABLE IF NOT EXISTS labels (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        turn_id TEXT,
        label TEXT NOT NULL,
        note TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (turn_id) REFERENCES turns(id) ON DELETE SET NULL
      );
      CREATE INDEX IF NOT EXISTS idx_labels_session ON labels(session_id);

      CREATE TABLE IF NOT EXISTS settings_changes (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT,
        previous_value TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
      );
      CREATE INDEX IF NOT EXISTS idx_settings_changes_session
        ON settings_changes(session_id, created_at);
    `);
	}

	private hasColumn(table: string, column: string): boolean {
		const rows = this.db.prepare(`PRAGMA table_info(${table})`).all() as Array<{
			name: string;
		}>;
		return rows.some((row) => row.name === column);
	}

	private prepareStatements(): void {
		const p = (key: string, sql: string): void => {
			this.statements[key] = this.db.prepare(sql) as any;
		};

		// ── Sessions ──
		p(
			"createSession",
			`
      INSERT INTO sessions (id, title, cwd, model) VALUES (?, ?, ?, ?)
    `,
		);
		p(
			"updateSessionTitle",
			`
      UPDATE sessions SET title = ?, updated_at = datetime('now') WHERE id = ?
    `,
		);
		p(
			"getSession",
			`
      SELECT * FROM sessions WHERE id = ?
    `,
		);
		p(
			"listSessions",
			`
      SELECT s.*,
        (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id) AS turn_count,
        (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id AND t.user_content != '') AS msg_count
      FROM sessions s
      ORDER BY s.updated_at DESC
    `,
		);
		p("deleteSession", "DELETE FROM sessions WHERE id = ?");
		p(
			"getOldestSession",
			"SELECT id FROM sessions ORDER BY created_at ASC LIMIT 1",
		);
		p("countSessions", "SELECT COUNT(*) AS cnt FROM sessions");
		p(
			"setSessionName",
			"UPDATE sessions SET name = ?, updated_at = datetime('now') WHERE id = ?",
		);
		p("findSessionByName", "SELECT * FROM sessions WHERE name = ? LIMIT 1");

		// ── Turns ──
		p(
			"insertTurn",
			`
      INSERT INTO turns (id, session_id, user_content, assistant_content,
                         thinking_content, tool_executions, turn_number, is_complete)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `,
		);
		p(
			"updateTurn",
			`
      UPDATE turns SET assistant_content = ?, thinking_content = ?,
                       tool_executions = ?, is_complete = ?
      WHERE id = ? AND session_id = ?
    `,
		);
		p(
			"getNextTurnNumber",
			`
      SELECT COALESCE(MAX(turn_number), 0) + 1 AS next_num
      FROM turns WHERE session_id = ?
    `,
		);
		p(
			"getTurns",
			`
      SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number ASC
    `,
		);
		p(
			"getTurnCount",
			`
      SELECT COUNT(*) AS cnt FROM turns WHERE session_id = ?
    `,
		);
		p("deleteTurns", "DELETE FROM turns WHERE session_id = ?");
		p(
			"getTurnPreview",
			`
      SELECT user_content, assistant_content
      FROM turns WHERE session_id = ?
      ORDER BY turn_number DESC
      LIMIT 20
    `,
		);

		// ── Labels ──
		p(
			"addLabel",
			"INSERT INTO labels (id, session_id, turn_id, label, note) VALUES (?, ?, ?, ?, ?)",
		);
		p(
			"listLabels",
			"SELECT * FROM labels WHERE session_id = ? ORDER BY created_at ASC",
		);
		p("deleteLabel", "DELETE FROM labels WHERE id = ?");

		// ── Settings changes ──
		p(
			"addSettingChange",
			"INSERT INTO settings_changes (id, session_id, key, value, previous_value) VALUES (?, ?, ?, ?, ?)",
		);
		p(
			"listSettingChanges",
			"SELECT * FROM settings_changes WHERE session_id = ? ORDER BY created_at ASC",
		);
	}

	// ── Session CRUD ─────────────────────────────────────────────────────

	/** Create a new session and set it as current. */
	createSession(opts?: { title?: string; model?: string }): string {
		const id = `sess_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
		this.statements.createSession.run(
			id,
			opts?.title || "Untitled Session",
			this.projectDir,
			opts?.model || "",
		);
		this.currentSessionId = id;
		return id;
	}

	/** Get session by ID. */
	getSession(id: string): SessionRow | null {
		return this.statements.getSession.get(id) as SessionRow | null;
	}

	/** List all sessions with metadata. */
	listSessions(): SessionSummary[] {
		const rows = this.statements.listSessions.all() as Array<
			SessionRow & { turn_count: number; msg_count: number }
		>;

		return rows.map((row) => {
			const previews = this.statements["getTurnPreview"].all(row.id) as Array<{
				user_content: string;
				assistant_content: string | null;
			}>;
			const previewText = previews
				.map(
					(p) =>
						toPreview(p.user_content, 60) || toPreview(p.assistant_content, 60),
				)
				.filter(Boolean)
				.slice(0, 2)
				.join(" · ");

			return {
				id: row.id,
				title: row.title,
				name: row.name,
				preview: previewText || "Empty session",
				lastUpdated: row.updated_at,
				messageCount: row.msg_count,
				model: row.model,
				cwd: row.cwd,
				created_at: row.created_at,
			};
		});
	}

	/** Rename a session. */
	renameSession(id: string, title: string): boolean {
		const result = this.statements.updateSessionTitle.run(title, id);
		return (result as { changes: number }).changes > 0;
	}

	/** Delete a session. */
	deleteSession(id: string): boolean {
		const result = this.statements.deleteSession.run(id);
		if (
			(result as { changes: number }).changes > 0 &&
			this.currentSessionId === id
		) {
			this.currentSessionId = null;
		}
		return (result as { changes: number }).changes > 0;
	}

	/** Get the current session ID. */
	getCurrentSessionId(): string | null {
		return this.currentSessionId;
	}

	/** Set the current session. */
	setCurrentSession(id: string): void {
		if (this.getSession(id)) {
			this.currentSessionId = id;
		}
	}

	// ── Turn Persistence ─────────────────────────────────────────────────

	/** Save a completed turn to the current session. */
	saveTurn(turn: Turn): void {
		if (!this.currentSessionId) return;

		const assistant = turn.assistantMessage;

		// Extract thinking and content from chunks
		let thinkingContent: string | null = null;
		let assistantContent: string | null = null;
		const toolExecs: Array<{
			tool: string;
			tool_name: string;
			args?: Record<string, unknown>;
			isError: boolean;
			isComplete: boolean;
		}> = [];

		if (assistant) {
			for (const chunk of assistant.chunks) {
				if (chunk.type === "thinking" && chunk.contentText) {
					thinkingContent =
						(thinkingContent || "") + chunk.contentText + "\n\n";
				} else if (chunk.type === "content" && chunk.contentText) {
					assistantContent = (assistantContent || "") + chunk.contentText;
				} else if (chunk.type === "tool" && chunk.tool) {
					toolExecs.push(chunk.tool);
				}
			}
		}

		const result = this.statements.getNextTurnNumber.get(
			this.currentSessionId,
		) as { next_num: number };

		this.statements.insertTurn.run(
			turn.id,
			this.currentSessionId,
			turn.userMessage.content,
			assistantContent,
			thinkingContent?.trim() || null,
			toolExecs.length > 0 ? serializeTools(toolExecs) : null,
			result.next_num,
			turn.isComplete ? 1 : 0,
		);

		// Update session timestamp
		this.db
			.prepare("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?")
			.run(this.currentSessionId);
	}

	/** Update an existing turn (for partial save during streaming). */
	updateTurn(
		turnId: string,
		assistantContent: string,
		thinkingContent: string,
		toolExecs: unknown[],
	): void {
		if (!this.currentSessionId) return;

		this.statements.updateTurn.run(
			assistantContent || null,
			thinkingContent || null,
			toolExecs.length > 0 ? JSON.stringify(toolExecs) : null,
			0, // not complete yet
			turnId,
			this.currentSessionId,
		);
	}

	/** Load all turns for a session, returning them in Transcript-compatible format. */
	loadTurns(sessionId: string): Turn[] {
		const rows = this.statements.getTurns.all(sessionId) as TurnRow[];

		return rows.map((row) => {
			const chunks = [];

			// Reconstruct assistant chunks from stored data
			if (row.assistant_content) {
				try {
					// Try parsing as structured chunks first
					const structured = JSON.parse(row.assistant_content);
					if (Array.isArray(structured)) {
						for (const s of structured) {
							if (s.type === "content") {
								chunks.push({
									type: "content" as const,
									contentText: s.text || "",
								});
							}
						}
					} else {
						// Stored as plain text (legacy format)
						chunks.push({
							type: "content" as const,
							contentText: row.assistant_content,
						});
					}
				} catch {
					// Not JSON — store as plain content
					chunks.push({
						type: "content" as const,
						contentText: row.assistant_content,
					});
				}
			}

			if (row.thinking_content) {
				chunks.push({
					type: "thinking" as const,
					contentText: row.thinking_content,
				});
			}

			// Assign sequence numbers to chunks
			const numberedChunks: import("./transcript.ts").AssistantChunk[] =
				chunks.map((c, idx) => ({
					...c,
					seq: idx,
					isComplete: !!row.is_complete,
					type: c.type,
				}));

			const turn: Turn = {
				id: row.id,
				userMessage: { type: "user", content: row.user_content },
				assistantMessage:
					numberedChunks.length > 0
						? {
								type: "assistant",
								chunks: numberedChunks,
								isComplete: !!row.is_complete,
							}
						: null,
				isComplete: !!row.is_complete,
			};

			return turn;
		});
	}

	/** Search sessions by text in user messages. */
	searchSessions(query: string, limit = 20): SessionSummary[] {
		const escaped = query.replace(/[%_]/g, "\\$&");
		const rows = this.db
			.prepare(`
        SELECT s.*,
          (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id) AS turn_count,
          (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id AND t.user_content != '') AS msg_count
        FROM sessions s
        WHERE s.id IN (
          SELECT DISTINCT session_id FROM turns
          WHERE user_content LIKE '%' || ? || '%' ESCAPE '\\'
             OR assistant_content LIKE '%' || ? || '%' ESCAPE '\\'
        )
        ORDER BY s.updated_at DESC
        LIMIT ?
      `)
			.all(escaped, escaped, limit) as unknown as Array<
			SessionRow & { turn_count: number; msg_count: number }
		>;

		return rows.map((row) => ({
			id: row.id,
			title: row.title,
			name: row.name,
			preview: row.title,
			lastUpdated: row.updated_at,
			messageCount: row.msg_count,
			model: row.model,
			cwd: row.cwd,
			created_at: row.created_at,
		}));
	}

	// ── Stats ─────────────────────────────────────────────────────────────

	getStats(): {
		totalSessions: number;
		totalTurns: number;
		latestSession: string | null;
	} {
		const totalSessions = (
			this.statements.countSessions.get() as { cnt: number }
		).cnt;
		const totalTurns = (
			this.db.prepare("SELECT COUNT(*) AS cnt FROM turns").get() as {
				cnt: number;
			}
		).cnt;
		const latestRow = this.statements.getOldestSession.get();
		const latestId = latestRow ? (latestRow as { id: string }).id : null;

		return {
			totalSessions,
			totalTurns,
			latestSession: latestId,
		};
	}

	// ── Cleanup ───────────────────────────────────────────────────────────

	/** Keep only the N most recent sessions, delete the rest. */
	keepRecent(maxSessions = 100): number {
		const toDelete = this.statements.listSessions.all() as SessionRow[];
		if (toDelete.length <= maxSessions) return 0;

		let deleted = 0;
		for (let i = maxSessions; i < toDelete.length; i++) {
			const id = toDelete[i].id;
			if (id !== this.currentSessionId) {
				this.statements.deleteSession.run(id);
				deleted++;
			}
		}
		return deleted;
	}

	// ── Named Sessions ────────────────────────────────────────────────────

	/** Set a short human name on the current or specified session. */
	setSessionName(name: string, sessionId?: string): boolean {
		const id = sessionId ?? this.currentSessionId;
		if (!id) return false;
		const result = this.statements.setSessionName.run(name.trim() || null, id);
		return (result as { changes: number }).changes > 0;
	}

	/** Look up a session by name (exact match). */
	findSessionByName(name: string): SessionRow | null {
		return this.statements.findSessionByName.get(name) as SessionRow | null;
	}

	// ── Labels / Bookmarks ────────────────────────────────────────────────

	/** Add a label/bookmark to the current session, optionally tied to a turn. */
	addLabel(label: string, opts?: { turnId?: string; note?: string; sessionId?: string }): string {
		const sessionId = opts?.sessionId ?? this.currentSessionId;
		if (!sessionId) throw new Error("No active session");
		const id = `lbl_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`;
		this.statements.addLabel.run(
			id,
			sessionId,
			opts?.turnId ?? null,
			label,
			opts?.note ?? null,
		);
		return id;
	}

	/** List all labels for a session. */
	listLabels(sessionId?: string): LabelRow[] {
		const id = sessionId ?? this.currentSessionId;
		if (!id) return [];
		return this.statements.listLabels.all(id) as LabelRow[];
	}

	/** Delete a label by ID. */
	deleteLabel(labelId: string): boolean {
		const result = this.statements.deleteLabel.run(labelId);
		return (result as { changes: number }).changes > 0;
	}

	// ── Settings Changes ──────────────────────────────────────────────────

	recordSettingChange(
		key: string,
		value: string | null,
		previousValue?: string | null,
		sessionId?: string,
	): string | null {
		const id = sessionId ?? this.currentSessionId;
		if (!id) return null;
		const changeId = `chg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`;
		this.statements.addSettingChange.run(
			changeId,
			id,
			key,
			value,
			previousValue ?? null,
		);
		this.db
			.prepare("UPDATE sessions SET updated_at = datetime('now') WHERE id = ?")
			.run(id);
		return changeId;
	}

	listSettingChanges(sessionId?: string): SettingChangeRow[] {
		const id = sessionId ?? this.currentSessionId;
		if (!id) return [];
		return this.statements.listSettingChanges.all(id) as SettingChangeRow[];
	}

	// ── Close ─────────────────────────────────────────────────────────────

	close(): void {
		this.db.close();
	}
}
