// ── TUI session service ──────────────────────────────────────────────────
// Backs the session browser, resume, rename, and delete over agent-core's
// JSONL Session/SessionRegistry tree — replaces the old SQLite SessionStore.
// One Session per conversation, cwd-scoped listing, Turn[] persisted as
// opaque entries via turn-entries.ts.

import { SessionRegistry, type SessionStore } from "@logician/log-core/runtime";
import type { Turn } from "../transcript/transcript.ts";
import {
	loadTurns as loadTurnsFromSession,
	saveTurn as saveTurnToSession,
} from "../transcript/turn-entries.ts";

export interface TuiSessionSummary {
	id: string;
	name: string;
	preview: string;
	lastUpdated: string;
	messageCount: number;
}

export interface TuiSessionTreeNode {
	id: string;
	parentId?: string;
	label: string;
	preview: string;
	timestamp: number;
	isCurrent: boolean;
	hasChildren: boolean;
}

/** Derive a compact topic label from the first completed user/agent exchange. */
export function isGeneratedSessionTitle(name: string): boolean {
	return name === "Untitled Session" || name === "New Session";
}

/** Derive a compact topic label from the first completed user/agent exchange. */
export function inferSessionTitle(
	content: string,
	agentResponse = "",
	maxLength = 60,
): string | null {
	if (!content?.trim()) return null;
	if (/^(?:hi|hello|hey|thanks|thank you|ok|okay)[!. ]*$/i.test(content.trim()))
		return null;
	const userTopic = extractSessionTopic(content);
	const responseTopic = extractSessionTopic(agentResponse, true);
	const userWords = userTopic?.split(/\s+/) || [];
	const userIsVague =
		!userTopic ||
		userWords.length < 4 ||
		/^(?:do|fix|change|update|add|remove|clean|check|try|yes|no)\s+(?:it|this|that|these|those)\b/i.test(
			userTopic,
		);
	const topic =
		userIsVague && responseTopic ? responseTopic : userTopic || responseTopic;
	if (!topic) return null;

	const firstClause =
		topic.split(/(?<=[.!?])\s+|\s+[—–]\s+|;\s+/)[0]?.trim() || topic;
	const words = firstClause.split(" ");
	let title = words.slice(0, 10).join(" ");
	if (words.length > 10 || title.length > maxLength) {
		title = `${title.slice(0, maxLength - 1).trimEnd()}…`;
	}
	return title.charAt(0).toUpperCase() + title.slice(1);
}

function extractSessionTopic(
	content: string | null | undefined,
	fromAgent = false,
): string | null {
	if (!content) return null;
	const lines = content
		.split(/\r?\n/)
		.map(line => line.trim())
		.filter(line => line.length > 0)
		.filter(
			line =>
				!/^#{1,6}\s*(?:files? mentioned(?: by the user)?|my request for codex)\s*:?$/i.test(
					line,
				),
		)
		.filter(line => !/^(?:file|attachment):\s*[/\\]/i.test(line));
	let topic = lines.find(line => !/^[-*]\s*[/\\]/.test(line)) || "";
	topic = topic
		.replace(/^#{1,6}\s*/, "")
		.replace(/^(?:my request for codex|request)\s*:\s*/i, "")
		.replace(/^(?:please\s+)?(?:can|could|would)\s+you\s+/i, "")
		.replace(/^(?:please\s+)?(?:we|i)\s+(?:need|want)\s+(?:you\s+)?to\s+/i, "")
		.replace(
			fromAgent
				? /^(?:i(?:'ve| have)?\s+)?(?:implemented|fixed|updated|added|removed|changed|completed)\s+/i
				: /$^/,
			"",
		)
		.replace(/^please\s+/i, "")
		.replace(/\s+/g, " ")
		.trim();

	if (
		!topic ||
		/^(?:hi|hello|hey|thanks|thank you|ok|okay)[!. ]*$/i.test(topic)
	) {
		return null;
	}
	return topic;
}

export class TuiSessionService {
	private manager: SessionRegistry;
	private cwd: string;
	private currentSessionId: string | null = null;
	// One Session instance per id, reused across calls. Session's own
	// activeLeafId is cached in-memory on construction — a second instance
	// for the same id would go stale the moment either one appends an entry
	// (branching, compaction, turn saves), so every accessor below must
	// route through this cache rather than ask SessionRegistry fresh each time.
	private openSessions = new Map<string, SessionStore>();

	constructor(cwd: string) {
		this.cwd = cwd;
		this.manager = new SessionRegistry();
	}

	getCwd(): string {
		return this.cwd;
	}

	private open(id: string): SessionStore | null {
		const cached = this.openSessions.get(id);
		if (cached) return cached;
		const session = this.manager.getSession(id);
		if (session) this.openSessions.set(id, session);
		return session;
	}

	/** Create a new session for this project. Returns its id. */
	createSession(name = "New Session"): string {
		const session = this.manager.createSession(this.cwd, name);
		const id = session.getMeta().id;
		this.openSessions.set(id, session);
		this.currentSessionId = id;
		return id;
	}

	getSession(id: string): TuiSessionSummary | null {
		const session = this.open(id);
		if (!session) return null;
		const meta = session.getMeta();
		const turns = loadTurnsFromSession(session);
		return {
			id: meta.id,
			name: meta.name ?? "Untitled Session",
			preview: firstUserPreview(turns),
			lastUpdated: new Date(meta.lastActivity).toISOString(),
			messageCount: turns.length,
		};
	}

	/**
	 * Get the underlying Session instance for a conversation, so a caller
	 * (e.g. AgentHarness.attachSession) can use it as the durable
	 * branch/compaction/model journal for the same conversation this service
	 * already persists Turns to. Always the same cached instance this service
	 * itself reads/writes through — never a second Session for the same id.
	 */
	getRawSession(id: string): SessionStore | null {
		return this.open(id);
	}

	/** List sessions for this project, most recently active first. */
	listSessions(): TuiSessionSummary[] {
		return this.manager.listSessionInfos(this.cwd).map(info => ({
			id: info.id,
			name: info.name ?? "Untitled Session",
			preview: info.preview || "(no messages)",
			lastUpdated: new Date(info.lastActivity).toISOString(),
			messageCount: info.messageCount,
		}));
	}

	renameSession(id: string, name: string): boolean {
		const session = this.open(id);
		if (!session) return false;
		session.setName(name);
		return true;
	}

	deleteSession(id: string): boolean {
		this.openSessions.delete(id);
		return this.manager.deleteSession(id);
	}

	getCurrentSessionId(): string | null {
		return this.currentSessionId;
	}

	setCurrentSession(id: string): void {
		this.currentSessionId = id;
	}

	/** Persist a completed turn to a session. */
	saveTurn(sessionId: string, turn: Turn): void {
		const session = this.open(sessionId);
		if (!session) return;
		saveTurnToSession(session, turn);
	}

	/** Load all turns for a session, in conversation order. */
	loadTurns(sessionId: string): Turn[] {
		const session = this.open(sessionId);
		if (!session) return [];
		return loadTurnsFromSession(session);
	}

	/** Return persisted TUI turns as a flat list (depth-first branch order). */
	getTurnTree(sessionId: string): TuiSessionTreeNode[] {
		const session = this.open(sessionId);
		if (!session) return [];
		const entries = session.loadEntries();
		const byId = new Map(entries.map(entry => [entry.id, entry]));
		const turns = entries.filter(
			entry => entry.type === "custom" && entry.customType === "tui_turn",
		);
		// Narrow to CustomSessionEntry so `.data` is accessible
		const turnsTyped = turns as Array<{
			type: "custom";
			id: string;
			parentId?: string;
			timestamp: number;
			customType: string;
			data: Turn;
		}>;
		const turnIds = new Set(turnsTyped.map(entry => entry.id));
		const nearestTurn = (start?: string): string | undefined => {
			let id = start;
			while (id) {
				if (turnIds.has(id)) return id;
				id = byId.get(id)?.parentId;
			}
			return undefined;
		};
		const children = new Map<string | undefined, typeof turnsTyped>();
		for (const entry of turnsTyped) {
			const parentId = nearestTurn(entry.parentId);
			const siblings = children.get(parentId) ?? [];
			siblings.push(entry);
			children.set(parentId, siblings);
		}
		const currentTurnId = nearestTurn(session.getLeafEntryId());
		const result: TuiSessionTreeNode[] = [];
		const visit = (entry: (typeof turnsTyped)[number]): void => {
			const turn = entry.data;
			const content = turn.userMessage?.content?.trim() || "(empty turn)";
			const childEntries = children.get(entry.id) ?? [];
			result.push({
				id: entry.id,
				parentId: nearestTurn(entry.parentId),
				label: content.split(/\r?\n/, 1)[0]?.slice(0, 72) || "(empty turn)",
				preview: content,
				timestamp: entry.timestamp,
				isCurrent: entry.id === currentTurnId,
				hasChildren: childEntries.length > 0,
			});
			for (const child of childEntries) visit(child);
		};
		for (const root of children.get(undefined) ?? []) visit(root);
		return result;
	}

	/** Select a persisted turn as the active leaf and return its transcript path. */
	checkoutTurn(sessionId: string, entryId: string): Turn[] {
		const session = this.open(sessionId);
		if (!session) return [];
		session.checkout(entryId);
		return loadTurnsFromSession(session);
	}

	/** Delete all but the `keep` most recently active sessions. Returns count deleted. */
	keepRecent(keep = 100): number {
		const deleted = this.manager.keepRecent(this.cwd, keep);
		this.openSessions.clear();
		return deleted;
	}
}

function firstUserPreview(turns: Turn[]): string {
	for (const turn of turns) {
		if (turn.userMessage?.content) return turn.userMessage.content;
	}
	return "";
}
