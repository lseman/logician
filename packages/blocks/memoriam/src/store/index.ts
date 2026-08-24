// ── Memoriam — SQLite-backed Memory Store ─────────────────────────────────────
// Implements the observation→compression→memory pipeline. Table creation and
// migrations live in schema.ts; CRUD and every other capability are split by
// concern into sibling files (sessions.ts, observations.ts, memories.ts, ...),
// each taking `db` and a `getWorkspace: () => string` accessor explicitly
// instead of closing over them — the two pieces of state every call needs.

import { Database } from "bun:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { MemoryStore } from "../types.ts";
import { getAccessStats, trackAccess } from "./access-tracker.js";
import { exportData, importData } from "./export-import.js";
import {
	clearMemories,
	create,
	deleteEntry,
	get,
	getAny,
	list,
	recall,
	update,
} from "./models/memories.js";
import {
	clearObservations,
	expandEntries,
	getObservation,
	listClaims,
	listObservations,
	listRecentObservations,
	observe,
	promoteClaim,
	searchObservations,
} from "./models/observations.js";
import {
	clearSessions,
	createSession,
	discardEmptySession,
	getSession,
	listSessions,
	updateSession,
} from "./models/sessions.js";
import { normalizeWorkspacePath } from "./module-helpers.js";
import { autoForget } from "./policy/auto-forget.js";
import { consolidate } from "./policy/consolidation.js";
import { dedupCheck, dedupRecord } from "./policy/dedup.js";
import {
	evolve,
	getRelatedMemories,
	getRelations,
	relate,
	removeRelation,
} from "./policy/memory-relations.js";
import {
	computeRetentionScore,
	listByRetentionScore,
	rescoreAll,
} from "./policy/retention-scoring.js";
import { slidingWindowCap } from "./policy/sliding-window.js";
import {
	autoTierMemories,
	getWorkingMemoryTier,
	setWorkingMemoryTier,
} from "./policy/working-memory-tiers.js";
import {
	getContext,
	getShadowPolicy,
	listOutcomeReceipts,
	listRetrievalTraces,
	recordOutcomeReceipt,
	retrieve,
} from "./retrieval/context-injection.js";
import {
	hasEmbedding,
	searchEmbeddings,
	upsertEmbedding,
} from "./retrieval/embeddings.js";
import {
	getFileContext,
	getFilesContext,
	rebuildFileIndex,
} from "./retrieval/file-context-index.js";
import { initMemoryStoreSchema } from "./schema.js";

export function createMemoryStore(dbPath: string): MemoryStore {
	const resolved = dbPath
		.replace(/^~(?=\/|$)/, process.env.HOME || "")
		.replace(/^~/, process.env.HOME || "");
	mkdirSync(dirname(resolved), { recursive: true });
	const db = new Database(resolved);
	db.exec(
		"PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000; PRAGMA foreign_keys = ON;",
	);
	let currentWorkspace = normalizeWorkspacePath(process.cwd());
	const getWorkspace = () => currentWorkspace;

	initMemoryStoreSchema(db);

	// ── Session ID tracking ────────────────────────────────────────────────
	let currentSessionId: string | null = null;
	function setCurrentSessionId(id: string): void {
		currentSessionId = id;
		if (!getSession(db, id)) {
			createSession(db, getWorkspace, id, { project: "" });
		}
		const session = getSession(db, id);
		if (session?.workspace) {
			currentWorkspace = normalizeWorkspacePath(session.workspace);
		}
	}

	function getCurrentSessionId(): string | null {
		return currentSessionId;
	}

	function setCurrentWorkspace(ws: string): void {
		currentWorkspace = normalizeWorkspacePath(ws);
	}

	function getCurrentWorkspace(): string {
		return currentWorkspace;
	}

	// ── Public API ─────────────────────────────────────────────────────────

	return {
		createSession: (id, data) => createSession(db, getWorkspace, id, data),
		getSession: id => getSession(db, id),
		listSessions: query => listSessions(db, getWorkspace, query),
		updateSession: (id, updates) => updateSession(db, id, updates),
		clearSessions: keepSessionId =>
			clearSessions(db, getWorkspace, keepSessionId),
		discardEmptySession: id => discardEmptySession(db, id),
		observe: (raw, compressed) => observe(db, getWorkspace, raw, compressed),
		getObservation: (id, sessionId) => getObservation(db, id, sessionId),
		listObservations: (sessionId, limit) =>
			listObservations(db, sessionId, limit),
		listRecentObservations: (limit, type) =>
			listRecentObservations(db, getWorkspace, limit, type),
		listClaims: options => listClaims(db, getWorkspace, options),
		promoteClaim: (claimId, evidenceEventIds) =>
			promoteClaim(db, getWorkspace, claimId, evidenceEventIds),
		searchObservations: (query, limit) =>
			searchObservations(db, getWorkspace, query, limit),
		expandEntries: ids => expandEntries(db, getWorkspace, ids),
		clearObservations: () => clearObservations(db, getWorkspace),
		create: (content, options) => create(db, getWorkspace, content, options),
		get: id => get(db, getWorkspace, id),
		getAny: id => getAny(db, getWorkspace, id),
		list: query => list(db, getWorkspace, query),
		remove: id => deleteEntry(db, getWorkspace, id),
		clearMemories: () => clearMemories(db, getWorkspace),
		update: (id, updates) => update(db, getWorkspace, id, updates),
		recall: (query, options) => recall(db, getWorkspace, query, options),
		consolidate: sessionId => consolidate(db, getWorkspace, sessionId),
		getContext: (sessionId, budget, query) =>
			getContext(db, getWorkspace, sessionId, budget, query),
		retrieve: (sessionId, budget, query) =>
			retrieve(db, getWorkspace, sessionId, budget, query),
		listRetrievalTraces: limit => listRetrievalTraces(db, getWorkspace, limit),
		recordOutcomeReceipt: input =>
			recordOutcomeReceipt(db, getWorkspace, input),
		listOutcomeReceipts: limit => listOutcomeReceipts(db, getWorkspace, limit),
		getShadowPolicy: () => getShadowPolicy(db, getWorkspace),
		upsertEmbedding: (id, kind, vector, sessionId, metadata) =>
			upsertEmbedding(db, getWorkspace, id, kind, vector, sessionId, metadata),
		hasEmbedding: (id, metadata) =>
			hasEmbedding(db, getWorkspace, id, metadata),
		searchEmbeddings: (vector, limit) =>
			searchEmbeddings(db, getWorkspace, vector, limit),
		setCurrentSessionId,
		getCurrentSessionId,
		setCurrentWorkspace,
		getCurrentWorkspace,

		// Dedup
		dedupCheck: (sessionId, toolName, toolInput) =>
			dedupCheck(db, sessionId, toolName, toolInput),
		dedupRecord: (sessionId, toolName, toolInput) =>
			dedupRecord(db, sessionId, toolName, toolInput),

		// Sliding Window
		slidingWindowCap: (sessionId, cap) =>
			slidingWindowCap(db, getWorkspace, sessionId, cap),

		// Access Tracker
		trackAccess: entityId => trackAccess(db, getWorkspace, entityId),
		getAccessStats: entityId => getAccessStats(db, getWorkspace, entityId),

		// Working Memory Tiers
		getWorkingMemoryTier: entityId =>
			getWorkingMemoryTier(db, getWorkspace, entityId),
		setWorkingMemoryTier: (entityId, tier) =>
			setWorkingMemoryTier(db, getWorkspace, entityId, tier),
		autoTierMemories: config => autoTierMemories(db, getWorkspace, config),

		// Auto-Forget
		autoForget: (ttlMs, minImportance, maxDeletes) =>
			autoForget(db, getWorkspace, ttlMs, minImportance, maxDeletes),

		// Memory Relations
		relate: (sourceId, targetId, type, confidence) =>
			relate(db, getWorkspace, sourceId, targetId, type, confidence),
		getRelations: memoryId => getRelations(db, getWorkspace, memoryId),
		getRelatedMemories: (memoryId, maxHops, minConfidence) =>
			getRelatedMemories(db, getWorkspace, memoryId, maxHops, minConfidence),
		evolve: (memoryId, newContent, newTitle) =>
			evolve(db, getWorkspace, memoryId, newContent, newTitle),
		removeRelation: relationId => removeRelation(db, getWorkspace, relationId),

		// Retention Scoring
		computeRetentionScore: (id, config) =>
			computeRetentionScore(db, getWorkspace, id, config),
		rescoreAll: config => rescoreAll(db, getWorkspace, config),
		listByRetentionScore: (config, limit) =>
			listByRetentionScore(db, getWorkspace, config, limit),

		// File Context Index
		getFileContext: (file, sessionId) =>
			getFileContext(db, getWorkspace, file, sessionId),
		getFilesContext: (files, sessionId) =>
			getFilesContext(db, getWorkspace, files, sessionId),
		rebuildFileIndex: () => rebuildFileIndex(db, getWorkspace),

		// Export/Import
		exportData: () => exportData(db, getWorkspace),
		importData: data => importData(db, getWorkspace, data),

		close() {
			db.close();
		},
	};
}

export type { MemoryStore } from "../types.ts";
