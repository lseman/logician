// ── @logician/log-memory — SQLite-backed Store ───────────────────────────────────
// Implements the observation→compression→memory pipeline. Table creation and
// migrations live in schema.ts; CRUD and every other capability are split by
// concern into sibling files (sessions.ts, observations.ts, memories.ts, ...),
// each taking `db` and a `getWorkspace: () => string` accessor explicitly
// instead of closing over them — the two pieces of state every call needs.
// This factory's only remaining job is constructing that state and wiring
// each module's functions into the public MemoryStore surface below.

import { Database } from "bun:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { MemoryStore } from "../types.js";
import { getAccessStats, trackAccess } from "./access-tracker.ts";
import { autoForget } from "./auto-forget.ts";
import { consolidate } from "./consolidation.ts";
import {
	getContext,
	getShadowPolicy,
	listOutcomeReceipts,
	listRetrievalTraces,
	recordOutcomeReceipt,
	retrieve,
} from "./context-injection.ts";
import { dedupCheck, dedupRecord } from "./dedup.ts";
import {
	hasEmbedding,
	searchEmbeddings,
	upsertEmbedding,
} from "./embeddings.ts";
import { exportData, importData } from "./export-import.ts";
import {
	claimExtractionJob,
	completeExtractionJob,
	enqueueExtractionJob,
	failExtractionJob,
	listExtractionJobs,
	renewExtractionJob,
} from "./extraction-queue.ts";
import {
	getFileContext,
	getFilesContext,
	rebuildFileIndex,
} from "./file-context-index.ts";
import {
	clearMemories,
	create,
	deleteEntry,
	get,
	getAny,
	list,
	recall,
	update,
} from "./memories.ts";
import {
	evolve,
	getRelatedMemories,
	getRelations,
	relate,
	removeRelation,
} from "./memory-relations.ts";
import { normalizeWorkspacePath } from "./module-helpers.ts";
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
} from "./observations.ts";
import {
	computeRetentionScore,
	listByRetentionScore,
	rescoreAll,
} from "./retention-scoring.ts";
import { initMemoryStoreSchema } from "./schema.ts";
import {
	clearSessions,
	createSession,
	discardEmptySession,
	getSession,
	listSessions,
	updateSession,
} from "./sessions.ts";
import { slidingWindowCap } from "./sliding-window.ts";
import {
	autoTierMemories,
	getWorkingMemoryTier,
	setWorkingMemoryTier,
} from "./working-memory-tiers.ts";

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
	// Owns currentSessionId/currentWorkspace mutation directly — every other
	// module reads currentWorkspace live through the getWorkspace accessor
	// passed to it below, rather than each holding its own copy.

	let currentSessionId: string | null = null;
	function setCurrentSessionId(id: string): void {
		currentSessionId = id;
		// Ensure session exists
		if (!getSession(db, id)) {
			createSession(db, getWorkspace, id, { project: "" });
		}
		// Sync workspace from session
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
		enqueueExtractionJob: (sessionId, workspace, payload) =>
			enqueueExtractionJob(db, sessionId, workspace, payload),
		claimExtractionJob: leaseMs =>
			claimExtractionJob(db, getWorkspace, leaseMs),
		renewExtractionJob: (id, leaseMs) => renewExtractionJob(db, id, leaseMs),
		completeExtractionJob: id => completeExtractionJob(db, id),
		failExtractionJob: (id, error, retryDelayMs) =>
			failExtractionJob(db, id, error, retryDelayMs),
		listExtractionJobs: status => listExtractionJobs(db, getWorkspace, status),
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
