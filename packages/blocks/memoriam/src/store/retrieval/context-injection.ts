/** Assembles the ranked, budget-fitted memory-context block injected into a
 * turn's prompt: session summary, quoted claims, semantic memory, and (as a
 * fallback when neither has a match) recent episodic observations. Records
 * a retrieval trace per call and folds outcome receipts into the shadow
 * ranking policy. */

import type { Database } from "bun:sqlite";
import {
	initialShadowPolicy,
	learnShadowPolicy,
	policyFeatures,
	shadowDecision,
} from "../../../evolution/shadow-policy.js";
import { predicatesAreValid } from "../../../evolution/validity.js";
import { selectContextCandidates } from "../../retrieval/context-selector.js";
import type {
	CompressedObservation,
	ContextBlock,
	ContextRetrievalQuery,
	Memory,
	MemoryOutcomeReceipt,
	RetrievalTrace,
	ShadowMemoryPolicy,
	WorkingMemoryTier,
} from "../../types.ts";
import { trackAccess } from "../access-tracker.js";
import { safeParseJson, safeParseJsonArray } from "../models/db-helpers.js";
import { searchEmbeddings } from "./embeddings.js";
import { list, rowToMemory } from "../models/memories.js";
import {
	generateId,
	now,
	sanitizeString,
	toFtsAnyQuery,
} from "../module-helpers.js";
import {
	listClaims,
	listRecentObservations,
	rowToObservation,
} from "../models/observations.js";
import { getSession } from "../models/sessions.js";
import { getWorkingMemoryTier } from "../policy/working-memory-tiers.js";

export function getShadowPolicy(
	db: Database,
	getWorkspace: () => string,
): ShadowMemoryPolicy {
	const workspace = getWorkspace();
	const row = db
		.prepare("SELECT * FROM memory_policy_state WHERE workspace = ?")
		.get(workspace) as any;
	if (row)
		return {
			version: row.version,
			mode: row.mode,
			weights: safeParseJsonArray(row.weights).map(Number),
			samples: row.samples,
			updatedAt: row.updated_at,
		};
	const policy = initialShadowPolicy();
	db.prepare(
		`INSERT INTO memory_policy_state
	  (workspace, version, mode, weights, samples, updated_at)
	  VALUES (?, ?, ?, ?, ?, ?)`,
	).run(
		workspace,
		policy.version,
		policy.mode,
		JSON.stringify(policy.weights),
		policy.samples,
		policy.updatedAt,
	);
	return policy;
}

function contextTokens(value: string): Set<string> {
	const stop = new Set([
		"a",
		"an",
		"and",
		"are",
		"as",
		"at",
		"be",
		"by",
		"for",
		"from",
		"in",
		"is",
		"it",
		"of",
		"on",
		"or",
		"the",
		"this",
		"to",
		"with",
	]);
	return new Set(
		(
			value
				.normalize("NFKC")
				.toLowerCase()
				.match(/[\p{L}\p{N}_-]{2,}/gu) || []
		)
			.map(token => token.replace(/(?:ing|ed|es|s)$/i, ""))
			.filter(token => token.length > 1 && !stop.has(token)),
	);
}

export function getContext(
	db: Database,
	getWorkspace: () => string,
	sessionId: string,
	budget: number = 4000,
	query: string | ContextRetrievalQuery = "",
): string {
	const workspace = getWorkspace();
	const retrievalStarted = performance.now();
	const retrieval = typeof query === "string" ? { objective: query } : query;
	const objective = retrieval.objective?.trim() || "";
	const changedFiles = retrieval.changedFiles || [];
	const queryText = [
		objective,
		...changedFiles,
		...(retrieval.recentEvidence || []),
	].join(" ");
	const queryTokens = contextTokens(queryText);
	const fileTokens = new Set(
		changedFiles.flatMap(file => [...contextTokens(file)]),
	);
	const nowMs = Date.now();
	const estimateTokens = (text: string) => Math.ceil(text.length / 3);
	type Candidate = ContextBlock & {
		id: string;
		score: number;
		sourceKey: string;
		similarityText: string;
		memoryId?: string;
		reasons: string[];
	};
	const candidates: Candidate[] = [];
	const episodicFallbackCandidates: Candidate[] = [];
	let claimCandidateCount = 0;
	const briefDescription = (value: string, maxLength: number = 220): string => {
		const normalized = value.replace(/\s+/g, " ").trim();
		if (normalized.length <= maxLength) return normalized;
		const slice = normalized.slice(0, maxLength - 1);
		const boundary = slice.lastIndexOf(" ");
		return `${slice.slice(0, boundary > maxLength * 0.65 ? boundary : undefined).trimEnd()}…`;
	};

	const overlapScore = (text: string): number => {
		if (!queryTokens.size) return 0;
		const candidateTokens = contextTokens(text);
		let overlap = 0;
		for (const token of queryTokens) if (candidateTokens.has(token)) overlap++;
		return (
			overlap / Math.sqrt(Math.max(1, queryTokens.size * candidateTokens.size))
		);
	};
	const fileScore = (files: string[] | undefined, text: string): number => {
		if (!fileTokens.size) return 0;
		const candidateTokens = contextTokens([...(files || []), text].join(" "));
		let matches = 0;
		for (const token of fileTokens) if (candidateTokens.has(token)) matches++;
		return matches / fileTokens.size;
	};
	const recencyScore = (timestamp: string | undefined): number => {
		const ageDays =
			Math.max(0, nowMs - Date.parse(timestamp || "")) / 86_400_000;
		return Number.isFinite(ageDays) ? 1 / (1 + ageDays / 14) : 0;
	};

	// Retention-scored working tier as a small ranking nudge — hot/warm
	// memories (recently relevant, reinforced by access) rank slightly
	// above cold ones, and archived (never-accessed) memories are gently
	// deprioritized rather than excluded outright.
	const tierWeight: Record<WorkingMemoryTier, number> = {
		hot: 1,
		warm: 0.5,
		cold: 0,
		archived: -0.5,
	};
	const tierScore = (memoryId: string): number =>
		tierWeight[getWorkingMemoryTier(db, getWorkspace, memoryId)];

	const session = getSession(db, sessionId);
	if (session?.summary) {
		const content = `# Session Summary\n\n${session.summary}`;
		candidates.push({
			id: `summary:${sessionId}`,
			type: "summary",
			content,
			tokens: estimateTokens(content),
			recency: nowMs,
			score: 8 + overlapScore(session.summary) * 12,
			sourceKey: sessionId,
			similarityText: session.summary,
			reasons: ["session-summary"],
		});
	}

	// Claims are quoted evidence-bearing data, never instructions. Only active,
	// trusted claims enter automatic context; invalidated, superseded and NOOP
	// audit rows remain available through listClaims()/the viewer.
	const claimsForValidation = listClaims(db, getWorkspace, { limit: 500 });
	for (const claim of claimsForValidation) {
		if (
			claim.lifecycle === "durable" &&
			claim.validityPredicates.length > 0 &&
			!predicatesAreValid(claim.workspace, claim.validityPredicates)
		) {
			db.prepare(
				"UPDATE claims SET lifecycle = 'stale' WHERE id = ? AND lifecycle = 'durable'",
			).run(claim.id);
			claim.lifecycle = "stale";
		}
	}
	const retrievableClaims = claimsForValidation.filter(
		claim =>
			claim.status !== "invalidated" &&
			claim.trust !== "untrusted" &&
			claim.operation !== "NOOP" &&
			claim.lifecycle === "durable",
	);
	const claimTextsByObservation = new Map<string, string[]>();
	for (const claim of retrievableClaims) {
		const texts = claimTextsByObservation.get(claim.observationId) || [];
		texts.push(claim.text);
		claimTextsByObservation.set(claim.observationId, texts);
		if (claim.sessionId === sessionId) continue;
		const relevance = overlapScore(claim.text);
		if (queryTokens.size && relevance === 0) continue;
		const evidence = claim.evidenceEventIds.slice(0, 3).join(", ");
		const validity = claim.validTo
			? `${claim.validFrom}–${claim.validTo}`
			: `since ${claim.validFrom}`;
		const content = `- [${claim.id}] Claim (quoted data) · ${claim.status} · confidence ${claim.confidence.toFixed(2)} · ${validity} — ${briefDescription(claim.text)}${evidence ? ` · evidence: ${evidence}` : ""}`;
		candidates.push({
			id: `claim:${claim.id}`,
			type: "claim",
			content,
			tokens: estimateTokens(content),
			recency: Date.parse(claim.transactionTime),
			score:
				relevance * 22 +
				(claim.status === "verified" ? 8 : 3) +
				claim.confidence * 5 +
				recencyScore(claim.transactionTime),
			sourceKey: claim.observationId,
			similarityText: claim.text,
			reasons: [
				claim.status,
				`confidence:${claim.confidence.toFixed(2)}`,
				"lexical-overlap",
			],
		});
		claimCandidateCount++;
	}

	// Generate candidates in SQLite so relevant older knowledge is not hidden
	// behind a fixed recent-item window. RRF makes lexical rank comparable to
	// task, file, recency, and salience signals without score calibration.
	const ftsQuery = toFtsAnyQuery(queryText);
	const lexicalObservationRank = new Map<string, number>();
	const lexicalMemoryRank = new Map<string, number>();
	const semanticObservationRank = new Map<string, number>();
	const semanticMemoryRank = new Map<string, number>();
	const lexicalObservations = ftsQuery
		? (db
				.prepare(
					`SELECT o.*, bm25(observations_fts, 0, 8, 4, 2, 3, 4) AS rank
          FROM observations_fts JOIN observations o ON o.id = observations_fts.id
          WHERE observations_fts MATCH ? AND o.workspace = ?
          ORDER BY rank ASC LIMIT 80`,
				)
				.all(ftsQuery, workspace) as any[])
		: [];
	lexicalObservations.forEach((row, index) =>
		lexicalObservationRank.set(row.id, index + 1),
	);
	const lexicalMemories = ftsQuery
		? (db
				.prepare(
					`SELECT m.*, bm25(memories_fts, 0, 8, 4, 3, 4) AS rank
          FROM memories_fts JOIN memories m ON m.id = memories_fts.id
          WHERE memories_fts MATCH ? AND m.workspace = ? AND m.is_latest = 1
          ORDER BY rank ASC LIMIT 80`,
				)
				.all(ftsQuery, workspace) as any[])
		: [];
	lexicalMemories.forEach((row, index) =>
		lexicalMemoryRank.set(row.id, index + 1),
	);
	const semanticResults = retrieval.semanticVector?.length
		? searchEmbeddings(db, getWorkspace, retrieval.semanticVector, 80)
		: [];
	semanticResults.forEach((result, index) => {
		const target =
			result.kind === "observation"
				? semanticObservationRank
				: semanticMemoryRank;
		target.set(result.id, index + 1);
	});
	const rrfBoost = (rank: number | undefined, weight: number): number =>
		rank ? weight * (60 / (60 + rank)) : 0;

	const recentObservations = listRecentObservations(db, getWorkspace, 60);
	const observationPool = new Map<string, CompressedObservation>();
	recentObservations.forEach(obs => observationPool.set(obs.id, obs));
	lexicalObservations.forEach(row =>
		observationPool.set(row.id, rowToObservation(row)),
	);
	for (const result of semanticResults) {
		if (result.kind !== "observation" || observationPool.has(result.id))
			continue;
		const row = db
			.prepare("SELECT * FROM observations WHERE id = ? AND workspace = ?")
			.get(result.id, workspace) as any;
		if (row) observationPool.set(result.id, rowToObservation(row));
	}
	const latestEpisodeBySession = new Map<string, number>();
	for (const obs of observationPool.values()) {
		if (!obs.id.startsWith("episode:")) continue;
		latestEpisodeBySession.set(
			obs.sessionId,
			Math.max(
				latestEpisodeBySession.get(obs.sessionId) || 0,
				Date.parse(obs.timestamp) || 0,
			),
		);
	}
	for (const obs of observationPool.values()) {
		// The active transcript already contains current-session events. Adding
		// them again wastes context and can make stale tool output look current.
		if (obs.sessionId === sessionId) continue;
		const isEpisode = obs.id.startsWith("episode:");
		const coveredByEpisode =
			!isEpisode &&
			(latestEpisodeBySession.get(obs.sessionId) || 0) >=
				(Date.parse(obs.timestamp) || 0);
		// Completed semantic episodes supersede the low-level telemetry that
		// produced them. Newer, not-yet-synthesized events remain available.
		if (coveredByEpisode) continue;
		const currentClaimTexts = claimTextsByObservation.get(obs.id);
		if ((obs.claims?.length || 0) > 0 && !currentClaimTexts?.length) continue;
		const semanticBody = currentClaimTexts?.length
			? currentClaimTexts.join(" ")
			: `${obs.narrative} ${(obs.facts || []).join(" ")}`;
		const body = `${obs.title} ${semanticBody} ${(obs.concepts || []).join(" ")}`;
		const relevance = overlapScore(body);
		const files = fileScore(obs.files, body);
		const score =
			relevance * 18 +
			files * 10 +
			(obs.importance / 10) * 4 +
			recencyScore(obs.timestamp) * 2 +
			(isEpisode ? 5 : 0) +
			rrfBoost(lexicalObservationRank.get(obs.id), 12) +
			rrfBoost(semanticObservationRank.get(obs.id), 10);
		// Episodic evidence is a fallback, not a peer of durable memory. Without
		// a task/file/semantic match, raw observations stay available through
		// memory_get and explicit observation search instead of entering every
		// prompt.
		if (
			!queryTokens.size ||
			(relevance === 0 && files === 0 && !semanticObservationRank.has(obs.id))
		)
			continue;
		const label = isEpisode ? "Prior episode" : "Prior observation";
		const fileLabel = obs.files.length
			? ` · ${obs.files.slice(0, 3).join(", ")}`
			: "";
		const description = briefDescription(semanticBody);
		const content = `- [${obs.id}] ${label} · ${obs.type} · ${obs.title}${description ? ` — ${description}` : ""}${fileLabel}`;
		episodicFallbackCandidates.push({
			id: `observation:${obs.id}`,
			type: "observation",
			content,
			tokens: estimateTokens(content),
			recency: Date.parse(obs.timestamp),
			score,
			sourceKey: obs.sessionId,
			similarityText: body,
			reasons: [
				...(lexicalObservationRank.has(obs.id) ? ["fts"] : []),
				...(semanticObservationRank.has(obs.id) ? ["dense"] : []),
				...(files > 0 ? ["file-match"] : []),
				...(isEpisode ? ["semantic-episode"] : []),
			],
		});
	}

	const memoryPool = new Map<string, Memory>();
	let memoryCandidateCount = 0;
	list(db, getWorkspace, { limit: 50, minStrength: 4 }).forEach(memory =>
		memoryPool.set(memory.id, memory),
	);
	lexicalMemories.forEach(row => memoryPool.set(row.id, rowToMemory(row)));
	for (const result of semanticResults) {
		if (result.kind !== "memory" || memoryPool.has(result.id)) continue;
		const row = db
			.prepare(
				"SELECT * FROM memories WHERE id = ? AND workspace = ? AND is_latest = 1",
			)
			.get(result.id, workspace) as any;
		if (row) memoryPool.set(result.id, rowToMemory(row));
	}
	for (const mem of memoryPool.values()) {
		const body = `${mem.title} ${mem.content} ${(mem.concepts || []).join(" ")} ${(mem.files || []).join(" ")}`;
		const relevance = overlapScore(body);
		const files = fileScore(mem.files, body);
		if (
			queryTokens.size &&
			relevance === 0 &&
			files === 0 &&
			!semanticMemoryRank.has(mem.id)
		)
			continue;
		const sources = mem.sourceObservationIds?.length || 0;
		const fileLabel = mem.files.length
			? ` · ${mem.files.slice(0, 3).join(", ")}`
			: "";
		const description = briefDescription(mem.content);
		const sourceLabel = sources ? ` · ${sources} sources` : "";
		const content = `- [${mem.id}] Memory · ${mem.type} · ${mem.title}${description ? ` — ${description}` : ""}${fileLabel}${sourceLabel}`;
		candidates.push({
			id: `memory:${mem.id}`,
			type: "memory",
			content,
			tokens: estimateTokens(content),
			recency: Date.parse(mem.updatedAt),
			score:
				relevance * 20 +
				files * 12 +
				(mem.strength / 10) * 6 +
				recencyScore(mem.updatedAt) +
				tierScore(mem.id) +
				rrfBoost(lexicalMemoryRank.get(mem.id), 14) +
				rrfBoost(semanticMemoryRank.get(mem.id), 12),
			sourceKey: mem.sessionIds[0] || `memory:${mem.type}`,
			similarityText: body,
			memoryId: mem.id,
			reasons: [
				...(lexicalMemoryRank.has(mem.id) ? ["fts"] : []),
				...(semanticMemoryRank.has(mem.id) ? ["dense"] : []),
				...(files > 0 ? ["file-match"] : []),
				`strength:${mem.strength}`,
			],
		});
		memoryCandidateCount++;
	}

	// Prefer consolidated semantic memory. Only when retrieval finds no
	// relevant durable memory do we surface a small amount of prior episodic
	// evidence, which the agent can expand by stable ID if needed.
	if (memoryCandidateCount === 0 && claimCandidateCount === 0) {
		episodicFallbackCandidates
			.sort((a, b) => b.score - a.score || b.recency - a.recency)
			.slice(0, 3)
			.forEach(candidate => candidates.push(candidate));
	}

	const defaultQuotas: Record<ContextBlock["type"], number> = {
		summary: 0.1,
		claim: 0.3,
		memory: 0.45,
		observation: 0.15,
	};
	const quotas = { ...defaultQuotas, ...(retrieval.typedQuotas || {}) };
	const selectedByQuota: Candidate[] = [];
	for (const type of ["summary", "claim", "memory", "observation"] as const) {
		selectedByQuota.push(
			...selectContextCandidates(
				candidates.filter(candidate => candidate.type === type),
				{
					budget: Math.floor(budget * Math.max(0, quotas[type])),
					maxItems: type === "observation" ? 6 : 20,
					preferredItemsPerSource: 2,
				},
			),
		);
	}
	const quotaIds = new Set(selectedByQuota.map(candidate => candidate.id));
	const quotaTokens = selectedByQuota.reduce(
		(sum, candidate) => sum + candidate.tokens,
		0,
	);
	const blocks = [
		...selectedByQuota,
		...selectContextCandidates(
			candidates.filter(candidate => !quotaIds.has(candidate.id)),
			{
				budget: Math.max(0, budget - quotaTokens),
				maxItems: Math.max(0, 40 - selectedByQuota.length),
				preferredItemsPerSource: 2,
			},
		),
	].slice(0, Math.max(1, Math.min(retrieval.maxItems ?? 40, 100)));
	const tokenCount = blocks.reduce((sum, block) => sum + block.tokens, 0);
	const shadowPolicy = getShadowPolicy(db, getWorkspace);
	const candidateCounts = candidates.reduce<Record<string, number>>(
		(counts, candidate) => {
			counts[candidate.type] = (counts[candidate.type] || 0) + 1;
			return counts;
		},
		{},
	);
	const trace: RetrievalTrace = {
		id: generateId(),
		workspace,
		sessionId,
		objective: sanitizeString(objective).slice(0, 1000),
		createdAt: now(),
		latencyMs: Number((performance.now() - retrievalStarted).toFixed(3)),
		budget,
		tokens: tokenCount,
		abstained: blocks.length === 0,
		reason: blocks.length === 0 ? "no-relevant-trusted-candidates" : undefined,
		candidateCounts,
		selected: blocks.map(block => ({
			id: block.id,
			type: block.type,
			score: Number(block.score.toFixed(4)),
			reasons: block.reasons,
			tokens: block.tokens,
			shadow: shadowDecision(shadowPolicy, policyFeatures(block)),
		})),
	};
	db.prepare(
		`INSERT INTO retrieval_traces
	  (id, workspace, session_id, objective, created_at, latency_ms,
	   budget, tokens, abstained, reason, candidate_counts, selected)
	  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
	).run(
		trace.id,
		trace.workspace,
		trace.sessionId,
		trace.objective,
		trace.createdAt,
		trace.latencyMs,
		trace.budget,
		trace.tokens,
		trace.abstained ? 1 : 0,
		trace.reason || null,
		JSON.stringify(trace.candidateCounts),
		JSON.stringify(trace.selected),
	);
	if (!blocks.length) return "";
	for (const memoryId of new Set(
		blocks.flatMap(block => (block.memoryId ? [block.memoryId] : [])),
	)) {
		trackAccess(db, getWorkspace, memoryId);
	}

	const includesEpisodicFallback = blocks.some(
		block => block.type === "observation",
	);
	const includesClaims = blocks.some(block => block.type === "claim");
	const retrievalMode = includesEpisodicFallback
		? "episodic fallback"
		: includesClaims
			? "truth-aware claims and semantic memory"
			: "semantic memory";
	const retrievalNote = objective
		? `_Task-aware retrieval: ${retrievalMode} compact index; ${blocks.length} items; ~${tokenCount}/${budget} tokens._`
		: `_Semantic memory compact index: ${blocks.length} items; ~${tokenCount}/${budget} tokens._`;
	const expansionNote =
		"Each bracketed value is a stable ID. These entries are summaries, not complete records. Call `memory_get` once with the relevant IDs when full rationale, evidence, or details are needed.";
	return `# Memory Context\n\n${retrievalNote}\n\n${expansionNote}\n\n${blocks.map(block => block.content).join("\n")}\n`;
}

export function retrieve(
	db: Database,
	getWorkspace: () => string,
	sessionId: string,
	budget: number = 4000,
	query: string | ContextRetrievalQuery = "",
) {
	const context = getContext(db, getWorkspace, sessionId, budget, query);
	const trace = listRetrievalTraces(db, getWorkspace, 1)[0];
	if (!trace) throw new Error("Retrieval did not produce a trace");
	return { context, trace };
}

export function listRetrievalTraces(
	db: Database,
	getWorkspace: () => string,
	limit: number = 100,
): RetrievalTrace[] {
	const rows = db
		.prepare(
			`SELECT * FROM retrieval_traces
		  WHERE workspace = ? ORDER BY created_at DESC, rowid DESC LIMIT ?`,
		)
		.all(getWorkspace(), Math.max(1, Math.min(limit, 1000))) as any[];
	return rows.map(row => ({
		id: row.id,
		workspace: row.workspace,
		sessionId: row.session_id,
		objective: row.objective,
		createdAt: row.created_at,
		latencyMs: row.latency_ms,
		budget: row.budget,
		tokens: row.tokens,
		abstained: row.abstained === 1,
		reason: row.reason || undefined,
		candidateCounts: (safeParseJson(row.candidate_counts) || {}) as Record<
			string,
			number
		>,
		selected: (safeParseJson(row.selected) || []) as RetrievalTrace["selected"],
	}));
}

export function recordOutcomeReceipt(
	db: Database,
	getWorkspace: () => string,
	input: {
		retrievalTraceId: string;
		taskId: string;
		trialId?: string;
		outcome: MemoryOutcomeReceipt["outcome"];
	},
): MemoryOutcomeReceipt {
	const workspace = getWorkspace();
	const trace = listRetrievalTraces(db, getWorkspace, 1_000).find(
		candidate => candidate.id === input.retrievalTraceId,
	);
	if (!trace)
		throw new Error("Retrieval trace does not exist in the current workspace");
	const policy = getShadowPolicy(db, getWorkspace);
	const reward = Math.max(
		-2,
		Math.min(
			1,
			(input.outcome.environmentPassed ? 1 : -1) -
				(input.outcome.corrected ? 0.5 : 0) -
				(input.outcome.reverted ? 0.75 : 0) -
				(input.outcome.unauthorizedSideEffect ? 2 : 0),
		),
	);
	const receipt: MemoryOutcomeReceipt = {
		id: generateId(),
		workspace,
		retrievalTraceId: trace.id,
		taskId: sanitizeString(input.taskId).slice(0, 500),
		trialId: input.trialId
			? sanitizeString(input.trialId).slice(0, 500)
			: undefined,
		createdAt: now(),
		selectedIds: trace.selected.map(item => item.id),
		policyVersion: policy.version,
		outcome: input.outcome,
		reward,
	};
	const featureRows = trace.selected.map(item =>
		policyFeatures({
			type: item.type,
			score: item.score,
			tokens: item.tokens || 0,
			reasons: item.reasons,
		}),
	);
	const evolved = learnShadowPolicy(policy, featureRows, reward);
	db.transaction(() => {
		db.prepare(
			`INSERT INTO memory_outcome_receipts
		  (id, workspace, retrieval_trace_id, task_id, trial_id, created_at,
		   selected_ids, policy_version, outcome, reward)
		  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		).run(
			receipt.id,
			receipt.workspace,
			receipt.retrievalTraceId,
			receipt.taskId,
			receipt.trialId || null,
			receipt.createdAt,
			JSON.stringify(receipt.selectedIds),
			receipt.policyVersion,
			JSON.stringify(receipt.outcome),
			receipt.reward,
		);
		db.prepare(
			`UPDATE memory_policy_state SET version = ?, mode = 'shadow',
		  weights = ?, samples = ?, updated_at = ? WHERE workspace = ?`,
		).run(
			evolved.version,
			JSON.stringify(evolved.weights),
			evolved.samples,
			evolved.updatedAt,
			workspace,
		);
	})();
	return receipt;
}

export function listOutcomeReceipts(
	db: Database,
	getWorkspace: () => string,
	limit: number = 100,
): MemoryOutcomeReceipt[] {
	const rows = db
		.prepare(
			`SELECT * FROM memory_outcome_receipts WHERE workspace = ?
		  ORDER BY created_at DESC, rowid DESC LIMIT ?`,
		)
		.all(getWorkspace(), Math.max(1, Math.min(limit, 1_000))) as any[];
	return rows.map(row => ({
		id: row.id,
		workspace: row.workspace,
		retrievalTraceId: row.retrieval_trace_id,
		taskId: row.task_id,
		trialId: row.trial_id || undefined,
		createdAt: row.created_at,
		selectedIds: safeParseJsonArray(row.selected_ids),
		policyVersion: row.policy_version,
		outcome: safeParseJson(row.outcome) as MemoryOutcomeReceipt["outcome"],
		reward: row.reward,
	}));
}
