// ── @logician/memory Domain Types ────────────────────────────────────────────
// Inspired by agentmemory's observation→compression→memory pipeline.

// ── Session ──────────────────────────────────────────────────────────────────

export interface Session {
	id: string;
	name?: string;
	project: string;
	cwd: string;
	workspace: string; // Normalized workspace identifier
	startedAt: string;
	endedAt?: string;
	status: "active" | "completed" | "abandoned";
	observationCount: number;
	model?: string;
	tags?: string[];
	firstPrompt?: string;
	summary?: string;
	commitShas?: string[];
}

// ── Working Memory Tiers ─────────────────────────────────────────────────────

export type WorkingMemoryTier = "hot" | "warm" | "cold" | "archived";

// ── Memory Relations ─────────────────────────────────────────────────────────

export type MemoryRelationType =
	| "supersedes"
	| "contradicts"
	| "related_to"
	| "supports"
	| "extends";

export interface MemoryRelation {
	id: string;
	type: MemoryRelationType;
	sourceId: string;
	targetId: string;
	confidence: number; // 0-1
	createdAt: string;
}

// ── Retention Scoring ────────────────────────────────────────────────────────

export interface DecayConfig {
	/** Exponential decay rate per day (default: 0.01) */
	lambda?: number;
	/** Reinforcement boost factor (default: 0.3) */
	sigma?: number;
	/** Tier thresholds for retention score (default: hot=0.7, warm=0.4, cold=0.15) */
	tierThresholds?: { hot: number; warm: number; cold: number };
}

export interface RetentionScore {
	id: string;
	score: number; // 0-1
	decayFactor: number;
	reinforcementBoost: number;
	tier: WorkingMemoryTier;
	type: MemoryType;
	strength: number;
}

export interface DecayConfigInput {
	lambda?: number;
	sigma?: number;
	tierThresholds?: { hot?: number; warm?: number; cold?: number };
}

// ── File Context Index ───────────────────────────────────────────────────────

export interface FileContextEntry {
	file: string;
	observations: Array<{
		sessionId: string;
		obsId: string;
		type: ObservationType;
		title: string;
		narrative: string;
		importance: number;
		timestamp: string;
	}>;
}

// ── Export/Import ────────────────────────────────────────────────────────────

export interface ExportData {
	version: number;
	exportedAt: string;
	sessions: Session[];
	observations: ExportedObservation[];
	claims: MemoryClaim[];
	memories: Memory[];
	relations: MemoryRelation[];
}

/** Complete persisted observation shape used by lossless backup/restore. */
export interface ExportedObservation extends CompressedObservation {
	hookType: HookPhase | "import";
	rawData?: unknown;
}

export interface ImportData {
	version: number;
	sessions: Session[];
	observations: Array<CompressedObservation | ExportedObservation>;
	claims?: MemoryClaim[];
	memories: Memory[];
	relations?: MemoryRelation[];
	onConflict?: "skip" | "update";
}

export interface ImportResult {
	imported: number;
	skipped: number;
	errors: string[];
}

// ── Snapshot ─────────────────────────────────────────────────────────────────

export interface SnapshotMeta {
	sessionId: string;
	gitCommit?: string;
	gitBranch?: string;
	createdAt: string;
	observationCount: number;
	memoryCount: number;
}

/** Live task signals used to rank bounded context candidates. */
export interface ContextRetrievalQuery {
	objective: string;
	phase?:
		| "orient"
		| "investigate"
		| "implement"
		| "verify"
		| "handoff"
		| "blocked";
	changedFiles?: string[];
	recentEvidence?: string[];
	toolFailures?: number;
	/** Internal optional query embedding used by hybrid retrieval. */
	semanticVector?: number[];
	/** Optional per-kind fractions; unspecified kinds use safe defaults. */
	typedQuotas?: Partial<Record<ContextBlock["type"], number>>;
}

// ── Dedup Config ─────────────────────────────────────────────────────────────

export interface DedupConfig {
	/** Window in ms to consider observations duplicates (default: 300_000 = 5min) */
	windowMs?: number;
}

// ── Auto-Forget Config ───────────────────────────────────────────────────────

export interface AutoForgetConfig {
	/** Observations older than this (ms) are eligible for auto-forget (default: 30d) */
	ttlMs?: number;
	/** Minimum importance score to keep (default: 3) */
	minImportance?: number;
	/** Max observations to delete per call (default: 100) */
	maxDeletes?: number;
}

// ── Access Tracker Config ────────────────────────────────────────────────────

export interface AccessTrackerConfig {
	/** Auto-set working memory tier based on last access (default: true) */
	autoTier?: boolean;
}

// ── Hook Payload (what hooks emit) ───────────────────────────────────────────

export type HookPhase =
	| "session_start"
	| "prompt_submit"
	| "pre_tool_use"
	| "post_tool_use"
	| "post_tool_failure"
	| "pre_compact"
	| "stop"
	| "notification";

export interface HookPayload {
	hookType: HookPhase;
	sessionId: string;
	project: string;
	cwd: string;
	workspace: string;
	timestamp: string;
	data: unknown;
}

// ── Observation ──────────────────────────────────────────────────────────────

export interface RawObservation {
	id: string;
	sessionId: string;
	timestamp: string;
	hookType: HookPhase;
	toolName?: string;
	toolInput?: unknown;
	toolOutput?: unknown;
	userPrompt?: string;
	workspace?: string;
	raw: unknown;
}

export type ClaimStatus = "tentative" | "verified" | "invalidated";
export type ObservationTrust = "trusted_local" | "external" | "untrusted";
export type ClaimLifecycle =
	| "probationary"
	| "durable"
	| "contested"
	| "stale"
	| "superseded"
	| "quarantined";

/** Deterministic environment assertion that can invalidate only its owning claim. */
export type ClaimValidityPredicate =
	| { type: "file_hash"; path: string; sha256: string }
	| { type: "git_revision"; revision: string }
	| { type: "config_value"; path: string; key: string; sha256: string };

export interface ClaimEvidenceCertificate {
	extractorVersion: string;
	schemaVersion: number;
	evidenceEventIds: string[];
	issuedAt: string;
}

/** An atomic claim derived from immutable turn evidence. */
export interface ObservationClaim {
	text: string;
	confidence: number;
	status: ClaimStatus;
	evidenceEventIds: string[];
	validityPredicates?: ClaimValidityPredicate[];
}

/** Extraction lineage kept alongside an observation for later auditing. */
export interface ObservationProvenance {
	source: "model" | "deterministic";
	trust: ObservationTrust;
	extractorVersion: string;
	schemaVersion: number;
	rejectionReason?: string;
}

export type ClaimRevisionOperation =
	| "ADD"
	| "SUPERSEDE"
	| "INVALIDATE"
	| "NOOP";

/** Append-only durable claim derived from one or more evidence events. */
export interface MemoryClaim {
	id: string;
	workspace: string;
	observationId: string;
	sessionId: string;
	text: string;
	status: ClaimStatus;
	confidence: number;
	operation: ClaimRevisionOperation;
	validFrom: string;
	validTo?: string;
	transactionTime: string;
	source: ObservationProvenance["source"];
	trust: ObservationTrust;
	extractorVersion: string;
	schemaVersion: number;
	supersedesClaimId?: string;
	supersededByClaimId?: string;
	tombstonedAt?: string;
	evidenceEventIds: string[];
	lifecycle: ClaimLifecycle;
	validityPredicates: ClaimValidityPredicate[];
	evidenceCertificate: ClaimEvidenceCertificate;
}

/** Synthetic compression: zero-LLM summary derived from raw observation. */
export interface CompressedObservation {
	id: string;
	sessionId: string;
	timestamp: string;
	type: ObservationType;
	title: string;
	subtitle?: string;
	facts: string[];
	narrative: string;
	concepts: string[];
	files: string[];
	importance: number;
	consolidated: boolean;
	workspace?: string;
	claims?: ObservationClaim[];
	provenance?: ObservationProvenance;
}

export type ObservationType =
	| "file_read"
	| "file_write"
	| "file_edit"
	| "command_run"
	| "search"
	| "web_fetch"
	| "conversation"
	| "error"
	| "decision"
	| "discovery"
	| "implementation"
	| "bugfix"
	| "notification"
	| "other";

// ── Memory ───────────────────────────────────────────────────────────────────

export type MemoryType =
	| "pattern"
	| "preference"
	| "architecture"
	| "bug"
	| "workflow"
	| "fact";

export interface Memory {
	id: string;
	createdAt: string;
	updatedAt: string;
	type: MemoryType;
	title: string;
	content: string;
	concepts: string[];
	files: string[];
	sessionIds: string[];
	strength: number; // 1-10
	version: number;
	parentId?: string;
	supersedes?: string[]; // parent memory IDs this supersedes
	relatedIds?: string[];
	sourceObservationIds?: string[];
	isLatest: boolean;
	project?: string;
	workspace?: string; // Workspace scope (derived from session)
	accessCount?: number;
	lastAccessed?: string;
	workingTier?: WorkingMemoryTier;
}

// ── Retrieval ────────────────────────────────────────────────────────────────

export interface SearchResult {
	observation: CompressedObservation;
	score: number;
	sessionId: string;
}

export interface ExpandedMemoryEntry {
	id: string;
	kind: "observation" | "memory";
	title: string;
	content: string;
	type: string;
	files: string[];
	concepts: string[];
	timestamp: string;
	sessionIds: string[];
}

export interface SemanticSearchResult {
	id: string;
	kind: "observation" | "memory";
	sessionId?: string;
	score: number;
}

export interface EmbeddingMetadata {
	model: string;
	contentHash: string;
	creationVersion: number;
}

export interface MemoryQuery {
	/** Full-text search across memory content and titles */
	search?: string;
	/** Filter by memory type */
	type?: MemoryType;
	/** Filter by tags/concepts (AND semantics) */
	concepts?: string[];
	/** Filter by session */
	sessionId?: string;
	/** Filter by project */
	project?: string;
	/** Filter by workspace. Default: current workspace. */
	workspace?: string;
	/** Minimum strength score */
	minStrength?: number;
	/** Max results (default: 10) */
	limit?: number;
}

export interface ContextBlock {
	type: "summary" | "observation" | "memory" | "claim";
	content: string;
	tokens: number;
	recency: number;
}

export interface RetrievalTrace {
	id: string;
	workspace: string;
	sessionId: string;
	objective: string;
	phase: ContextRetrievalQuery["phase"];
	createdAt: string;
	latencyMs: number;
	budget: number;
	tokens: number;
	abstained: boolean;
	reason?: string;
	candidateCounts: Record<string, number>;
	selected: Array<{
		id: string;
		type: ContextBlock["type"];
		score: number;
		reasons: string[];
		tokens?: number;
		shadow?: {
			action: "inject" | "withhold";
			score: number;
			policyVersion: number;
		};
	}>;
}

export interface MemoryOutcomeReceipt {
	id: string;
	workspace: string;
	retrievalTraceId: string;
	taskId: string;
	trialId?: string;
	createdAt: string;
	selectedIds: string[];
	policyVersion: number;
	outcome: {
		environmentPassed: boolean;
		corrected?: boolean;
		reverted?: boolean;
		unauthorizedSideEffect?: boolean;
		tokens?: number;
		durationMs?: number;
	};
	reward: number;
}

export interface ShadowMemoryPolicy {
	version: number;
	mode: "deterministic" | "shadow";
	weights: number[];
	samples: number;
	updatedAt: string;
}

export type ExtractionJobStatus =
	| "pending"
	| "running"
	| "completed"
	| "failed";

export interface ExtractionJob {
	id: string;
	sessionId: string;
	workspace: string;
	payload: string;
	status: ExtractionJobStatus;
	attempts: number;
	createdAt: string;
	updatedAt: string;
	nextAttemptAt: string;
	lastError?: string;
}

// ── Options ──────────────────────────────────────────────────────────────────

export interface CreateMemoryOptions {
	/** Memory type, default: "fact" */
	type?: MemoryType;
	/** Explicit concepts (overrides auto-extraction) */
	concepts?: string[];
	/** Explicit files list */
	files?: string[];
	/** Strength score 1-10, default: 5 */
	strength?: number;
	/** Source sessions */
	sessionIds?: string[];
	/** Parent memory this supersedes */
	parentId?: string;
	/** Project scope */
	project?: string;
	/** Workspace scope (derived from session if omitted) */
	workspace?: string;
}

export interface RecallOptions {
	/** Output format */
	format?: "text" | "system-prompt" | "markdown";
	/** Template for text format — {{content}}, {{importance}}, {{title}} */
	template?: string;
	/** Inject into a specific context (e.g., system prompt) */
	contextId?: string;
}

export interface ObserveOptions {
	/** Session ID for the observation */
	sessionId: string;
	/** Project name */
	project?: string;
	/** Current working directory */
	cwd?: string;
	/** Workspace identifier */
	workspace?: string;
	/** Auto-compress into synthetic summary */
	autoCompress?: boolean;
}

// ── Store Interface ──────────────────────────────────────────────────────────

export interface MemoryStore {
	// Sessions
	createSession(id: string, data: Partial<Session>): Session;
	getSession(id: string): Session | null;
	listSessions(query?: { status?: string; project?: string }): Session[];
	updateSession(id: string, updates: Partial<Session>): Session | null;
	/** Remove folder sessions except an optional active session, with their observations. */
	clearSessions(keepSessionId?: string): {
		sessions: number;
		observations: number;
	};
	/** Remove one session only when it has no observations or durable references. */
	discardEmptySession(id: string): boolean;

	// Observations
	observe(
		raw: RawObservation,
		compressed?: CompressedObservation,
	): CompressedObservation | null;
	getObservation(id: string, sessionId: string): CompressedObservation | null;
	listObservations(sessionId: string, limit?: number): CompressedObservation[];
	/** List recent observations directly, scoped to the current workspace by default. */
	listRecentObservations(
		limit?: number,
		type?: ObservationType,
	): CompressedObservation[];
	/** List append-only derived claims, newest transaction first. */
	listClaims(options?: {
		observationId?: string;
		status?: ClaimStatus;
		includeSuperseded?: boolean;
		limit?: number;
	}): MemoryClaim[];
	searchObservations(query: string, limit?: number): SearchResult[];
	expandEntries(ids: string[]): ExpandedMemoryEntry[];
	/** Permanently remove observations in the current workspace. */
	clearObservations(): number;

	// Memories
	create(content: string, options?: CreateMemoryOptions): Memory;
	get(id: string): Memory | null;
	/** Get any version of a memory (including non-latest) */
	getAny(id: string): Memory | null;
	list(query?: MemoryQuery): Memory[];
	remove(id: string): boolean;
	/** Permanently remove memories and their relations in the current workspace. */
	clearMemories(): number;
	update(
		id: string,
		updates: Partial<
			Pick<Memory, "content" | "concepts" | "strength" | "title">
		>,
	): Memory | null;
	recall(query: MemoryQuery, options?: RecallOptions): string;
	consolidate(sessionId: string): Memory[];

	// Durable semantic extraction queue
	enqueueExtractionJob(
		sessionId: string,
		workspace: string,
		payload: string,
	): ExtractionJob;
	claimExtractionJob(): ExtractionJob | null;
	completeExtractionJob(id: string): void;
	failExtractionJob(id: string, error: string, retryDelayMs?: number): void;
	listExtractionJobs(status?: ExtractionJobStatus): ExtractionJob[];

	// Context injection
	getContext(
		sessionId: string,
		budget?: number,
		query?: string | ContextRetrievalQuery,
	): string;
	listRetrievalTraces(limit?: number): RetrievalTrace[];
	recordOutcomeReceipt(input: {
		retrievalTraceId: string;
		taskId: string;
		trialId?: string;
		outcome: MemoryOutcomeReceipt["outcome"];
	}): MemoryOutcomeReceipt;
	listOutcomeReceipts(limit?: number): MemoryOutcomeReceipt[];
	promoteClaim(
		claimId: string,
		evidenceEventIds?: string[],
	): MemoryClaim | null;
	getShadowPolicy(): ShadowMemoryPolicy;
	upsertEmbedding(
		id: string,
		kind: "observation" | "memory",
		vector: number[],
		sessionId?: string,
		metadata?: EmbeddingMetadata,
	): void;
	hasEmbedding(id: string, metadata?: Partial<EmbeddingMetadata>): boolean;
	searchEmbeddings(vector: number[], limit?: number): SemanticSearchResult[];

	// Session tracking
	setCurrentSessionId(id: string): void;
	getCurrentSessionId(): string | null;
	setCurrentWorkspace(ws: string): void;
	getCurrentWorkspace(): string;

	// ── Dedup ──────────────────────────────────────────────────────────────
	/** Check if an observation is a duplicate (same session/tool/input within window). */
	dedupCheck(sessionId: string, toolName: string, toolInput: unknown): boolean;
	/** Record a dedup hash after processing. */
	dedupRecord(sessionId: string, toolName: string, toolInput: unknown): void;

	// ── Sliding Window ────────────────────────────────────────────────────
	/** Enforce observation cap per session; returns number of evicted obs. */
	slidingWindowCap(sessionId: string, cap?: number): number;

	// ── Access Tracker ────────────────────────────────────────────────────
	/** Increment access count for a memory. */
	trackAccess(entityId: string): void;
	/** Get access stats for a memory. */
	getAccessStats(
		entityId: string,
	): { lastAccessed: string; accessCount: number } | null;

	// ── Working Memory Tiers ──────────────────────────────────────────────
	/** Get working memory tier for a memory entity. */
	getWorkingMemoryTier(entityId: string): WorkingMemoryTier;
	/** Set working memory tier explicitly. */
	setWorkingMemoryTier(entityId: string, tier: WorkingMemoryTier): void;

	// ── Memory Relations ─────────────────────────────────────────────────
	/** Create a relation between two memories. */
	relate(
		sourceId: string,
		targetId: string,
		type: MemoryRelationType,
		confidence?: number,
	): MemoryRelation | null;
	/** Get all relations for a memory. */
	getRelations(memoryId: string): MemoryRelation[];
	/** Get related memories via BFS traversal. */
	getRelatedMemories(
		memoryId: string,
		maxHops?: number,
		minConfidence?: number,
	): Array<{ memory: Memory; hop: number; confidence: number }>;
	/** Evolve a memory (versioning: creates new version, links old as supersedes). */
	evolve(
		memoryId: string,
		newContent: string,
		newTitle?: string,
	): { memory: Memory; previousId: string } | null;
	/** Delete a relation. */
	removeRelation(relationId: string): boolean;

	// ── Retention Scoring ────────────────────────────────────────────────
	/** Compute retention score for a memory using exponential decay. */
	computeRetentionScore(
		id: string,
		config?: DecayConfigInput,
	): RetentionScore | null;
	/** Rescore all memories and return scores. */
	rescoreAll(config?: DecayConfigInput): RetentionScore[];
	/** Get memories sorted by retention score (highest first). */
	listByRetentionScore(
		config?: DecayConfigInput,
		limit?: number,
	): RetentionScore[];

	// ── File Context Index ───────────────────────────────────────────────
	/** Get file-specific context: observations mentioning this file. */
	getFileContext(file: string, sessionId?: string): FileContextEntry | null;
	/** Get file context for multiple files. */
	getFilesContext(files: string[], sessionId?: string): FileContextEntry[];
	/** Rebuild the file context index from all observations. */
	rebuildFileIndex(): number;

	// ── Export/Import ────────────────────────────────────────────────────
	/** Export all data to a JSON-serializable snapshot. */
	exportData(): ExportData;
	/** Import data from a snapshot. */
	importData(data: ImportData): ImportResult;
	/** Auto-tier all memories by retention score (exponential decay + access
	 *  reinforcement, weighted by type salience) — see computeRetentionScore. */
	autoTierMemories(
		config?: DecayConfigInput,
	): Record<string, WorkingMemoryTier>;

	// ── Auto-Forget ─────────────────────────────────────────────────────
	/** Delete old, low-importance observations. */
	autoForget(
		ttlMs?: number,
		minImportance?: number,
		maxDeletes?: number,
	): { deleted: number; details: string[] };

	// Utility
	close(): void;
}
