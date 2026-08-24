// ── Memoriam — Entry Point ─────────────────────────────────────────────────────
// Standalone memory algorithm: observation→compression→memory pipeline.
// SQLite-backed with synthetic compression, consolidation, retention scoring,
// and context-aware retrieval.

export {
	createMemoryStore,
	type MemoryStore,
} from "./store/index.ts";

export {
	selectContextCandidates,
	type ContextSelectionCandidate,
	type ContextSelectionOptions,
} from "./retrieval/context-selector.ts";

export {
	computeRetentionScore,
	type RetentionScore,
	rescoreAll,
	listByRetentionScore,
} from "./store/policy/retention-scoring.ts";

export { consolidate } from "./store/policy/consolidation.ts";

export { autoForget } from "./store/policy/auto-forget.ts";

export { dedupCheck, dedupRecord } from "./store/policy/dedup.ts";

export { buildSyntheticCompression } from "./store/synthetic-compression.ts";

export {
	initialShadowPolicy,
	shadowDecision,
	learnShadowPolicy,
	policyFeatures,
} from "../evolution/shadow-policy.ts";

export {
	predicatesAreValid,
	evaluateValidityPredicate,
	sha256,
} from "../evolution/validity.ts";

// Re-export all domain types from the types module.
export type {
	AutoForgetConfig,
	ClaimEvidenceCertificate,
	ClaimLifecycle,
	ClaimValidityPredicate,
	CompressedObservation,
	ContextBlock,
	ContextRetrievalQuery,
	CreateMemoryOptions,
	DecayConfig,
	DecayConfigInput,
	EmbeddingMetadata,
	ExpandedMemoryEntry,
	ExportData,
	FileContextEntry,
	ImportData,
	ImportResult,
	Memory,
	MemoryClaim,
	MemoryOutcomeReceipt,
	MemoryQuery,
	MemoryRelation,
	MemoryRelationType,
	MemoryRetrievalResult,
	MemoryType,
	ObservationClaim,
	ObservationProvenance,
	ObservationTrust,
	ObservationType,
	RawObservation,
	RecallOptions,
	SearchResult,
	SemanticSearchResult,
	Session,
	ShadowMemoryPolicy,
	SnapshotMeta,
	WorkingMemoryTier,
} from "./types.ts";
