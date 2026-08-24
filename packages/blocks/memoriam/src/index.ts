// ── Memoriam — Entry Point ─────────────────────────────────────────────────────
// Standalone memory algorithm: observation→compression→memory pipeline.
// SQLite-backed with synthetic compression, consolidation, retention scoring,
// and context-aware retrieval.

export {
	initialShadowPolicy,
	learnShadowPolicy,
	policyFeatures,
	shadowDecision,
} from "../evolution/shadow-policy.ts";
export {
	evaluateValidityPredicate,
	predicatesAreValid,
	sha256,
} from "../evolution/validity.ts";
export {
	type ContextSelectionCandidate,
	type ContextSelectionOptions,
	selectContextCandidates,
} from "./retrieval/context-selector.ts";
export {
	createMemoryStore,
	type MemoryStore,
} from "./store/index.ts";

export { autoForget } from "./store/policy/auto-forget.ts";
export { consolidate } from "./store/policy/consolidation.ts";
export { dedupCheck, dedupRecord } from "./store/policy/dedup.ts";
export {
	computeRetentionScore,
	listByRetentionScore,
	type RetentionScore,
	rescoreAll,
} from "./store/policy/retention-scoring.ts";
export { buildSyntheticCompression } from "./store/synthetic-compression.ts";

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
