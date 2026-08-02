// ── @logician/memory Entry Point ────────────────────────────────────────────
// Persistent memory for Logician agent — SQLite-backed with observation capture,
// synthetic compression, consolidation, and context injection.

export { createMemoryStore } from "./store.js";
export {
  registerMemoryHooks,
  remember,
  recall,
  recallWithTier,
  searchObservations,
  listMemories,
  forget,
  consolidate,
  getContext,
  setSessionId,
  autoForget,
  autoTierMemories,
} from "./hook-adapter.js";
export { createMemoryHooks, type MemoryHooksConfig } from "./memory-hooks.js";

export type {
  // Sessions
  Session,
  // Observations
  RawObservation,
  CompressedObservation,
  ObservationType,
  HookPhase,
  HookPayload,
  // Memories
  Memory,
  MemoryType,
  // Memory Relations
  MemoryRelation,
  MemoryRelationType,
  // Retrieval
  MemoryQuery,
  RecallOptions,
  ContextBlock,
  SearchResult,
  // Working Memory
  WorkingMemoryTier,
  // Retention Scoring
  DecayConfig,
  DecayConfigInput,
  RetentionScore,
  // File Context
  FileContextEntry,
  // Export/Import
  ExportData,
  ImportData,
  ImportResult,
  // Dedup / Auto-Forget
  DedupConfig,
  AutoForgetConfig,
  // Options
  CreateMemoryOptions,
  ObserveOptions,
  // Store
  MemoryStore,
} from "./types.js";
