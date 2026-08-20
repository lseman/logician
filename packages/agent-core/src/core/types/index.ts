// ── Types block ─────────────────────────────────────────────────────────────
// Shared type definitions for config, messages, and agent behavior.

export * from "./types-config.ts";
export * from "./types-messages.ts";
// Re-export execution profile type from core for config consumers.
export type { ExecutionProfile } from "../policy/execution-policy.ts";
