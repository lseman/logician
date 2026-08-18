// ── Agent Core Entry Point ─────────────────────────────────────────────────
// Barrel re-exporting every module. This is the only supported way to import
// from @logician/agent-core — there is no other public entry point.
//
// Module layout:
//   core        — agent loop, harness, backend, session, runtime state
//   harness     — harness orchestration, branching, session lifecycle
//   hooks       — hook bus, built-in policy hooks
//   extension   — extension system (loader, runner, types, adapters/*)
//   guards      — loop detector, output guard, acceptance
//   types       — config, messages, agent types
//   tools       — tool registry, parser, permissions, utils
//   compaction  — context window compaction
//   queue       — steering/follow-up message queues
//   loop        — provider interaction, streaming, reflection
//   summaries   — branch/summary generation
//   tasks       — task state, continuation policy
//   config      — config validation

// ── Core ────────────────────────────────────────────────────────────────────
export * from "./core/agent-loop-runner.ts";
export * from "./core/agent-settings.ts";
export * from "./core/backend.ts";
export {
	type ContinuationDecision,
	type ContinuationLimits,
	type ContinuationState,
	ContinuationTracker,
	type RunBudgetStatus,
} from "./core/continuation-tracker.ts";
export * from "./core/execution-policy.ts";
export * from "./core/file-checkpoints.ts";
export * from "./core/intervention-controller.ts";
export * from "./core/messages.ts";
export {
	createRuntimeState,
	reduceRuntimeState,
	type AgentRuntimeState,
	type HarnessPhase,
} from "./core/runtime-state.ts";
export * from "./core/session.ts";
export * from "./core/tool-cache.ts";
export * from "./core/exit-path.ts";
export * from "./core/run-budget.ts";
export * from "./core/conclusion-policy.ts";
export * from "./core/harness-queue-hooks.ts";
export * from "./core/tool-batch-controller.ts";

// ── Harness ─────────────────────────────────────────────────────────────────
export * from "./harness/harness.ts";
export type { BranchInfo, BranchSummaryData } from "./summaries/types.ts";
export type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
} from "./harness/contracts.ts";
export { HarnessBusyError } from "./harness/phase.ts";

// ── Guards ──────────────────────────────────────────────────────────────────
export * from "./guards/index.ts";
export type { ReflectionConfig } from "./loop/reflection.ts";

// ── Types ───────────────────────────────────────────────────────────────────
export * from "./types/index.ts";

// ── Tasks ───────────────────────────────────────────────────────────────────
export type { TaskStatusRecord } from "./tasks/task-status-state.ts";
export {
	getTaskStatus,
	recordTaskStatus,
	resetTaskStatus,
} from "./tasks/task-status-state.ts";
export type { Task, TaskStatus } from "./tasks/todo-state.ts";
export {
	allocateTaskId,
	getTasks,
	getTasksMutable,
	notifyTodosChanged,
	onTodosChanged,
	replaceTasks,
} from "./tasks/todo-state.ts";

// ── Compaction ──────────────────────────────────────────────────────────────
export {
	type CompactionSettings,
	compactToFit,
	DEFAULT_COMPACTION_SETTINGS,
	estimateContextTokens,
	shouldCompact,
} from "./compaction/index.ts";

// ── Extension ───────────────────────────────────────────────────────────────
export * from "./extension/index.ts";

// ── Hooks ───────────────────────────────────────────────────────────────────
export * from "./hooks/builtin/index.ts";
export * from "./hooks/index.ts";

// ── Queue ───────────────────────────────────────────────────────────────────
export * from "./queue/index.ts";

// ── Tools ───────────────────────────────────────────────────────────────────
export * from "./tools/index.ts";
