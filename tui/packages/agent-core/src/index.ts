// ── Agent Core Entry Point ─────────────────────────────────────────────────
// Thin barrel re-exporting the three sub-modules plus the tools barrel.
// External consumers import from here; internals import from sub-paths.

// Core: loop, harness, types, backend, messages, events, session
export * from "./core/backend.ts";
export * from "./core/agent-loop-runner.ts";
export * from "./core/events.ts";
export * from "./core/file-checkpoints.ts";
export * from "./core/harness.ts";
export * from "./core/loop-detector.ts";
export * from "./core/messages.ts";
export * from "./core/session.ts";
export * from "./core/tool-cache.ts";
export * from "./core/types.ts";

// Hooks: hook bus, builtin hooks, budget
export * from "./hooks/builtin-hooks.ts";
export * from "./hooks/hook-bus.ts";
export * from "./hooks/plugin-hooks.ts";

// Shared (registry, parser, permissions, plugins, subagent, skills, mcp, utils)
export * from "./tools/shared/permissions.ts";
export * from "./tools/shared/parser.ts";
export * from "./tools/shared/plugins.ts";
export * from "./tools/shared/async-utils.ts";
export * from "./tools/shared/path-utils.ts";
export * from "./tools/shared/syntax-highlighter.ts";
export * from "./tools/shared/registry.ts";

// Generic workflow tools used by built-in hooks.
export {
	getTaskStatus,
	resetTaskStatus,
	task_status,
} from "./tools/workflow/task-status.ts";
export type { TaskStatusRecord } from "./tools/workflow/task-status.ts";
export type { Task, TaskStatus } from "./tools/todos/todo.ts";
export { getTasks, onTodosChanged } from "./tools/todos/todo.ts";

// Compaction (kept at top level for backwards compat).
// Note: core/messages.ts and compaction/compaction.ts both define
// estimateTokens (string→number vs message→number) and compactToFit.
// The messages.ts versions are exported through the barrel; the
// compaction.ts versions are internal (no export) to avoid collisions.
export {
	compact,
	estimateContextTokens,
	shouldCompact,
	type CompactionSettings,
	DEFAULT_COMPACTION_SETTINGS,
	recoverFromContextFull,
} from "./compaction/index.ts";

// Extensions: TypeScript extension system
export * from "./extensions/index.ts";

// Message queue: steering and follow-up messages
export * from "./message-queue/index.ts";
