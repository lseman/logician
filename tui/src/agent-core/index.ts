// ── Agent Core Entry Point ─────────────────────────────────────────────────
// Thin barrel re-exporting the three sub-modules plus the tools barrel.
// External consumers import from here; internals import from sub-paths.

// Core: loop, harness, types, backend, messages, events, session
export * from "./core/backend.ts";
export * from "./core/events.ts";
export * from "./core/file-checkpoints.ts";
export * from "./core/harness.ts";
export * from "./core/loop-detector.ts";
export * from "./core/loop.ts";
export * from "./core/messages.ts";
export * from "./core/session.ts";
export * from "./core/tool-cache.ts";
export * from "./core/types.ts";

// Hooks: hook bus, builtin hooks, budget
export * from "./hooks/builtin-hooks.ts";
export * from "./hooks/hook-bus.ts";
export * from "./hooks/plugin-hooks.ts";

// Shared (registry, parser, permissions, plugins, subagent, skills, mcp, helpers, utils)
export * from "./tools/shared/permissions.ts";
export * from "./tools/shared/parser.ts";
export * from "./tools/shared/plugins.ts";
export * from "./tools/shared/subagent.ts";
export * from "./tools/shared/skills.ts";
export * from "./tools/shared/mcp.ts";
export * from "./tools/shared/system-prompt.ts";
export * from "./tools/shared/async-utils.ts";
export * from "./tools/shared/helpers.ts";
export * from "./tools/shared/path-utils.ts";
export * from "./tools/shared/file-mutation-queue.ts";
export * from "./tools/shared/syntax-highlighter.ts";
export * from "./tools/shared/default-tools.ts";
export * from "./tools/shared/registry.ts";

// Skills (individual tool implementations)
export * from "./tools/skills/truncate.ts";
export * from "./tools/skills/output-accumulator.ts";
export * from "./tools/skills/read-tracker.ts";

// Specific tool exports (for backward compat)
export { bash } from "./tools/skills/bash.ts";
export { edit_file } from "./tools/skills/edit-file.ts";
export { file_diff } from "./tools/skills/file-diff.ts";
export { find } from "./tools/skills/find.ts";
export { git } from "./tools/skills/git.ts";
export { list_files } from "./tools/skills/list-files.ts";
export { read_file } from "./tools/skills/read-file.ts";
export { grep } from "./tools/skills/search.ts";
export {
	getTaskStatus,
	resetTaskStatus,
	task_status,
} from "./tools/skills/task-status.ts";
export type { TaskStatusRecord } from "./tools/skills/task-status.ts";
export type { Task, TaskStatus } from "./tools/skills/todo.ts";
export { getTasks, onTodosChanged } from "./tools/skills/todo.ts";
export { write_file } from "./tools/skills/write-file.ts";

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

// Trust: project trust model
export * from "./trust/index.ts";

// Context files: AGENTS.md/CLAUDE.md discovery
export * from "./context-files/index.ts";

// Prompt templates: reusable markdown templates
export * from "./prompts/index.ts";

// Message queue: steering and follow-up messages
export * from "./message-queue/index.ts";
