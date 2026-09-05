// ── Eval capability ──────────────────────────────────────────────────────────
// Persistent eval kernel with Python and JS support, workpool batching.

export { createEvalTool, type EvalToolDeps } from "./eval-tool.ts";
export {
	createKernelManager,
	type EvalKernelConfig,
	type EvalResult,
	type KernelManager,
	type KernelManagerConfig,
	type KernelState,
} from "./kernel-manager.ts";

export {
	createWorkpoolTool,
	type WorkpoolDeps,
	type WorkpoolItem,
} from "./workpool-tool.ts";

export {
	createCompletionTool,
	type CompletionToolDeps,
} from "./completion-tool.ts";

export {
	createWaitTool,
	type WaitToolDeps,
} from "./wait-tool.ts";
