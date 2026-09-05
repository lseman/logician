// ── Hub (Process Lifecycle Manager) ───────────────────────────────────────────
// Named process management with readiness detection, restart policies, PTY,
// log cursors, and signal control.
//
// Core exports:
//   - ProcessManager      — manages named processes, readiness, restart
//   - createHubTool       — creates the hub tool from a ProcessManager
//   - HubProcessSpec      — launch specification
//   - HubProcessState     — current process state
//   - HubOp               — allowed operations
//   - HubToolDeps         — dependencies injected into the hub tool
//   - HubHookContext      — context passed to lifecycle hooks

/* ── Types ─────────────────────────────────────────────────────────────────── */

/** Allowed operations for the hub tool. */
export type HubOp =
	| "start"
	| "ps"
	| "logs"
	| "stop"
	| "restart"
	| "send"
	| "wait"
	| "describe";

/** Readiness detection strategy. */
export interface HubReadinessSpec {
	/** Regex matched against process stdout/stderr; at least one match required. */
	log?: string;
	/** TCP port that must accept a connection. */
	port?: number;
	/** Host to connect to for port check (default: "127.0.0.1"). */
	host?: string;
	/** Seconds to wait for readiness (default: 30). */
	timeout?: number;
}

/** Launch specification for a named process. */
export interface HubProcessSpec {
	/** Stable, unique name (≤48 chars). */
	name: string;
	/** Application binary or script path. */
	application: string;
	/** CLI arguments. */
	args?: string[];
	/** Working directory (default: cwd of the caller). */
	cwd?: string;
	/** Environment variables to inject. */
	env?: Record<string, string>;
	/** Allocate a PTY for interactive processes (default: false). */
	pty?: boolean;
	/** Readiness detection conditions. */
	ready?: HubReadinessSpec;
	/** Restart policy (default: "no"). */
	restart?: "no" | "on-failure" | "always";
	/** Persist process state to disk so it survives agent shutdown (default: false). */
	persist?: boolean;
	/** Process survives broker/omp shutdown (implies persist, default: false). */
	detached?: boolean;
}

/** Current state of a managed process. */
export interface HubProcessState {
	/** Stable process name. */
	name: string;
	/** Original launch specification. */
	spec: HubProcessSpec;
	/** OS process ID (while running). */
	pid?: number;
	/** Lifecycle status. */
	status: "starting" | "ready" | "running" | "exited" | "killed" | "failed";
	/** Epoch millis when the process was started. */
	startTime?: number;
	/** Exit code (null = still running). */
	exitCode?: number | null;
	/** Log file byte offset for cursor-based reads. */
	logCursor?: number;
	/** Number of times the process has been restarted. */
	restartCount?: number;
}

/** Dependency interface injected into createHubTool. */
export interface HubToolDeps {
	/** Project directory root. */
	projectDir: string;
	/** Default timeout in ms for hub tool operations (default: 60000). */
	defaultTimeoutMs?: number;
	/** Readiness detection timeout in ms (default: 30000). */
	readinessTimeoutMs?: number;
}

/* ── Hooks ──────────────────────────────────────────────────────────────────── */

/** Context passed to hub lifecycle hooks. */
export interface HubHookContext {
	/** Process state at the time the hook fires. */
	process: HubProcessState;
}

/** Hook called before a process is launched. Return false to cancel. */
export type BeforeProcessStartHook = (
	ctx: HubHookContext,
) => boolean | undefined;

/** Hook called immediately after the process is spawned. */
export type AfterProcessStartHook = (ctx: HubHookContext) => void;

/** Hook called when the process reaches its readiness condition. */
export type ProcessReadyHook = (ctx: HubHookContext) => void;

/** Hook called when the process exits. */
export type ProcessExitHook = (ctx: HubHookContext) => void;

/** Hook called when new stdout/stderr data is available. */
export type ProcessStdoutHook = (ctx: HubHookContext, chunk: string) => void;

/** Registry of all hub lifecycle hooks. */
export interface HubHooks {
	beforeProcessStart?: BeforeProcessStartHook;
	afterProcessStart?: AfterProcessStartHook;
	processReady?: ProcessReadyHook;
	processExit?: ProcessExitHook;
	processStdout?: ProcessStdoutHook;
}

/* ── Tool output types ─────────────────────────────────────────────────────── */

/** Generic hub tool response envelope. */
export interface HubResponse {
	/** Human-readable summary. */
	output: string;
	/** Machine-readable state (optional). */
	state?: HubProcessState;
}

/* ── Re-exports from sibling modules ───────────────────────────────────────── */
// ProcessManager and createHubTool are defined in sibling modules.
// These re-exports wire them into the public barrel.

export { createHubTool } from "./hub-tool.ts";
export type { HubProcessManager } from "./process-manager.ts";
