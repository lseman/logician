// ── Time-Traveling Stream Rules (TTSR) ────────────────────────────────────────
// Types for rules that interrupt the agent mid-stream when their conditions
// match the output, injecting the rule content and retrying from that point.

// ── Rule ──────────────────────────────────────────────────────────────────────

export interface TtsrRule {
	/** Unique rule identifier (no spaces, no special chars). */
	name: string;
	/** File path where this rule is defined (for display). */
	path: string;
	/** Human-readable description. */
	description: string;
	/** The actual rule content injected on match. */
	content: string;
	/** Regex patterns that trigger this rule against the stream buffer. */
	conditions: string[];
	/** AST-grep patterns for matching tool argument snapshots. */
	astConditions?: string[];
	/** Which streams this rule can match against. */
	scope: TtsrScope[];
	/** Glob patterns for file paths this rule applies to (tool streams only). */
	globs?: string[];
	/** When to interrupt. */
	interruptMode: "always" | "prose-only" | "tool-only" | "never";
	/** Whether this is a built-in rule or user-defined. */
	builtin?: boolean;
}

// ── Scope ─────────────────────────────────────────────────────────────────────

export type TtsrScope =
	| "text"
	| "thinking"
	| "tool"
	| `tool:${string}`;

export const TTSR_SCOPES: readonly TtsrScope[] = ["text", "thinking", "tool", "tool:bash", "tool:eval", "tool:bun"];

/** Validate that a scope string is a recognized TTSR scope. */
export function isTtsrScope(value: string): value is TtsrScope {
	return TTSR_SCOPES.some(s => s === value || s === "tool" || s.startsWith("tool:"));
}

// ── Match Context ─────────────────────────────────────────────────────────────

export type TtsrMatchSource = "text" | "thinking" | "tool";

export interface TtsrMatchContext {
	source: TtsrMatchSource;
	/** Tool name for tool argument deltas (e.g. "bash", "write"). */
	toolName?: string;
	/** Candidate file paths associated with the current stream chunk. */
	filePaths?: string[];
	/** Stable key to isolate buffering (e.g. a tool call ID). */
	streamKey?: string;
}

// ── Settings ──────────────────────────────────────────────────────────────────

export interface TtsrSettings {
	/** Enable/disable TTSR entirely. */
	enabled: boolean;
	/** What to do with messages after the interrupted assistant. */
	contextMode: "discard" | "keep";
	/** Default interrupt mode when not specified per-rule. */
	interruptMode: "always" | "prose-only" | "tool-only" | "never";
	/** How often a rule can trigger. */
	repeatMode: "once" | "gap";
	/** Number of messages between repeats (only for repeatMode: "gap"). */
	repeatGap: number;
	/** Whether to load built-in rules. */
	builtinRules: boolean;
	/** Rule names to disable. */
	disabledRules: string[];
}

// ── Injection Tracking ────────────────────────────────────────────────────────

/** Session entry recording that TTSR rules were injected. */
export interface TtsrInjectionEntry {
	type: "ttsr_injection";
	/** Rule names that were injected. */
	injectedRules: string[];
}

/** Settings snapshot for TTSR, used for comparison in the bridge options. */
export type TtsrBridgeSettings = Partial<TtsrSettings>;
