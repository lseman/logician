// ── Tool Call Guards ──────────────────────────────────────────────────────
//
// Post-turn batch detection (OMP-style): records the full assistant message
// after each turn, canonicalises the tool-call batch, and flags when the
// same batch repeats across consecutive turns.
//
// Failure-loop detection runs independently — it does not depend on argument
// patterns and catches repeated failures on the same tool regardless of args.

export interface LoopGuardDecision {
	block: boolean;
	message?: string;
	/** Which guard tripped — lets callers report/emit without parsing message text. */
	guard?: "batch-loop" | "duplicate" | "failure";
}

/** A batch-loop detection result returned by `recordTurn`. */
export interface RepeatedToolCallDetection {
	readonly kind: "repeated_tool_call";
	/** Name of the first non-exempt tool call that triggered the report. */
	readonly toolName: string;
	/** Number of consecutive turns producing an identical batch. */
	readonly count: number;
	/** Text summary of the tool result for the reported call (first 200 chars). */
	readonly resultSummary: string;
	/** Canonicalised argument text for the reported call (first 400 chars). */
	readonly argumentsSummary: string;
}

export interface LoopDetectorOptions {
	/** Batch-loop threshold — block when the same tool-call batch repeats
	 * across N consecutive assistant turns (default 5, matching OMP). */
	batchThreshold?: number | undefined;
	/** Per-call duplicate threshold — block when the same tool+args is
	 * called N times within a single turn (default 3). */
	duplicateThreshold?: number | undefined;
	/** Failure loop threshold — block when same tool/path/category failed
	 * N times (default 3). */
	failureThreshold?: number | undefined;
	/** Tools excluded from batch/duplicate detection (e.g. polling tools). */
	exemptTools?: readonly string[] | undefined;
}

const DEFAULT_BATCH_THRESHOLD = 5;
const DEFAULT_DUPLICATE_THRESHOLD = 3;
const DEFAULT_FAILURE_THRESHOLD = 3;
const RESULT_SUMMARY_LIMIT = 200;
const ARGUMENT_SUMMARY_LIMIT = 400;
const MAX_CATEGORY_LEN = 120;

// ── Canonicalisation ──────────────────────────────────────────────────────

// Strip harness internals and sort object keys so key order never defeats
// duplicate detection. Numbers stay significant (line ranges, ports, IDs).
function canonicalizeToolCallValue(value: unknown): unknown {
	if (Array.isArray(value)) {
		return value.map(item => canonicalizeToolCallValue(item));
	}
	if (!value || typeof value !== "object") {
		return value;
	}
	const input = value as Record<string, unknown>;
	const output: Record<string, unknown> = {};
	for (const key of Object.keys(input).sort()) {
		// Strip intent fields used by some providers/harnesses — they carry
		// no semantic meaning for duplicate detection.
		if (key === "__intent" || key === "_intent") continue;
		output[key] = canonicalizeToolCallValue(input[key]);
	}
	return output;
}

// Normalize a single string leaf so timestamps and whitespace noise don't
// defeat duplicate detection. Ordinary numbers remain significant.
function normalizeLeaf(value: unknown): unknown {
	if (typeof value === "number") return value;
	if (typeof value !== "string") return value;
	return value
		.replace(
			/\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?/g,
			"#ts",
		)
		.replace(/\b\d{10,13}\b/g, "#ts")
		.replace(/\s+/g, " ")
		.trim();
}

// Target path from common arg names — used to bucket failures by file.
function callPath(args: string): string {
	try {
		const parsed = JSON.parse(args) as Record<string, unknown>;
		const raw = parsed.path ?? parsed.file_path ?? parsed.filename ?? "";
		return String(raw).trim();
	} catch {
		return "";
	}
}

// Coarse error bucket for failure detection.
function failureCategory(toolName: string, result: string): string {
	const body = normalizeLeaf(result.replace(/^Error:\s*/i, "").trim());
	return `${toolName} ${String(body).slice(0, MAX_CATEGORY_LEN)}`;
}

function inc(map: Map<string, number>, key: string): number {
	const next = (map.get(key) || 0) + 1;
	map.set(key, next);
	return next;
}

export class LoopDetector {
	// ── Batch detection (post-turn, OMP-style) ──────────────────────────
	#batchThreshold: number;
	#exemptTools: ReadonlySet<string>;
	#lastHash: string | undefined;
	#batchCount = 0;
	#lastReportedCall:
		| { toolCallId: string; name: string; arguments: string }
		| undefined;

	// ── Per-call duplicate detection (pre-execution) ────────────────────
	#dupThreshold: number;
	#lastCallSignature: string | null = null;
	#consecutiveCallCount = 0;

	// ── Failure detection ───────────────────────────────────────────────
	#failThreshold: number;
	#failSignatureCounts = new Map<string, number>();
	#failCategoryCounts = new Map<string, number>();
	#failPathCounts = new Map<string, number>();

	constructor(options: LoopDetectorOptions = {}) {
		this.#batchThreshold = Math.max(
			1,
			Math.trunc(options.batchThreshold ?? DEFAULT_BATCH_THRESHOLD),
		);
		this.#dupThreshold = Math.max(
			0,
			Math.trunc(options.duplicateThreshold ?? DEFAULT_DUPLICATE_THRESHOLD),
		);
		this.#failThreshold = Math.max(
			0,
			Math.trunc(options.failureThreshold ?? DEFAULT_FAILURE_THRESHOLD),
		);
		this.#exemptTools = new Set(options.exemptTools ?? ["hub"]);
	}

	/**
	 * Record a completed assistant turn and check for repeated batches.
	 * Called after tool results are collected.
	 * Returns a detection when the same batch repeats across turns.
	 */
	recordTurn(
		toolCalls: readonly { id: string; name: string; arguments: string }[],
		toolResults: { id: string; content: string }[],
	): RepeatedToolCallDetection | null {
		// No tool calls in this turn — reset.
		if (toolCalls.length === 0) {
			this.#lastHash = undefined;
			this.#batchCount = 0;
			return null;
		}

		// If every call is exempt, reset.
		if (toolCalls.every(tc => this.#exemptTools.has(tc.name))) {
			this.#lastHash = undefined;
			this.#batchCount = 0;
			return null;
		}

		// Canonicalise: sort keys in args, sort calls by name then id.
		const calls = toolCalls.map(tc => ({
			...tc,
			arguments: JSON.stringify(
				canonicalizeToolCallValue(JSON.parse(tc.arguments || "{}")),
			),
		}));
		calls.sort(
			(a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id),
		);
		const hash = JSON.stringify(calls.map(tc => [tc.name, tc.arguments]));

		if (hash === this.#lastHash) {
			this.#batchCount++;
		} else {
			this.#lastHash = hash;
			this.#batchCount = 1;
		}

		if (this.#batchCount !== this.#batchThreshold) return null;

		// Find first non-exempt call for the report.
		const reportCall =
			toolCalls.find(tc => !this.#exemptTools.has(tc.name)) ?? toolCalls[0]!;
		this.#lastReportedCall = {
			toolCallId: reportCall.id,
			name: reportCall.name,
			arguments: reportCall.arguments,
		};

		// Summarize result.
		const resultMsg = toolResults.find(r => r.id === reportCall.id);
		let resultSummary = "";
		if (resultMsg) {
			const text = String(resultMsg.content).replace(/\s+/g, " ").trim();
			resultSummary =
				text.length > RESULT_SUMMARY_LIMIT
					? `${text.slice(0, RESULT_SUMMARY_LIMIT)}…`
					: text;
		}

		const argSummary = JSON.stringify(
			canonicalizeToolCallValue(JSON.parse(reportCall.arguments || "{}")),
		)
			.replace(/\s+/g, " ")
			.trim();

		return {
			kind: "repeated_tool_call",
			toolName: reportCall.name,
			count: this.#batchCount,
			resultSummary,
			argumentsSummary:
				argSummary.length > ARGUMENT_SUMMARY_LIMIT
					? `${argSummary.slice(0, ARGUMENT_SUMMARY_LIMIT)}…`
					: argSummary,
		};
	}

	// ── Pre-execution: per-call duplicate blocking ─────────────────────

	/**
	 * Check a tool call before execution. Blocks when the same tool+args
	 * is repeated within a single turn (N times). Also blocks if a batch
	 * loop was detected by `recordTurn` in a prior turn.
	 */
	checkToolCall(name: string, args: string): LoopGuardDecision {
		// Batch-loop guard: only block the exact offending call. A changed
		// argument set is a changed approach and must be allowed through.
		const repeatedCallMatches =
			this.#lastReportedCall?.name === name &&
			this.#callSignature(name, args) ===
				this.#callSignature(
					this.#lastReportedCall.name,
					this.#lastReportedCall.arguments,
				);

		if (this.#batchCount >= this.#batchThreshold && repeatedCallMatches) {
			return {
				block: true,
				guard: "batch-loop",
				message: this.#batchLoopMessage(name, this.#batchCount),
			};
		}

		// Per-call duplicate guard.
		if (this.#dupThreshold <= 0) return { block: false };

		const sig = this.#callSignature(name, args);

		if (sig === this.#lastCallSignature) {
			this.#consecutiveCallCount++;
		} else {
			this.#lastCallSignature = sig;
			this.#consecutiveCallCount = 1;
		}

		if (this.#consecutiveCallCount >= this.#dupThreshold) {
			return {
				block: true,
				guard: "duplicate",
				message: this.#dupMessage(name, this.#consecutiveCallCount),
			};
		}

		if (this.#failThreshold > 0) {
			const path = callPath(args);
			if ((this.#failSignatureCounts.get(sig) || 0) >= this.#failThreshold) {
				return this.#tripFailure(name, "the same call");
			}
			if (
				path &&
				(this.#failPathCounts.get(path) || 0) >= this.#failThreshold
			) {
				return this.#tripFailure(name, `\`${path}\``);
			}
			for (const [cat, count] of this.#failCategoryCounts) {
				if (count >= this.#failThreshold && cat.startsWith(`${name} `)) {
					return this.#tripFailure(name, "this kind of operation");
				}
			}
		}

		return { block: false };
	}

	// ── Failure recording ──────────────────────────────────────────────

	recordFailure(name: string, args: string, result: string): void {
		const sig = this.#callSignature(name, args);
		inc(this.#failSignatureCounts, sig);
		const path = callPath(args);
		if (path) inc(this.#failPathCounts, path);
		inc(this.#failCategoryCounts, failureCategory(name, result));
	}

	recordSuccess(name: string, args: string): void {
		this.#failSignatureCounts.delete(this.#callSignature(name, args));
		const path = callPath(args);
		if (path) this.#failPathCounts.delete(path);
		for (const cat of this.#failCategoryCounts.keys()) {
			if (cat.startsWith(`${name} `)) {
				this.#failCategoryCounts.delete(cat);
			}
		}
	}

	// ── Helpers ────────────────────────────────────────────────────────

	#callSignature(name: string, args: string): string {
		try {
			const canon = JSON.stringify(
				canonicalizeToolCallValue(JSON.parse(args || "{}")),
			);
			return `${name} ${canon}`;
		} catch {
			return `${name} ${args || ""}`;
		}
	}

	#batchLoopMessage(name: string, count: number): string {
		return (
			`Error: [batch-loop-guard] blocked — \`${name}\` was part of an identical ` +
			`tool-call batch ${count} times in a row (threshold: ${this.#batchThreshold}). ` +
			`Change your approach.`
		);
	}

	#dupMessage(name: string, count: number): string {
		return (
			`Error: [duplicate-guard] blocked — \`${name}\` called with the same ` +
			`(or cosmetically-varied) arguments ${count} times in a row ` +
			`(threshold: ${this.#dupThreshold}). Stop repeating; change your approach.`
		);
	}

	#tripFailure(toolName: string, target: string): LoopGuardDecision {
		return {
			block: true,
			guard: "failure",
			message:
				`Error: [failure-guard] blocked — \`${toolName}\` has failed on ` +
				`${target} ${this.#failThreshold} times. Stop retrying; inspect the ` +
				`actual error, fix the root cause, or use a different tool.`,
		};
	}

	// ── Reset ──────────────────────────────────────────────────────────

	reset(): void {
		this.#lastHash = undefined;
		this.#batchCount = 0;
		this.#lastReportedCall = undefined;
		this.#lastCallSignature = null;
		this.#consecutiveCallCount = 0;
		this.#failSignatureCounts.clear();
		this.#failCategoryCounts.clear();
		this.#failPathCounts.clear();
	}
}

/**
 * Renders the corrective user message injected after a batch loop is
 * detected (OMP-style loop redirect). The `[loop-redirect:<cause>]` prefix
 * follows the `[continuation-nudge:<reason>]` convention so the TUI renders
 * it as a NOTICE block on transcript replay instead of a "YOU" bubble.
 */
export function renderLoopRedirectMessage(
	detection: RepeatedToolCallDetection,
): string {
	const result =
		detection.resultSummary.length > 0
			? detection.resultSummary
			: "(no text result)";
	return (
		`[loop-redirect:batch_loop] You called \`${detection.toolName}\` ` +
		`${detection.count} consecutive times with identical arguments: ` +
		`\`${detection.argumentsSummary}\`. ` +
		`Last result (truncated): \`${result}\`. ` +
		`Stop repeating this call. Change the arguments, use a different tool, ` +
		`or summarize your findings and yield if the work is complete.`
	);
}
