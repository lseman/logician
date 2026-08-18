// ── Tool Call Guards ──────────────────────────────────────────────────────
// Pre-execution guards that block individual tool calls:
//   - Duplicates (same tool+args called N times)
//   - Repeated failures (same tool/path/category failed N times)

export interface LoopGuardDecision {
	block: boolean;
	message?: string;
	/** Which guard tripped — lets callers report/emit without parsing message text. */
	guard?: "duplicate" | "failure";
}

export interface LoopDetectorOptions {
	/** Duplicate call threshold — block when same tool+args called N times (default 3). */
	duplicateThreshold?: number;
	/** Failure loop threshold — block when same tool/path/category failed N times (default 3). */
	failureThreshold?: number;
}

const DEFAULT_DUPLICATE_THRESHOLD = 3;
const DEFAULT_FAILURE_THRESHOLD = 3;
const MAX_CATEGORY_LEN = 120;

// Normalize a single string leaf so timestamps and whitespace don't defeat
// duplicate detection. Ordinary numbers remain significant: line ranges,
// ports, IDs, offsets, and retry parameters can represent genuinely different
// work and must not be collapsed into one call signature.
function normalizeLeaf(value: unknown): unknown {
	if (typeof value === "number") return value;
	if (typeof value !== "string") return value;
	return value
		.replace(
			/\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?/g,
			"#ts",
		)
		.replace(/\b\d{10,13}\b/g, "#ts") // unix ms/sec timestamps
		.replace(/\s+/g, " ")
		.trim();
}

function normalizeForSignature(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(normalizeForSignature);
	if (value && typeof value === "object") {
		const out: Record<string, unknown> = {};
		for (const k of Object.keys(value as Record<string, unknown>).sort()) {
			out[k] = normalizeForSignature((value as Record<string, unknown>)[k]);
		}
		return out;
	}
	return normalizeLeaf(value);
}

// Stable signature for a tool call: name + canonical args, with cosmetic
// noise normalized out so near-duplicates collapse to the same signature.
function callSignature(name: string, args: string): string {
	let argsKey = args || "";
	try {
		argsKey = JSON.stringify(normalizeForSignature(JSON.parse(args || "{}")));
	} catch (_e: unknown) {
		// Non-JSON args: use the raw string.
	}
	return `${name} ${argsKey}`;
}

// Target path from common arg names — used to bucket failures by file.
function callPath(args: string): string {
	try {
		const parsed = JSON.parse(args) as Record<string, unknown>;
		const raw = parsed.path ?? parsed.file_path ?? parsed.filename ?? "";
		return String(raw).trim();
	} catch (_e: unknown) {
		return "";
	}
}

// Coarse error bucket so distinct-but-equivalent failures collapse together.
// The body is normalized the same way call args are (numbers/timestamps ->
// placeholders) before truncation, so two errors that differ only in a line
// number or a timestamp still bucket together, while errors that differ in
// substance (a different missing module, a different variable name) do not
// collapse just because they happen to share a raw text prefix.
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
	// ── Guard state ───────────────────────────────────────────────────────
	// Duplicate guard: only counts CONSECUTIVE identical tool+args calls.
	// Reset when a different tool or different args is called.
	private lastCallSignature: string | null = null;
	private consecutiveCallCount = 0;
	private failSignatureCounts = new Map<string, number>();
	private failCategoryCounts = new Map<string, number>();
	private failPathCounts = new Map<string, number>();

	private readonly duplicateThreshold: number;
	private readonly failureThreshold: number;

	constructor(options: LoopDetectorOptions = {}) {
		this.duplicateThreshold = Math.max(
			0,
			options.duplicateThreshold ?? DEFAULT_DUPLICATE_THRESHOLD,
		);
		this.failureThreshold = Math.max(
			0,
			options.failureThreshold ?? DEFAULT_FAILURE_THRESHOLD,
		);
	}

	// ── Guard: pre-execution check ────────────────────────────────────────
	/**
	 * Check a tool call before execution. Returns block=true with a message
	 * the loop records instead of executing the tool.
	 */
	checkToolCall(name: string, args: string): LoopGuardDecision {
		const sig = callSignature(name, args);

		// Duplicate guard: only count consecutive identical calls.
		// Reset counter when a different tool or different args is called.
		if (sig === this.lastCallSignature) {
			this.consecutiveCallCount++;
		} else {
			this.lastCallSignature = sig;
			this.consecutiveCallCount = 1;
		}

		if (
			this.duplicateThreshold > 0 &&
			this.consecutiveCallCount >= this.duplicateThreshold
		) {
			return {
				block: true,
				guard: "duplicate",
				message: `Error: [duplicate-guard] blocked — \`${name}\` called with the same (or cosmetically-varied) arguments ${this.consecutiveCallCount} times in a row (threshold: ${this.duplicateThreshold}). Stop repeating the same call; change your approach.`,
			};
		}

		if (this.failureThreshold > 0) {
			const path = callPath(args);
			if ((this.failSignatureCounts.get(sig) || 0) >= this.failureThreshold) {
				return this.tripFailure(name, "the same call");
			}
			if (
				path &&
				(this.failPathCounts.get(path) || 0) >= this.failureThreshold
			) {
				return this.tripFailure(name, `\`${path}\``);
			}
			// Category bucket: distinct-but-equivalent failures collapse
			// to one category. Trip when any category for this tool crosses
			// the threshold.
			for (const [cat, count] of this.failCategoryCounts) {
				if (count >= this.failureThreshold && cat.startsWith(`${name} `)) {
					return this.tripFailure(name, "this kind of operation");
				}
			}
		}

		return { block: false };
	}

	/**
	 * Record a failed tool call. Updates failure counts for the guard layer
	 * and loop detection. Does NOT check thresholds — use checkToolCall() for that.
	 */
	recordFailure(name: string, args: string, result: string): void {
		const sig = callSignature(name, args);
		inc(this.failSignatureCounts, sig);
		const path = callPath(args);
		if (path) inc(this.failPathCounts, path);
		const cat = failureCategory(name, result);
		inc(this.failCategoryCounts, cat);
	}

	/** Successful work is evidence that a previously failing route recovered.
	 * Decay matching failure state so old incidents do not poison the tool or
	 * path for the remainder of a long-running harness session. */
	recordSuccess(name: string, args: string): void {
		this.failSignatureCounts.delete(callSignature(name, args));
		const path = callPath(args);
		if (path) this.failPathCounts.delete(path);
		for (const category of this.failCategoryCounts.keys()) {
			if (category.startsWith(`${name} `)) {
				this.failCategoryCounts.delete(category);
			}
		}
	}

	private tripFailure(toolName: string, target: string): LoopGuardDecision {
		return {
			block: true,
			guard: "failure",
			message: `Error: [failure-guard] blocked — \`${toolName}\` has failed on ${target} ${this.failureThreshold} times (threshold: ${this.failureThreshold}). Stop retrying the same approach; inspect the actual error, fix the root cause, or use a different tool.`,
		};
	}

	// ── Reset ─────────────────────────────────────────────────────────────
	reset(): void {
		this.lastCallSignature = null;
		this.consecutiveCallCount = 0;
		this.failSignatureCounts.clear();
		this.failCategoryCounts.clear();
		this.failPathCounts.clear();
	}
}
