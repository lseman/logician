// ── Loop Detection + Tool Guards ───────────────────────────────────────────
// Unified loop detection and guard engine. Merged from two separate systems:
//
// Guard layer (pre-execution): blocks individual tool calls that are:
//   - Duplicates (same tool+args called N times)
//   - Repeated failures (same tool/path/category failed N times)
//
// Turn layer (post-execution): detects if the agent is stuck in a
// repetitive or degenerate loop via three strategies running in parallel;
// any one triggers termination:
//
// 1. **Exact repeat** — the last N turns are (near-)identical. Fast path for
//    obvious infinite loops.
//
// 2. **Degenerate / circular** — the agent calls the same tools in the same
//    order over and over, with only minor arg variations, and gets the same
//    results. The agent is technically "productive" (tool calls every turn) but
//    making zero forward progress.
//
// 3. **Stagnation** — the agent keeps calling tools without making real progress.
//    Tracks the set of "new things" (distinct tool+result shapes) and flags the
//    turn when that set stops growing for a configurable window.
//
// Configuration is driven by AgentConfig fields so the harness can tune behaviour
// without touching the detector code.

export interface TurnSignature {
	assistantContent: string;
	toolCalls: Array<{
		name: string;
		args: string;
		result: string;
	}>;
}

// Fingerprint of a single tool call: name + normalized arg hash + result prefix.
// Used for degenerate-loop detection (same shape, different args).
interface ToolFingerprint {
	name: string;
	argHash: string;
	resultPrefix: string;
}

export interface GuardDecision {
	block: boolean;
	message?: string;
}

export interface LoopDetectorOptions {
	/** Rolling history kept for analysis (default 10). */
	maxHistory?: number;
	/** Consecutive identical turns to trigger exact-repeat (default 3). */
	exactRepeatWindow?: number;
	/** Consecutive turns with the same tool-name sequence to flag (default 4). */
	degenerateWindow?: number;
	/** Consecutive turns with zero new signal to flag (default 5). */
	stagnationWindow?: number;
	/** Duplicate call threshold — block when same tool+args called N times (default 3). */
	duplicateThreshold?: number;
	/** Failure loop threshold — block when same tool/path/category failed N times (default 3). */
	failureThreshold?: number;
}

const DEFAULT_DUPLICATE_THRESHOLD = 3;
const DEFAULT_FAILURE_THRESHOLD = 3;
const MAX_CATEGORY_LEN = 120;

// Stable signature for a tool call: name + canonical args.
function callSignature(name: string, args: string): string {
	let argsKey = args || "";
	try {
		argsKey = JSON.stringify(sortKeys(JSON.parse(args || "{}")));
	} catch (_e: unknown) {
		// Non-JSON args: use the raw string.
	}
	return `${name} ${argsKey}`;
}

function sortKeys(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(sortKeys);
	if (value && typeof value === "object") {
		const out: Record<string, unknown> = {};
		for (const k of Object.keys(value as Record<string, unknown>).sort()) {
			out[k] = sortKeys((value as Record<string, unknown>)[k]);
		}
		return out;
	}
	return value;
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
function failureCategory(toolName: string, result: string): string {
	const body = result.replace(/^Error:\s*/i, "").trim();
	return `${toolName} ${body.slice(0, MAX_CATEGORY_LEN)}`;
}

function inc(map: Map<string, number>, key: string): number {
	const next = (map.get(key) || 0) + 1;
	map.set(key, next);
	return next;
}

export class LoopDetector {
	private history: Array<{
		signature: string; // exact-repeat key
		toolFingerprints: ToolFingerprint[];
		toolNames: string[]; // just the sequence of names
		contentDirection: string; // first ~80 chars, normalized
	}> = [];

	// ── Guard state ───────────────────────────────────────────────────────
	// Duplicate guard: only counts CONSECUTIVE identical tool+args calls.
	// Reset when a different tool or different args is called.
	private lastCallSignature: string | null = null;
	private consecutiveCallCount = 0;
	private failSignatureCounts = new Map<string, number>();
	private failCategoryCounts = new Map<string, number>();
	private failPathCounts = new Map<string, number>();

	// ── Turn layer state ──────────────────────────────────────────────────
	private readonly maxHistory: number;
	private readonly exactRepeatWindow: number;
	private readonly degenerateWindow: number;
	private readonly stagnationWindow: number;
	private readonly duplicateThreshold: number;
	private readonly failureThreshold: number;
	private readonly seenShapes = new Set<string>();

	constructor(options: LoopDetectorOptions = {}) {
		this.maxHistory = options.maxHistory ?? 10;
		this.exactRepeatWindow = options.exactRepeatWindow ?? 3;
		this.degenerateWindow = options.degenerateWindow ?? 4;
		this.stagnationWindow = options.stagnationWindow ?? 5;
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
	checkToolCall(name: string, args: string): GuardDecision {
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
				message: `Error: blocked — \`${name}\` was called with identical arguments ${this.consecutiveCallCount} times in a row. Stop repeating the same call; change your approach.`,
			};
		}

		if (this.failureThreshold > 0) {
			const path = callPath(args);
			if ((this.failSignatureCounts.get(sig) || 0) >= this.failureThreshold) {
				return this.tripFailure(name, "the same call");
			}
			if (path && (this.failPathCounts.get(path) || 0) >= this.failureThreshold) {
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

	private tripFailure(toolName: string, target: string): GuardDecision {
		return {
			block: true,
			message: `Error: blocked — \`${toolName}\` has failed on ${target} ${this.failureThreshold} times. Stop retrying the same approach; inspect the actual error, fix the root cause, or use a different tool.`,
		};
	}

	// ── Record a turn + run all detections ─────────────────────────────────
	/**
	 * Record a turn and check for loops. Returns true if a loop is detected.
	 * Guard checks happen in the beforeToolCall hook; this only does
	 * turn-level analysis (exact repeat, degenerate, stagnation).
	 */
	recordAndDetect(
		assistantContent: string,
		toolCalls: Array<{ name: string; args: string; result: string }>,
	): boolean {
		// Snapshot shapes before adding the current turn.
		const shapesBefore = new Set(this.seenShapes);

		// Build fingerprint and add to history.
		const fingerprint = this.buildFingerprint(assistantContent, toolCalls);
		const entry = {
			signature: fingerprint.signature,
			toolFingerprints: fingerprint.fingerprints,
			toolNames: toolCalls.map((tc) => tc.name),
			contentDirection: fingerprint.contentDirection,
		};
		this.history.push(entry);
		if (this.history.length > this.maxHistory) {
			this.history.shift();
		}

		// Accumulate shapes for future stagnation checks.
		this.updateSeenShapes(fingerprint.fingerprints);

		return this.isLoopingWithShapesBefore(shapesBefore);
	}

	// ── Fingerprint builder ───────────────────────────────────────────────
	private buildFingerprint(
		assistantContent: string,
		toolCalls: Array<{ name: string; args: string; result: string }>,
	): {
		signature: string;
		fingerprints: ToolFingerprint[];
		contentDirection: string;
	} {
		const contentSnippet = assistantContent
			.toLowerCase()
			.slice(0, 200)
			.replace(/\s+/g, " ")
			.trim();
		const toolSnippet = toolCalls
			.map(
				(tc) =>
					`${tc.name}:${tc.args.toLowerCase().slice(0, 80)}:${tc.result.toLowerCase().slice(0, 80)}`,
			)
			.join("|");
		const signature = `${contentSnippet}||${toolSnippet}`;

		const fingerprints = toolCalls.map((tc) => ({
			name: tc.name,
			argHash: this.hashArgs(tc.args),
			// Prefix alone collapses genuinely different results that happen to
			// share the same opening words (e.g. every successful edit_file
			// confirmation, or similarly-headered file reads). Bucket in the
			// result length too so distinct-sized results don't fingerprint as
			// the same "shape".
			resultPrefix: `${this.lengthBucket(tc.result.length)}:${tc.result
				.toLowerCase()
				.slice(0, 60)
				.replace(/\s+/g, " ")
				.trim()}`,
		}));

		const contentDirection = assistantContent
			.trim()
			.split(/\s+/)
			.slice(0, 12)
			.join(" ")
			.toLowerCase()
			.slice(0, 80);

		return { signature, fingerprints, contentDirection };
	}

	// Coarse log2 bucket so results of similar size fingerprint the same but
	// substantially different-sized results (e.g. a 40-char confirmation vs a
	// 4000-char file dump) do not.
	private lengthBucket(length: number): number {
		return length <= 0 ? 0 : Math.floor(Math.log2(length + 1));
	}

	private hashArgs(args: string): string {
		try {
			const parsed = JSON.parse(args);
			if (typeof parsed !== "object" || parsed === null) {
				return typeof parsed;
			}
			const parts = Object.entries(parsed)
				.map(([k, v]) => `${k}:${typeof v}`)
				.sort()
				.join(",");
			return `{${parts}}`;
		} catch (_e: unknown) {
			return "malformed";
		}
	}

	// ── Turn-level loop detection ─────────────────────────────────────────
	private isLooping(): boolean {
		return this.isLoopingWithShapesBefore(new Set(this.seenShapes));
	}

	private isLoopingWithShapesBefore(shapesBefore: Set<string>): boolean {
		return (
			this.isExactRepeat() ||
			this.isDegenerateLoop() ||
			this.isStagnatingWith(shapesBefore)
		);
	}

	private isExactRepeat(): boolean {
		if (this.history.length < this.exactRepeatWindow) return false;
		const window = this.history.slice(-this.exactRepeatWindow);
		return window.every((h) => h.signature === window[0].signature);
	}

	private isDegenerateLoop(): boolean {
		if (this.history.length < this.degenerateWindow) return false;
		const window = this.history.slice(-this.degenerateWindow);

		const firstNames = window[0].toolNames.join(",");
		if (!window.every((h) => h.toolNames.join(",") === firstNames)) {
			return false;
		}
		if (!firstNames) return false;

		const firstFps = new Set(
			window[0].toolFingerprints.map((fp) => `${fp.name}:${fp.resultPrefix}`),
		);
		const allSameShape = window.every((h) =>
			h.toolFingerprints.every((fp) =>
				firstFps.has(`${fp.name}:${fp.resultPrefix}`),
			),
		);
		if (!allSameShape) return false;

		return true;
	}

	private isStagnatingWith(shapesBefore: Set<string>): boolean {
		if (this.history.length < this.stagnationWindow) return false;
		const window = this.history.slice(-this.stagnationWindow);

		const hasTools = window.some((h) => h.toolNames.length > 0);
		if (!hasTools) return false;

		let anyNew = false;
		for (const entry of window) {
			for (const fp of entry.toolFingerprints) {
				const shapeKey = `${fp.name}:${fp.resultPrefix}`;
				if (!shapesBefore.has(shapeKey)) {
					anyNew = true;
					break;
				}
			}
			if (anyNew) break;
		}

		for (const entry of window) {
			for (const fp of entry.toolFingerprints) {
				this.seenShapes.add(`${fp.name}:${fp.resultPrefix}`);
			}
		}

		return !anyNew;
	}

	private updateSeenShapes(fingerprints: ToolFingerprint[]): void {
		for (const fp of fingerprints) {
			this.seenShapes.add(`${fp.name}:${fp.resultPrefix}`);
		}
	}

	// ── Consume a turn (advance the turn counter) ─────────────────────────
	consumeTurn(): void {
		// No-op on the detector itself; the harness tracks turn count.
		// This method exists for the harness to signal turn completion.
	}

	// ── Reset ─────────────────────────────────────────────────────────────
	reset(): void {
		this.history = [];
		this.seenShapes.clear();
		this.lastCallSignature = null;
		this.consecutiveCallCount = 0;
		this.failSignatureCounts.clear();
		this.failCategoryCounts.clear();
		this.failPathCounts.clear();
	}

	// ── Diagnostics ───────────────────────────────────────────────────────
	getLoopDiagnostic(): string | null {
		if (this.history.length < 2) return null;

		if (this.isExactRepeat()) {
			const window = this.history.slice(-this.exactRepeatWindow);
			const first = window[0];
			const toolSeq = first.toolNames.join(", ");
			const snippet = first.contentDirection;
			return (
				`Exact repeat detected: the last ${this.exactRepeatWindow} turns are identical. ` +
				`You keep saying "${snippet}…" and calling: ${toolSeq}. ` +
				"This is a dead loop — the same input produces the same output every time."
			);
		}

		if (this.isDegenerateLoop()) {
			const window = this.history.slice(-this.degenerateWindow);
			const first = window[0];
			const toolSeq = first.toolNames.join(", ");
			const resultPrefixes = new Set(
				first.toolFingerprints.map((fp) => fp.resultPrefix),
			);
			const results = Array.from(resultPrefixes).slice(0, 3).join("; ");
			return (
				`Degenerate loop detected: ${this.degenerateWindow} turns in a row calling the same tools ` +
				`(${toolSeq}) and getting the same results (${results}). ` +
				"You may be varying arguments but the outcome is unchanged."
			);
		}

		if (this.isStagnating()) {
			const window = this.history.slice(-this.stagnationWindow);
			const toolNames = new Set(window.flatMap((h) => h.toolNames));
			const shapes = Array.from(this.seenShapes).slice(0, 5).join(", ");
			return (
				`Stagnation detected: ${this.stagnationWindow} turns with no new signal. ` +
				`You've been calling: ${Array.from(toolNames).join(", ")}. ` +
				`All results fall into known shapes: ${shapes}. ` +
				"You are not making progress on the task."
			);
		}

		return null;
	}

	private isStagnating(): boolean {
		return this.isStagnatingWith(new Set(this.seenShapes));
	}
}
