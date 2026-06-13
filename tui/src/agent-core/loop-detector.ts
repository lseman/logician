// ── Loop Detection ──────────────────────────────────────────────────────────
// Detects if the agent is stuck in a repetitive or degenerate loop.
//
// Three detection strategies run in parallel; any one triggers termination:
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

export class LoopDetector {
	private history: Array<{
		signature: string; // exact-repeat key
		toolFingerprints: ToolFingerprint[];
		toolNames: string[]; // just the sequence of names
		contentDirection: string; // first ~80 chars, normalized
	}> = [];

	// ── Configuration ───────────────────────────────────────────────────────
	private readonly maxHistory: number;
	private readonly exactRepeatWindow: number;
	private readonly degenerateWindow: number;
	private readonly stagnationWindow: number;

	// Distinct tool+result shape keys seen so far (for stagnation).
	private readonly seenShapes = new Set<string>();

	constructor(
		config: {
			/** Rolling history kept for analysis (default 10). */
			maxHistory?: number;
			/** Consecutive identical turns to trigger exact-repeat (default 3). */
			exactRepeatWindow?: number;
			/** Consecutive turns with the same tool-name sequence to flag (default 4). */
			degenerateWindow?: number;
			/** Consecutive turns with zero new signal to flag (default 5). */
			stagnationWindow?: number;
		} = {},
	) {
		this.maxHistory = config.maxHistory ?? 10;
		this.exactRepeatWindow = config.exactRepeatWindow ?? 3;
		this.degenerateWindow = config.degenerateWindow ?? 4;
		this.stagnationWindow = config.stagnationWindow ?? 5;
	}

	/**
	 * Record a turn and check for loops. Returns true if a loop is detected.
	 */
	recordAndDetect(
		assistantContent: string,
		toolCalls: Array<{ name: string; args: string; result: string }>,
	): boolean {
		const fingerprint = this.buildFingerprint(assistantContent, toolCalls);

		// Snapshot shapes before adding the current turn — so stagnation can
		// compare the window against what was known before this turn.
		const shapesBefore = new Set(this.seenShapes);

		// Add current turn to history.
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

		// Accumulate shapes after the history push (so they're available for
		// future turns but not included in the current turn's stagnation check).
		this.updateSeenShapes(fingerprint.fingerprints);

		// Override isStagnating to use the pre-add snapshot.
		return this.isLoopingWithShapesBefore(shapesBefore);
	}

	// ── Fingerprint builder ─────────────────────────────────────────────────
	private buildFingerprint(
		assistantContent: string,
		toolCalls: Array<{ name: string; args: string; result: string }>,
	): {
		signature: string;
		fingerprints: ToolFingerprint[];
		contentDirection: string;
	} {
		// Exact-repeat signature: normalized content + tool details.
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

		// Per-tool fingerprints: shape-agnostic, used for degenerate detection.
		const fingerprints = toolCalls.map((tc) => ({
			name: tc.name,
			// Hash the argument structure (not value) to detect "same tool,
			// different args" pattern without being confused by legitimate changes.
			argHash: this.hashArgs(tc.args),
			// Result prefix captures the *kind* of result, not its content.
			resultPrefix: tc.result.toLowerCase().slice(0, 60).replace(/\s+/g, " ").trim(),
		}));

		// Content direction: the first meaningful phrase of the assistant's text.
		const contentDirection = assistantContent
			.trim()
			.split(/\s+/)
			.slice(0, 12)
			.join(" ")
			.toLowerCase()
			.slice(0, 80);

		return { signature, fingerprints, contentDirection };
	}

	// Lightweight hash of argument structure: extract keys and types, not values.
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
		} catch {
			return "malformed";
		}
	}

	// ── Loop detection (three strategies) ───────────────────────────────────
	private isLooping(): boolean {
		return this.isLoopingWithShapesBefore(new Set(this.seenShapes));
	}

	/** Same as isLooping but with a pre-computed shapes snapshot for stagnation. */
	private isLoopingWithShapesBefore(
		shapesBefore: Set<string>,
	): boolean {
		return (
			this.isExactRepeat() ||
			this.isDegenerateLoop() ||
			this.isStagnatingWith(shapesBefore)
		);
	}

	/**
	 * Strategy 1 — exact repeat: the last N turns are identical (or nearly so).
	 * Fast path for obvious infinite loops.
	 */
	private isExactRepeat(): boolean {
		if (this.history.length < this.exactRepeatWindow) return false;
		const window = this.history.slice(-this.exactRepeatWindow);
		return window.every((h) => h.signature === window[0].signature);
	}

	/**
	 * Strategy 2 — degenerate / circular: the agent calls the same tools in the
	 * same order repeatedly and gets the same result prefixes. The agent is
	 * "productive" but going nowhere.
	 *
	 * Detection: for each window of `degenerateWindow` turns, check whether the
	 * tool-name sequence is the same AND the result prefixes are the same.
	 * The exact-repeat detector (strategy 1) already catches truly identical
	 * turns, so we don't need to check arg structure here.
	 */
	private isDegenerateLoop(): boolean {
		if (this.history.length < this.degenerateWindow) return false;
		const window = this.history.slice(-this.degenerateWindow);

		// Check that every turn in the window has the same tool-name sequence.
		const firstNames = window[0].toolNames.join(",");
		if (!window.every((h) => h.toolNames.join(",") === firstNames)) {
			return false;
		}

		// The sequence must contain at least one tool call.
		if (!firstNames) return false;

		// Check that tool fingerprints share the same (name, resultPrefix) pairs
		// within the window.
		const firstFps = new Set(
			window[0].toolFingerprints.map((fp) => `${fp.name}:${fp.resultPrefix}`),
		);
		const allSameShape = window.every((h) =>
			h.toolFingerprints.every((fp) => firstFps.has(`${fp.name}:${fp.resultPrefix}`)),
		);
		if (!allSameShape) return false;

		return true;
	}

	/**
	 * Strategy 3 — stagnation: the agent keeps calling tools but introduces no
	 * new "signal" (distinct (name:resultPrefix) shapes) across a configurable
	 * window.
	 *
	 * @param shapesBefore — snapshot of seenShapes BEFORE the current turn was
	 *   added. Allows the check to correctly identify shapes that are genuinely
	 *   new (introduced by the current turn) vs already known.
	 */
	private isStagnatingWith(shapesBefore: Set<string>): boolean {
		if (this.history.length < this.stagnationWindow) return false;
		const window = this.history.slice(-this.stagnationWindow);

		// The window must contain at least one turn with tool calls.
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

		// Always accumulate all shapes from the window for future checks.
		for (const entry of window) {
			for (const fp of entry.toolFingerprints) {
				this.seenShapes.add(`${fp.name}:${fp.resultPrefix}`);
			}
		}

		return !anyNew;
	}

	// Public helper: always update seenShapes for a turn (called from recordAndDetect
	// so shapes accumulate even when the window check early-returns).
	private updateSeenShapes(fingerprints: ToolFingerprint[]): void {
		for (const fp of fingerprints) {
			this.seenShapes.add(`${fp.name}:${fp.resultPrefix}`);
		}
	}

	/**
	 * Reset the detector state (called between independent runs).
	 */
	reset(): void {
		this.history = [];
		this.seenShapes.clear();
	}

	/**
	 * Return a diagnostic string describing the detected loop type and
	 * evidence. Used to build a targeted recovery message instead of a
	 * generic "you're stuck" nudge.
	 */
	getLoopDiagnostic(): string | null {
		if (this.history.length < 2) return null;

		// 1. Exact repeat — the last N turns are (near-)identical.
		if (this.isExactRepeat()) {
			const window = this.history.slice(-this.exactRepeatWindow);
			const first = window[0];
			const toolSeq = first.toolNames.join(", ");
			const snippet = first.contentDirection;
			return (
				`Exact repeat detected: the last ${this.exactRepeatWindow} turns are identical. ` +
				`You keep saying "${snippet}…" and calling: ${toolSeq}. ` +
				`This is a dead loop — the same input produces the same output every time.`
			);
		}

		// 2. Degenerate / circular — same tools, same results, different args.
		if (this.isDegenerateLoop()) {
			const window = this.history.slice(-this.degenerateWindow);
			const first = window[0];
			const toolSeq = first.toolNames.join(", ");
			// Extract unique result prefixes to show what the agent keeps getting.
			const resultPrefixes = new Set(
				first.toolFingerprints.map((fp) => fp.resultPrefix),
			);
			const results = Array.from(resultPrefixes).slice(0, 3).join("; ");
			return (
				`Degenerate loop detected: ${this.degenerateWindow} turns in a row calling the same tools ` +
				`(${toolSeq}) and getting the same results (${results}). ` +
				`You may be varying arguments but the outcome is unchanged.`
			);
		}

		// 3. Stagnation — no new signal across the window.
		if (this.isStagnating()) {
			const window = this.history.slice(-this.stagnationWindow);
			const toolNames = new Set(
				window.flatMap((h) => h.toolNames),
			);
			const shapes = Array.from(this.seenShapes).slice(0, 5).join(", ");
			return (
				`Stagnation detected: ${this.stagnationWindow} turns with no new signal. ` +
				`You've been calling: ${Array.from(toolNames).join(", ")}. ` +
				`All results fall into known shapes: ${shapes}. ` +
				`You are not making progress on the task.`
			);
		}

		return null;
	}

	/**
	 * Convenience wrapper for `getLoopDiagnostic` — checks stagnation against the
	 * current `seenShapes` snapshot (no pre-computed shapesBefore needed).
	 * Delegates to isStagnatingWith; the recordAndDetect path uses the
	 * shapesBefore snapshot for accuracy (before the current turn's shapes
	 * are added to seenShapes).
	 */
	private isStagnating(): boolean {
		return this.isStagnatingWith(new Set(this.seenShapes));
	}
}
