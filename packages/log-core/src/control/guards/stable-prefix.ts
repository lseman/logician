/**
 * Stable prefix — frozen system prompt + tool spec with fingerprinting.
 *
 * When the system prompt and tool specifications haven't changed between
 * turns, we reuse the previous serialized prefix. This prevents token waste
 * from re-serializing identical content on every provider call.
 *
 * The fingerprint is computed from the canonical serialized form of the
 * system prompt and tool definitions. If the fingerprint matches the previous
 * build, the cached prefix is returned; otherwise a new one is built.
 *
 * This mirrors OMP's StablePrefix from packages/agent/src/append-only-context.ts.
 */

import type { Tool } from "../../system/types/types-messages.ts";

/** Options for building a stable prefix. */
export interface StablePrefixOptions {
	/** System prompt text. */
	systemPrompt: string;
	/** Tool definitions available in this turn. */
	tools: Tool[];
	/** Maximum number of turns to retain the cache before forcing rebuild. */
	maxCacheTtl?: number;
}

/** A frozen system prompt + tool spec snapshot. */
export interface StablePrefixSnapshot {
	/** Canonical fingerprint of this prefix. */
	fingerprint: string;
	/** The serialized system prompt text. */
	prompt: string;
	/** Version counter — increments on each rebuild. */
	version: number;
	/** Whether the prefix was rebuilt from live state. */
	wasRebuilt: boolean;
}

/** Simple deterministic hash for prefix fingerprinting. */
function hash(str: string): string {
	let h = 0;
	for (let i = 0; i < str.length; i++) {
		const ch = str.charCodeAt(i);
		h = ((h << 5) - h + ch) | 0;
	}
	return (h >>> 0).toString(36);
}

/** Compute a fingerprint from system prompt and tool names+descriptions. */
function computeFingerprint(systemPrompt: string, tools: Tool[]): string {
	const toolFragments = tools
		.filter(t => t.name && t.description)
		.map(t => `${t.name}:${t.description.slice(0, 200)}`)
		.sort()
		.join("|");
	return hash(`${systemPrompt}\x00${toolFragments}`);
}

export class StablePrefix {
	#snapshot: StablePrefixSnapshot | null = null;
	#version = 0;
	readonly #maxCacheTtl: number;

	constructor(options?: { maxCacheTtl?: number }) {
		this.#maxCacheTtl = options?.maxCacheTtl ?? 50;
	}

	/** Get the fingerprint of the current snapshot, or "unbuilt" if not yet built. */
	get fingerprint(): string {
		return this.#snapshot?.fingerprint ?? "unbuilt";
	}

	/** Get the version counter. */
	get version(): number {
		return this.#version;
	}

	/**
	 * Build or rebuild the prefix from live state.
	 * Returns true if the prefix was rebuilt (fingerprint changed or cache expired).
	 */
	build(systemPrompt: string, tools: Tool[]): boolean {
		const fingerprint = computeFingerprint(systemPrompt, tools);
		const needsRebuild =
			!this.#snapshot ||
			this.#snapshot.fingerprint !== fingerprint ||
			this.#version >= this.#maxCacheTtl;

		if (needsRebuild) {
			this.#snapshot = {
				fingerprint,
				prompt: systemPrompt,
				version: ++this.#version,
				wasRebuilt: true,
			};
			return true;
		}

		// Cache hit — update version without rebuilding
		this.#version++;
		return false;
	}

	/** Invalidate the cache so the next build always rebuilds. */
	invalidate(): void {
		this.#snapshot = null;
	}

	/** Get the current snapshot (or null if not built). */
	get snapshot(): StablePrefixSnapshot | null {
		return this.#snapshot;
	}

	/** Get the cached prompt text, or undefined if not built. */
	getPrompt(): string | undefined {
		return this.#snapshot?.prompt;
	}
}
