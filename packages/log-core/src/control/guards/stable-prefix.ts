/**
 * Stable prefix + append-only log — frozen system prompt + tool spec with
 * fingerprinting, plus an append-only message log for stable byte prefixes.
 *
 * When the system prompt and tool specifications haven't changed between
 * turns, we reuse the previous serialized prefix. This prevents token waste
 * from re-serializing identical content on every provider call.
 *
 * AppendOnlyLog guarantees that the only mutation path is `replaceTail()`,
 * reserved for compaction. Every other operation is append-only. Combined
 * with fingerprint-based prefix caching, this keeps the provider's prompt
 * cache warm up to the divergence point on every turn.
 */

import type { Message, Tool } from "../../system/types/types-messages.ts";

// ---------------------------------------------------------------------------
// StablePrefix (fingerprint-based cache, no TTL)
// ---------------------------------------------------------------------------

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

/**
 * A frozen prefix (system prompt + tools) that produces stable byte
 * sequences across `build()` calls.
 *
 * The first `build()` snapshots the live state. Subsequent calls reuse
 * the cached copy until `invalidate()` is called or the live state's
 * fingerprint changes. No TTL — cache lives until explicit invalidation.
 */
export class StablePrefix {
	#snapshot: StablePrefixSnapshot | null = null;
	#version = 0;

	get fingerprint(): string {
		return this.#snapshot?.fingerprint ?? "unbuilt";
	}
	get version(): number {
		return this.#version;
	}
	get built(): boolean {
		return this.#snapshot !== null;
	}

	/**
	 * Build or rebuild from live state.
	 * Returns `true` if the prefix actually changed (cache miss imminent).
	 */
	build(systemPrompt: string, tools: Tool[]): boolean {
		const fingerprint = computeFingerprint(systemPrompt, tools);
		if (this.#snapshot && this.#snapshot.fingerprint === fingerprint) {
			return false;
		}
		this.#snapshot = {
			fingerprint,
			prompt: systemPrompt,
			version: ++this.#version,
			wasRebuilt: true,
		};
		return true;
	}

	/** Force rebuild on the next `build()` call. */
	invalidate(): void {
		this.#snapshot = null;
	}

	/**
	 * Returns the cached prefix.
	 * @throws if `build()` was never called.
	 */
	toContext(): { systemPrompt: string; tools: Tool[] } {
		const s = this.#snapshot;
		if (!s) throw new Error("StablePrefix.toContext() called before build()");
		return { systemPrompt: s.prompt, tools: [] };
	}
}

// ---------------------------------------------------------------------------
// AppendOnlyLog
// ---------------------------------------------------------------------------

/**
 * Append-only message log at the provider-level message array layer.
 *
 * The only mutation path is `replaceTail()`, reserved for compaction.
 * Every other operation is append-only. This preserves byte-stable
 * prefixes so the provider's prompt-cache stays warm.
 */
export class AppendOnlyLog {
	#entries: Message[] = [];

	get length(): number {
		return this.#entries.length;
	}

	append(message: Message): void {
		this.#entries.push(message);
	}

	extend(messages: Message[]): void {
		for (const m of messages) this.#entries.push(m);
	}

	/** Replace the last entry — only legal for compaction. */
	replaceTail(replacement: Message): void {
		const idx = this.#entries.length - 1;
		if (idx >= 0) this.#entries[idx] = replacement;
	}

	/** Returns a shallow copy of all entries. */
	toMessages(): Message[] {
		return this.#entries.slice();
	}

	/** Direct readonly access for in-place inspection. */
	entries(): readonly Message[] {
		return this.#entries;
	}

	/** Drop entries past index `count`, keeping the first `count` byte-stable. */
	truncate(count: number): void {
		if (count < 0) count = 0;
		if (count >= this.#entries.length) return;
		this.#entries.length = count;
	}

	clear(): void {
		this.#entries = [];
	}
}

// ---------------------------------------------------------------------------
// AppendOnlyContextManager
// ---------------------------------------------------------------------------

/**
 * Manages a stable prefix + append-only log for the agent loop.
 *
 * Call `build(context)` each turn to get a context with stable
 * `systemPrompt` and `tools` and append-only messages. Call
 * `syncMessages(messages)` after converting messages each
 * turn to keep the log in sync.
 *
 * Example:
 * ```
 * const mgr = new AppendOnlyContextManager();
 * const ctx = mgr.build(context);  // first call snapshots prefix
 * mgr.syncMessages(normalized);    // grow the log
 * ctx = mgr.build(context);        // subsequent calls use cache
 * ```
 */
export class AppendOnlyContextManager {
	readonly prefix = new StablePrefix();
	readonly log = new AppendOnlyLog();
	/** How many normalized messages were synced into the log as of the last sync. */
	#lastSyncCount = 0;

	/**
	 * Build context with stable prefix + append-only messages.
	 */
	build(systemPrompt: string, tools: Tool[]): {
		systemPrompt: string;
		tools: Tool[];
		messages: Message[];
	} {
		this.prefix.build(systemPrompt, tools);
		return {
			systemPrompt: this.prefix.toContext().systemPrompt,
			tools: this.prefix.toContext().tools,
			messages: this.log.toMessages(),
		};
	}

	/**
	 * Sync normalized messages into the append-only log.
	 *
	 * Three cases:
	 *
	 * 1. **Append**: same prefix, new tail → push the new entries.
	 * 2. **Compaction**: shorter array → clear the log and replay.
	 * 3. **In-place rewrite**: find the longest byte-stable prefix between
	 *    the previously-synced messages and the new ones, drop the log
	 *    down to that prefix, then append the diverged tail.
	 */
	syncMessages(normalizedMessages: Message[]): void {
		// Compaction (array shrunk) — every previously-synced message is gone,
		// so the log can't carry any byte-stable bytes forward.
		if (normalizedMessages.length < this.#lastSyncCount) {
			this.log.clear();
			this.#lastSyncCount = 0;
		}

		// In-place rewrite: trim the log down to the longest stable prefix
		// that both the previous sync and the new messages share.
		if (this.#lastSyncCount > 0) {
			const stableCount = Math.min(
				this.#lastStablePrefix(normalizedMessages),
				this.log.length,
			);
			if (stableCount < this.#lastSyncCount) {
				this.log.truncate(stableCount);
				this.#lastSyncCount = stableCount;
			}
		}

		// Append the diverged tail (or the full delta on a normal turn).
		for (let i = this.#lastSyncCount; i < normalizedMessages.length; i++) {
			this.log.append(normalizedMessages[i]);
		}
		this.#lastSyncCount = normalizedMessages.length;
	}

	/** Reset prefix + log for a model/provider switch. */
	invalidateForModelChange(): void {
		this.prefix.invalidate();
		this.log.clear();
		this.#lastSyncCount = 0;
	}

	/** Reset the sync cursor AND clear the log. */
	resetSyncCursor(): void {
		this.log.clear();
		this.#lastSyncCount = 0;
	}

	appendMessage(message: Message): void {
		this.log.append(message);
	}

	replaceTailMessage(message: Message): void {
		this.log.replaceTail(message);
	}

	invalidate(): void {
		this.prefix.invalidate();
	}

	/** Index of the first message whose serialized bytes differ from the
	 * previously-synced log. */
	#lastStablePrefix(normalizedMessages: readonly unknown[]): number {
		const bound = Math.min(this.#lastSyncCount, normalizedMessages.length);
		for (let i = 0; i < bound; i++) {
			if (JSON.stringify(normalizedMessages[i]) !==
				JSON.stringify((this.log.entries() as unknown[])[i])) {
				return i;
			}
		}
		return bound;
	}
}
