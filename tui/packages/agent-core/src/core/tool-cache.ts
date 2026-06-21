// ── Tool Result Cache ──────────────────────────────────────────────────────
// LRU cache for tool execution results. Skips re-executing identical tool calls
// (same name + same parsed arguments). Only caches successful results.

type Sortable = unknown;

function sortKeys(value: Sortable): Sortable {
	if (Array.isArray(value)) return value.map(sortKeys);
	if (value && typeof value === "object") {
		const sorted = Object.keys(value as Record<string, unknown>).sort();
		const result: Record<string, unknown> = {};
		for (const key of sorted) {
			result[key] = sortKeys((value as Record<string, unknown>)[key]);
		}
		return result;
	}
	return value;
}

function canonicalizeArgs(args: string): string {
	try {
		const parsed = JSON.parse(args);
		return JSON.stringify(sortKeys(parsed));
	} catch {
		// Non-JSON args: use raw string (rare, but safe fallback)
		return args;
	}
}

export interface CacheEntry {
	result: string;
	isError: boolean;
	/** Epoch ms when this entry expires. undefined = no expiry. */
	expiresAt?: number;
}

export class ToolResultCache {
	private cache = new Map<string, CacheEntry>();
	private maxSize: number;
	private defaultTtlMs: number;

	constructor(maxSize = 1000, defaultTtlMs = 30_000) {
		this.maxSize = maxSize;
		this.defaultTtlMs = defaultTtlMs;
	}

	/** Look up a cached result. Returns null if not found, expired, or error. */
	get(toolName: string, args: string): CacheEntry | null {
		const key = `${toolName}::${canonicalizeArgs(args)}`;
		const entry = this.cache.get(key);
		if (!entry) return null;
		// Never serve cached errors — re-execute to detect transient failures
		if (entry.isError) {
			this.cache.delete(key);
			return null;
		}
		// TTL check
		if (entry.expiresAt && Date.now() > entry.expiresAt) {
			this.cache.delete(key);
			return null;
		}
		return entry;
	}

	/** Store a tool result. Skips error results. */
	put(toolName: string, args: string, result: string, isError: boolean): void {
		if (isError) return; // never cache errors
		const key = `${toolName}::${canonicalizeArgs(args)}`;
		if (this.cache.size >= this.maxSize) {
			// Evict oldest entry (first-in = oldest in insertion order)
			const firstKey = this.cache.keys().next().value;
			if (firstKey) this.cache.delete(firstKey);
		}
		this.cache.set(key, {
			result,
			isError,
			expiresAt: Date.now() + this.defaultTtlMs,
		});
	}

	/** Clear the entire cache. */
	clear(): void {
		this.cache.clear();
	}

	/** Current cache size. */
	get size(): number {
		return this.cache.size;
	}
}
