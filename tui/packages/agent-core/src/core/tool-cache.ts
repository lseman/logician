// ── Tool Result Cache ──────────────────────────────────────────────────────
// LRU cache for tool execution results. Skips re-executing identical tool calls
// (same name + same parsed arguments). Only caches successful results.
//
// Caching is opt-in per tool (Tool.cacheable) — the registry never consults
// this cache for tools with side effects or time-varying output.
//
// Entries can carry an mtime sentinel; the registry passes the current
// sentinel on lookup and a mismatch (file changed on disk) invalidates the
// entry.

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
	/** Mtime sentinel for file-based invalidation. undefined = no mtime key. */
	mtimeKey?: string;
}

export interface CacheStats {
	hits: number;
	misses: number;
	evictions: number;
	hitRate: number;
}

export class ToolResultCache {
	private cache = new Map<string, CacheEntry>();
	private maxSize: number;
	private defaultTtlMs: number;
	private hits = 0;
	private misses = 0;
	private evictions = 0;

	constructor(maxSize = 2000, defaultTtlMs = 60_000) {
		this.maxSize = maxSize;
		this.defaultTtlMs = defaultTtlMs;
	}

	/**
	 * Look up a cached result. Returns null if not found, expired, error, or
	 * stale (stored mtime sentinel differs from `currentMtimeKey`).
	 */
	get(
		toolName: string,
		args: string,
		currentMtimeKey?: string,
	): CacheEntry | null {
		const key = `${toolName}::${canonicalizeArgs(args)}`;
		const entry = this.cache.get(key);
		if (entry) {
			// Never serve cached errors — re-execute to detect transient failures
			if (entry.isError) {
				this.cache.delete(key);
				this.misses++;
				return null;
			}
			// TTL check
			if (entry.expiresAt && Date.now() > entry.expiresAt) {
				this.cache.delete(key);
				this.misses++;
				return null;
			}
			// Mtime check: the underlying file changed since the entry was stored
			if (entry.mtimeKey && entry.mtimeKey !== currentMtimeKey) {
				this.cache.delete(key);
				this.misses++;
				return null;
			}
			this.hits++;
			return entry;
		}

		this.misses++;
		return null;
	}

	/** Store a tool result. Skips error results. */
	put(
		toolName: string,
		args: string,
		result: string,
		isError: boolean,
		mtimeKey?: string,
	): void {
		if (isError) return; // never cache errors
		const key = `${toolName}::${canonicalizeArgs(args)}`;
		if (this.cache.size >= this.maxSize) {
			// Evict oldest entry (first-in = oldest in insertion order)
			const firstKey = this.cache.keys().next().value;
			if (firstKey) {
				this.cache.delete(firstKey);
				this.evictions++;
			}
		}
		this.cache.set(key, {
			result,
			isError: false,
			expiresAt: Date.now() + this.defaultTtlMs,
			mtimeKey,
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

	/** Get cache performance statistics. */
	stats(): CacheStats {
		const total = this.hits + this.misses;
		return {
			hits: this.hits,
			misses: this.misses,
			evictions: this.evictions,
			hitRate: total > 0 ? (this.hits / total) * 100 : 0,
		};
	}

	/** Reset statistics counters. */
	resetStats(): void {
		this.hits = 0;
		this.misses = 0;
		this.evictions = 0;
	}
}
