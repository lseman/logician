// ── Tool Result Cache ──────────────────────────────────────────────────────
// LRU cache for tool execution results. Skips re-executing identical tool calls
// (same name + same parsed arguments). Only caches successful results.
//
// Supports mtime-based invalidation: when a file changes on disk, all cached
// results that depend on that file are automatically invalidated.
//
// Supports semantic caching: for read operations, uses content fingerprinting
// to detect similar file content and reuse cached results.

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

// ── Semantic Cache for Read Operations ──────────────────────────────────────
// Simple content fingerprinting using rolling hash + n-gram signature.
// Used for read_file and similar operations to detect similar content.

export function computeContentFingerprint(content: string): string {
	if (content.length === 0) return "empty";
	// Normalize: strip trailing whitespace per line, normalize newlines
	const normalized = content
		.split("\n")
		.map((line) => line.trimEnd())
		.join("\n");
	// Compute simple hash of first/last 500 chars + length
	const head = normalized.slice(0, 500);
	const tail = normalized.slice(Math.max(0, normalized.length - 500));
	const len = normalized.length;
	// Simple hash function
	let hash = 0;
	const combined = `${len}:${head}:${tail}`;
	for (let i = 0; i < combined.length; i++) {
		const char = combined.charCodeAt(i);
		hash = ((hash << 5) - hash + char) | 0;
	}
	return `fp:${Math.abs(hash).toString(16).padStart(8, "0")}:${len}`;
}

function computeSimilarityScore(fp1: string, fp2: string): number {
	// Extract length from fingerprints
	const match1 = fp1.match(/fp:[a-f0-9]{8}:(\d+)/);
	const match2 = fp2.match(/fp:[a-f0-9]{8}:(\d+)/);
	if (!match1 || !match2) return 0;
	const len1 = parseInt(match1[1], 10);
	const len2 = parseInt(match2[1], 10);
	if (len1 === 0 || len2 === 0) return 0;
	// Jaccard-like similarity based on length ratio
	const minLen = Math.min(len1, len2);
	const maxLen = Math.max(len1, len2);
	return minLen / maxLen;
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
	semanticHits: number;
}

export class ToolResultCache {
	private cache = new Map<string, CacheEntry>();
	private semanticCache = new Map<
		string,
		{ fingerprint: string; key: string }[]
	>();
	private maxSize: number;
	private defaultTtlMs: number;
	private hits = 0;
	private misses = 0;
	private evictions = 0;
	private semanticHits = 0;

	constructor(maxSize = 2000, defaultTtlMs = 60_000) {
		this.maxSize = maxSize;
		this.defaultTtlMs = defaultTtlMs;
	}

	/** Look up a cached result. Returns null if not found, expired, or error. */
	get(
		toolName: string,
		args: string,
		contentFingerprint?: string,
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
			this.hits++;
			return entry;
		}

		// Semantic cache fallback for read operations
		if (toolName === "read_file" && contentFingerprint) {
			const candidates = this.semanticCache.get(toolName) || [];
			for (const candidate of candidates) {
				const sim = computeSimilarityScore(
					contentFingerprint,
					candidate.fingerprint,
				);
				if (sim >= 0.85) {
					const entry = this.cache.get(candidate.key);
					if (
						entry &&
						!entry.isError &&
						!(entry.expiresAt && Date.now() > entry.expiresAt)
					) {
						this.semanticHits++;
						return entry;
					}
				}
			}
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
		contentFingerprint?: string,
	): void {
		if (isError) return; // never cache errors
		const key = `${toolName}::${canonicalizeArgs(args)}`;
		if (this.cache.size >= this.maxSize) {
			// Evict oldest entry (first-in = oldest in insertion order)
			const firstKey = this.cache.keys().next().value;
			if (firstKey) {
				this.cache.delete(firstKey);
				this.evictions++;
				// Also remove from semantic cache
				const semCache = this.semanticCache.get(firstKey.split("::")[0]);
				if (semCache) {
					const idx = semCache.findIndex((c) => c.key === firstKey);
					if (idx !== -1) semCache.splice(idx, 1);
				}
			}
		}
		this.cache.set(key, {
			result,
			isError: false,
			expiresAt: Date.now() + this.defaultTtlMs,
			mtimeKey,
		});
		// Index for semantic cache if read operation
		if (toolName === "read_file" && contentFingerprint) {
			if (!this.semanticCache.has(toolName)) {
				this.semanticCache.set(toolName, []);
			}
			const entries = this.semanticCache.get(toolName)!;
			entries.push({ fingerprint: contentFingerprint, key });
			// Keep semantic cache size bounded
			if (entries.length > this.maxSize / 2) {
				entries.shift();
			}
		}
	}

	/** Invalidate all entries that share the same mtime sentinel. */
	invalidByMtime(mtimeKey: string): number {
		let invalidated = 0;
		for (const [key, entry] of this.cache) {
			if (entry.mtimeKey === mtimeKey) {
				this.cache.delete(key);
				invalidated++;
			}
		}
		return invalidated;
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
			semanticHits: this.semanticHits,
		};
	}

	/** Reset statistics counters. */
	resetStats(): void {
		this.hits = 0;
		this.misses = 0;
		this.evictions = 0;
		this.semanticHits = 0;
	}
}
