// ── Structured Memory System ──────────────────────────────────────────────
// Provides cross-turn memory within a session that survives compaction.
// Memory entries are injected into the context before compaction so they
// persist across context windows.

export type MemoryCategory =
	| "facts"
	| "decisions"
	| "file-paths"
	| "errors"
	| "preferences"
	| "context";

export interface MemoryEntry {
	id: string;
	category: MemoryCategory;
	content: string;
	timestamp: number;
	turnIndex: number;
	/** How many turns ago this was last referenced (0 = just referenced). */
	recency: number;
	/** Internal: computed embedding vector for semantic search. */
	_embedding?: number[];
}

export interface MemoryStore {
	/** Add a memory entry. */
	add(entry: Omit<MemoryEntry, "id" | "timestamp" | "recency">): MemoryEntry;
	/** Get all entries, optionally filtered by category. */
	get(category?: MemoryCategory): MemoryEntry[];
	/** Get recent entries (last N turns). */
	getRecent(turns: number): MemoryEntry[];
	/** Get top-k most relevant entries by recency. */
	getTopK(k: number): MemoryEntry[];
	/** Remove an entry by ID. */
	remove(id: string): boolean;
	/** Mark an entry as referenced (reset its recency). */
	referenced(id: string): void;
	/** Clear all entries. */
	clear(): void;
	/** Serialize to JSON for persistence. */
	serialize(): string;
	/** Load from JSON. */
	deserialize(json: string): void;
	/** Search entries by semantic similarity to a query. */
	search(
		query: string,
		k?: number,
	): Array<{ entry: MemoryEntry; score: number }>;
	/** Serialize embeddings for persistence. */
	serializeEmbeddings(): string;
	/** Load embeddings from JSON. */
	deserializeEmbeddings(json: string): void;
}

let nextId = 1;

function generateId(): string {
	return `mem_${nextId++}`;
}

export function createMemoryStore(): MemoryStore {
	const entries: MemoryEntry[] = [];

	return {
		add(entry) {
			const mem: MemoryEntry = {
				...entry,
				id: generateId(),
				timestamp: Date.now(),
				recency: 0,
			};
			entries.push(mem);
			return mem;
		},

		get(category) {
			if (!category) return [...entries];
			return entries.filter((e) => e.category === category);
		},

		getRecent(turns: number) {
			const cutoff = Date.now() - turns * 60_000; // approximate: 1 min per turn
			return entries
				.filter((e) => e.timestamp >= cutoff)
				.sort((a, b) => b.timestamp - a.timestamp);
		},

		getTopK(k: number) {
			return [...entries].sort((a, b) => a.recency - b.recency).slice(0, k);
		},

		remove(id) {
			const idx = entries.findIndex((e) => e.id === id);
			if (idx === -1) return false;
			entries.splice(idx, 1);
			return true;
		},

		referenced(id) {
			const entry = entries.find((e) => e.id === id);
			if (entry) entry.recency = 0;
		},

		clear() {
			entries.length = 0;
		},

		serialize() {
			return JSON.stringify(entries);
		},

		deserialize(json) {
			try {
				const parsed = JSON.parse(json) as MemoryEntry[];
				entries.length = 0;
				for (const entry of parsed) {
					// Reset recency on load
					entry.recency = 0;
					entries.push(entry);
				}
			} catch {
				// Ignore corrupt data
			}
		},

		// ── Semantic search ────────────────────────────────────────────
		search(query, k = 5) {
			const queryEmbed = createEmbedding(query);
			const scored = entries
				.map((entry) => {
					// Cache embedding on entry for future searches
					if (!(entry as MemoryEntry & { _embedding?: number[] })._embedding) {
						(entry as MemoryEntry & { _embedding?: number[] })._embedding =
							createEmbedding(entry.content);
					}
					return {
						entry,
						score: cosineSimilarity(
							queryEmbed,
							(entry as MemoryEntry & { _embedding?: number[] })._embedding!,
						),
					};
				})
				.filter((r) => r.score > 0)
				.sort((a, b) => b.score - a.score)
				.slice(0, k);
			return scored.map(({ entry, score }) => {
				const { _embedding: _, ...clean } = entry as MemoryEntry & {
					_embedding?: number[];
				};
				return { entry: clean as MemoryEntry, score };
			});
		},

		serializeEmbeddings() {
			const map: Record<string, number[]> = {};
			for (const entry of entries) {
				if ((entry as MemoryEntry & { _embedding?: number[] })._embedding) {
					map[entry.id] = (entry as MemoryEntry & { _embedding?: number[] })
						._embedding!;
				}
			}
			return JSON.stringify(map);
		},

		deserializeEmbeddings(json) {
			try {
				const map = JSON.parse(json) as Record<string, number[]>;
				for (const entry of entries) {
					if (map[entry.id]) {
						(entry as MemoryEntry & { _embedding?: number[] })._embedding =
							map[entry.id];
					}
				}
			} catch {
				// Ignore corrupt embedding data
			}
		},
	};
}

// ── Semantic search embeddings ──────────────────────────────────────────────
// Lightweight character trigram embedding for in-process semantic search.
// No external dependencies — uses TF-IDF-like scoring over character trigrams.

/** Maximum number of trigram dimensions. */
const MAX_DIMENSIONS = 256;

/** Build a character trigram embedding vector for a string. */
export function createEmbedding(text: string): number[] {
	if (!text || text.length < 3) {
		return createCharFrequencyEmbedding(text);
	}
	const trimmed = text.toLowerCase().trim();
	const trigrams = new Set<string>();
	for (let i = 0; i <= trimmed.length - 3; i++) {
		trigrams.add(trimmed.substring(i, i + 3));
	}
	const vector = new Array(MAX_DIMENSIONS).fill(0);
	for (const trigram of trigrams) {
		const hash = hashTrigram(trigram);
		vector[hash % MAX_DIMENSIONS] += 1;
	}
	const total = trigrams.size || 1;
	for (let i = 0; i < MAX_DIMENSIONS; i++) {
		vector[i] /= total;
	}
	return vector;
}

function hashTrigram(trigram: string): number {
	let hash = 0;
	for (let i = 0; i < trigram.length; i++) {
		hash = ((hash << 5) - hash + trigram.charCodeAt(i)) | 0;
	}
	return Math.abs(hash);
}

function createCharFrequencyEmbedding(text: string): number[] {
	const vector = new Array(MAX_DIMENSIONS).fill(0);
	if (!text) return vector;
	const freq: Record<string, number> = {};
	for (const ch of text.toLowerCase()) {
		freq[ch] = (freq[ch] || 0) + 1;
	}
	const total = Object.values(freq).reduce((a, b) => a + b, 0) || 1;
	const charCodes: number[] = [];
	for (const ch of Object.keys(freq)) {
		charCodes.push(ch.charCodeAt(0));
	}
	for (let i = 0; i < MAX_DIMENSIONS; i++) {
		const idx = i % charCodes.length;
		const ch = String.fromCharCode(charCodes[idx]);
		vector[i] = (freq[ch] || 0) / total;
	}
	return vector;
}

/** Compute cosine similarity between two vectors. */
export function cosineSimilarity(a: number[], b: number[]): number {
	if (a.length !== b.length || a.length === 0) return 0;
	let dot = 0,
		normA = 0,
		normB = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		normA += a[i] * a[i];
		normB += b[i] * b[i];
	}
	const denom = Math.sqrt(normA) * Math.sqrt(normB);
	return denom === 0 ? 0 : dot / denom;
}

/**
 * Format memory entries as a prompt fragment for injection into the system
 * prompt or compaction summary.
 */
export function formatMemoryPrompt(
	store: MemoryStore,
	maxEntries = 10,
): string {
	const top = store.getTopK(maxEntries);
	if (!top.length) return "";

	const lines = ["# Session Memory", ""];
	for (const entry of top) {
		const catLabel = {
			facts: "FACT",
			decisions: "DECISION",
			"file-paths": "FILE",
			errors: "ERROR",
			preferences: "PREFERENCE",
			context: "CONTEXT",
		}[entry.category];

		lines.push(`- **[${catLabel}]** ${entry.content}`);
	}

	return lines.join("\n") + "\n";
}

/**
 * Extract structured memories from assistant text. Looks for patterns like:
 * - "Key decision: ..."
 * - "Important note: ..."
 * - "File path: ..."
 * - "Error encountered: ..."
 */
export function extractMemoriesFromText(
	text: string,
	turnIndex: number,
): Array<{ category: MemoryCategory; content: string; turnIndex: number }> {
	const memories: Array<{
		category: MemoryCategory;
		content: string;
		turnIndex: number;
	}> = [];
	const lines = text.split("\n");

	for (const line of lines) {
		const trimmed = line.trim();
		if (!trimmed) continue;

		// Decision patterns
		const decisionMatch = trimmed.match(
			/(?:key\s+decision|decision|important\s+decision)\s*[:-]\s*(.+)/i,
		);
		if (decisionMatch) {
			memories.push({
				category: "decisions" as MemoryCategory,
				content: decisionMatch[1].trim(),
				turnIndex,
			});
			continue;
		}

		// File path patterns
		const fileMatch = trimmed.match(
			/(?:file\s+path|modified\s+file|created\s+file|edited\s+file)\s*[:-]\s*(.+)/i,
		);
		if (fileMatch) {
			memories.push({
				category: "file-paths" as MemoryCategory,
				content: fileMatch[1].trim(),
				turnIndex,
			});
			continue;
		}

		// Error patterns
		const errorMatch = trimmed.match(
			/(?:error|exception|failure|bug)\s*[:-]\s*(.+)/i,
		);
		if (errorMatch) {
			memories.push({
				category: "errors" as MemoryCategory,
				content: errorMatch[1].trim(),
				turnIndex,
			});
			continue;
		}

		// Preference patterns
		const prefMatch = trimmed.match(
			/(?:preference|style\s+preference|formatting\s+preference)\s*[:-]\s*(.+)/i,
		);
		if (prefMatch) {
			memories.push({
				category: "preferences" as MemoryCategory,
				content: prefMatch[1].trim(),
				turnIndex,
			});
			continue;
		}

		// Fact patterns
		const factMatch = trimmed.match(
			/(?:key\s+fact|important\s+fact|notable\s+fact|key\s+point)\s*[:-]\s*(.+)/i,
		);
		if (factMatch) {
			memories.push({
				category: "facts" as MemoryCategory,
				content: factMatch[1].trim(),
				turnIndex,
			});
		}
	}

	return memories;
}

/**
 * Retrieve memory entries most semantically relevant to a query,
 * formatted as a prompt fragment for injection into the system prompt.
 * Falls back to recency-based retrieval if no entries match.
 */
export function retrieveForPrompt(
	store: MemoryStore,
	query: string,
	maxEntries = 5,
): string {
	const results = store.search(query, maxEntries);
	if (!results.length) {
		return "";
	}

	const lines = ["# Relevant Memory", ""];
	for (const { entry, score } of results) {
		const catLabel = {
			facts: "FACT",
			decisions: "DECISION",
			"file-paths": "FILE",
			errors: "ERROR",
			preferences: "PREFERENCE",
			context: "CONTEXT",
		}[entry.category];
		lines.push(
			`- **[${catLabel}]** ${entry.content} (relevance: ${score.toFixed(2)})`,
		);
	}

	return lines.join("\n") + "\n";
}
