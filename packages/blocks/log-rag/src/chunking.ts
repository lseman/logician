// ── Smart Chunking ───────────────────────────────────────────────────────────
// Three strategies: recursive (hierarchical splitting), semantic (embedding-based
// boundary detection), and parent-child (small chunks with large context parents).
// All produce chunks with overlap and respect structural boundaries.

import type { ChunkingConfig, ParentContext, RAGChunk } from "./types.ts";

// ── Recursive Chunking ───────────────────────────────────────────────────────

/**
 * Recursively split text using a prioritized list of separators.
 * Respects chunk size limits, overlap, and minimum size.
 */
export function recursiveChunk(
	text: string,
	config?: Partial<ChunkingConfig>,
): RAGChunk[] {
	const {
		chunkSize = 512,
		overlap = 128,
		minChunkSize = 64,
		maxDepth = 3,
		separators = ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
	} = { ...config };

	let chunkIndex = 0;

	function split(content: string, depth: number, baseId: string): RAGChunk[] {
		if (depth >= maxDepth || content.length <= chunkSize) {
			// Base case: emit the content as one or more chunks with overlap
			return emitChunks(content, baseId);
		}

		for (const sep of separators) {
			if (sep === "") {
				// Last resort: force-split at chunkSize
				return emitChunks(content, baseId);
			}

			const parts = content.split(sep);
			if (parts.length <= 1) continue;

			let accumulated: string[] = [];
			let accumulatedLen = 0;
			const partChunks: RAGChunk[] = [];

			for (let i = 0; i < parts.length; i++) {
				const part = parts[i];
				const partLen = part.length;

				if (accumulatedLen + partLen <= chunkSize) {
					accumulated.push(part);
					accumulatedLen += partLen;
				} else {
					// Emit accumulated content as chunks
					const joined = accumulated.join(sep);
					// Emit this accumulated block
					const emitted = emitChunks(joined, `${baseId}_${depth}`);
					partChunks.push(...emitted);
					accumulated = [part];
					accumulatedLen = partLen;
				}
			}

			// Don't forget the last accumulated block
			if (accumulated.length > 0) {
				const joined = accumulated.join(sep);
				const emitted = emitChunks(joined, `${baseId}_${depth}`);
				partChunks.push(...emitted);
			}

			if (partChunks.length > 0) {
				return partChunks;
			}
		}

		// Fallback: force split
		return emitChunks(content, baseId);
	}

	function emitChunks(content: string, prefix: string): RAGChunk[] {
		if (content.length <= chunkSize) {
			const trimmed = content.trim();
			if (trimmed.length < minChunkSize) return [];
			return [
				{
					id: `${prefix}_${chunkIndex++}`,
					text: trimmed,
					metadata: {},
					approxTokens: Math.ceil(trimmed.length / 4),
				},
			];
		}

		const chunks: RAGChunk[] = [];
		let start = 0;

		while (start < content.length) {
			const end = Math.min(start + chunkSize, content.length);
			// Try to break at whitespace near the boundary
			let breakPoint = end;
			if (end < content.length) {
				const lookBack = Math.min(overlap + 50, end - start);
				const segment = content.slice(end - lookBack, end);
				const spaceIdx = segment.lastIndexOf(" ");
				if (spaceIdx > 0) {
					breakPoint = end - lookBack + spaceIdx;
				}
			}

			const chunkText = content.slice(start, breakPoint).trim();
			if (chunkText.length >= minChunkSize) {
				chunks.push({
					id: `${prefix}_${chunkIndex++}`,
					text: chunkText,
					metadata: {},
					approxTokens: Math.ceil(chunkText.length / 4),
				});
			}

			// Move forward with overlap
			start = breakPoint - overlap;
			if (start <= 0) start = end;
		}

		return chunks;
	}

	return split(text, 0, "chunk");
}

// ── Semantic Chunking ────────────────────────────────────────────────────────

/**
 * Split text at semantic boundaries detected by embedding similarity drops.
 * Uses a sliding window: compares adjacent windows via cosine similarity.
 * Low similarity → new chunk boundary.
 *
 * Requires an `embed` function. Falls back to recursive chunking if not provided.
 */
export async function semanticChunk(
	text: string,
	options?: {
		embed?: (text: string) => Promise<number[]>;
		threshold?: number; // min similarity to stay in same chunk (0–1)
		windowSize?: number; // window size in chars for comparison
		chunkSize?: number; // target chunk size for post-processing
	},
): Promise<RAGChunk[]> {
	const {
		embed,
		threshold = 0.85,
		windowSize = 256,
		chunkSize = 512,
	} = { ...options };

	if (!embed) {
		// No embedder provided — fall back to recursive chunking
		return recursiveChunk(text, { chunkSize });
	}

	const words = text.split(/\s+/);
	if (words.length === 0) return [];

	// Build windows of words
	const windowWordCount = Math.max(8, Math.floor(windowSize / 4)); // ~4 chars per token
	const windows: Array<{ startWord: number; endWord: number; text: string }> =
		[];

	for (
		let i = 0;
		i <= words.length - windowWordCount;
		i += Math.floor(windowWordCount / 2)
	) {
		const end = Math.min(i + windowWordCount, words.length);
		const windowText = words.slice(i, end).join(" ");
		windows.push({ startWord: i, endWord: end, text: windowText });
	}

	if (windows.length < 2) {
		return recursiveChunk(text, { chunkSize });
	}

	// Embed all windows
	const embeddings: number[][] = [];
	for (const w of windows) {
		const vec = await embed(w.text);
		embeddings.push(vec);
	}

	// Find boundary indices where similarity drops below threshold
	const boundaryWordIndices: number[] = [];
	for (let i = 1; i < embeddings.length; i++) {
		const sim = cosineSimilarity(embeddings[i - 1], embeddings[i]);
		if (sim < threshold) {
			// Boundary between window i-1 and window i
			boundaryWordIndices.push(windows[i].startWord);
		}
	}

	// Split text at boundary indices
	const chunks: RAGChunk[] = [];
	let start = 0;
	for (const boundary of boundaryWordIndices) {
		const chunkText = text.slice(start, boundary).trim();
		if (chunkText.length >= 32) {
			chunks.push({
				id: `semantic_${chunks.length}`,
				text: chunkText,
				metadata: {},
				approxTokens: Math.ceil(chunkText.length / 4),
			});
		}
		start = boundary;
	}
	// Final chunk
	if (start < text.length) {
		const chunkText = text.slice(start).trim();
		if (chunkText.length >= 32) {
			chunks.push({
				id: `semantic_${chunks.length}`,
				text: chunkText,
				metadata: {},
				approxTokens: Math.ceil(chunkText.length / 4),
			});
		}
	}

	return chunks.length > 0 ? chunks : recursiveChunk(text, { chunkSize });
}

// ── Parent-Child Chunking ────────────────────────────────────────────────────

/**
 * Split text into small chunks for precise retrieval, plus larger parent
 * contexts that wrap each child for richer context at query time.
 *
 * Usage: Retrieve small chunks, then return their parent context blocks.
 */
export function parentChildChunk(
	text: string,
	options?: {
		childSize?: number; // small chunk size for embedding
		childOverlap?: number;
		parentSize?: number; // larger parent context size
		parentOverlap?: number;
	},
): { children: RAGChunk[]; parents: ParentContext[] } {
	const {
		childSize = 256,
		childOverlap = 64,
		parentSize = 1024,
		parentOverlap = 128,
	} = { ...options };

	// Step 1: Create small child chunks
	const children: RAGChunk[] = [];
	let start = 0;
	let childIdx = 0;

	while (start < text.length) {
		const end = Math.min(start + childSize, text.length);
		// Try to break at sentence or word boundary
		let breakPoint = end;
		if (end < text.length) {
			const lookBack = Math.min(childOverlap + 30, end - start);
			const segment = text.slice(end - lookBack, end);
			const dotIdx = segment.lastIndexOf(". ");
			const newlineIdx = segment.lastIndexOf("\n");
			const spaceIdx = segment.lastIndexOf(" ");
			const bestIdx = Math.max(dotIdx, newlineIdx, spaceIdx);
			if (bestIdx > 0) {
				breakPoint = end - lookBack + bestIdx + (dotIdx >= 0 ? 1 : 0);
			}
		}

		const childText = text.slice(start, breakPoint).trim();
		if (childText.length >= 16) {
			const childId = `child_${childIdx++}`;
			children.push({
				id: childId,
				text: childText,
				metadata: {},
				approxTokens: Math.ceil(childText.length / 4),
			});
		}

		start = breakPoint - childOverlap;
		if (start <= 0) start = end;
	}

	// Step 2: Create parent contexts by grouping children
	const parents: ParentContext[] = [];
	const childrenPerParent = Math.max(1, Math.floor(parentSize / childSize));
	const overlappingChildren = Math.min(
		childrenPerParent - 1,
		Math.max(0, Math.floor(parentOverlap / childSize)),
	);
	const parentStep = Math.max(1, childrenPerParent - overlappingChildren);
	for (let i = 0; i < children.length; i += parentStep) {
		const groupEnd = Math.min(i + childrenPerParent, children.length);
		const groupChildren = children.slice(i, groupEnd);
		const parentText = groupChildren.map(c => c.text).join("\n\n");

		if (parentText.length > 0) {
			const parentId = `parent_${i}`;
			parents.push({
				id: parentId,
				text: parentText,
				childIds: groupChildren.map(c => c.id),
			});
		}
	}

	return { children, parents };
}

// ── Utility ──────────────────────────────────────────────────────────────────

/**
 * Smart chunking dispatcher — chooses the best strategy based on config.
 */
export async function smartChunk(
	text: string,
	config?: Partial<ChunkingConfig>,
	embedder?: (text: string) => Promise<number[]>,
): Promise<RAGChunk[]> {
	const strategy = config?.strategy ?? "recursive";

	switch (strategy) {
		case "semantic":
			return semanticChunk(text, {
				...config,
				embed: embedder,
			});
		default:
			return recursiveChunk(text, config);
	}
}

// ── Inline helpers ───────────────────────────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
	let dot = 0,
		na = 0,
		nb = 0;
	const len = Math.min(a.length, b.length);
	for (let i = 0; i < len; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	if (na === 0 || nb === 0) return 0;
	return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
