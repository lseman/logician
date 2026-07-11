// ── Agent-facing recall tool ─────────────────────────────────────────────
// Provides the `recall` tool that agents use to recover exact evidence
// for a memory ID.

import type { MemoryStore } from "./store.ts";
import { recallMemory, formatRecallResult, isValidMemoryId } from "./recall.ts";

export const RECALL_TOOL_NAME = "recall";

export interface RecallToolOptions {
	memoryStore: MemoryStore;
	/** Available source entries from session (for source resolution) */
	sourceEntries: Array<{
		id: string;
		type: string;
		origin: string;
		timestamp: string;
		content?: string;
	}>;
}

export interface RecallToolResult {
	status:
		| "ok"
		| "partial"
		| "invalid_id"
		| "not_found"
		| "no_source"
		| "source_unavailable";
	memoryId: string;
	content?: string;
}

/**
 * Create a recall tool handler.
 * Returns a function that takes a memory ID and returns a RecallToolResult.
 */
export function createRecallTool(ctx: RecallToolOptions) {
	const { memoryStore, sourceEntries } = ctx;

	return function recall(memoryId: string): RecallToolResult {
		// Validate ID format
		if (!isValidMemoryId(memoryId)) {
			return {
				status: "invalid_id",
				memoryId,
				content: `Memory ID must be 12 lowercase hex characters. Received: ${memoryId}`,
			};
		}

		// Look up in memory store
		const dropped = memoryStore.getAllDroppedIds();

		const result = recallMemory(
			memoryId,
			memoryStore.getAllObservations(),
			memoryStore.getReflections(),
			dropped,
			sourceEntries,
		);

		if (result.status === "not_found") {
			return {
				status: "not_found",
				memoryId,
				content:
					result.status === "not_found"
						? `No observation or reflection with id ${memoryId} was found.`
						: undefined,
			};
		}

		return {
			status: result.partial ? "partial" : "ok",
			memoryId,
			content: formatRecallResult(result),
		};
	};
}
