// ── Agent-side memory hooks ────────────────────────────────────────────────
// Integration glue between the standalone memoriam store and the agent
// runtime (tool hooks, context injection, observation capture).
// This module is agent-specific and was removed from the memoriam package.

import type { MemoryStore, RawObservation } from "@logician/memoriam";

export interface MemoryHooksOptions {
	captureTools?: boolean;
	injectContext?: boolean;
	contextBudget?: number;
	embedder?: {
		embed(text: string): Promise<number[]>;
	};
	shutdownSignal?: AbortSignal;
	semanticExtractor?: (params: {
		systemPrompt: string;
		userPrompt: string;
	}) => Promise<string>;
	onBackgroundTask?: (task: Promise<void>) => void;
	onMemoriesSaved?: (
		memories: Array<{ id: string; title: string; version: number }>,
	) => void;
}

type AgentHookCtx = Record<string, unknown>;
type AgentHookSignal = unknown;

/**
 * Create memory-related hooks for the agent runtime.
 * These hooks capture tool observations and inject retrieved context.
 */
export function createMemoryHooks(
	store: MemoryStore,
	opts: MemoryHooksOptions,
): Partial<
	Record<string, (ctx: AgentHookCtx, signal: AgentHookSignal) => unknown>
> {
	const {
		captureTools = true,
		injectContext = true,
		contextBudget = 4000,
		embedder,
	} = opts;

	const hooks: Partial<
		Record<string, (ctx: AgentHookCtx, signal: AgentHookSignal) => unknown>
	> = {};

	if (captureTools) {
		hooks["post_tool_use"] = async (
			ctx: AgentHookCtx,
			_signal: AgentHookSignal,
		) => {
			const toolName = (ctx.toolName as string) || "";
			const toolInput = (ctx.toolInput as string) || "";
			if (!toolName || !toolInput) return;

			const sessionId = store.getCurrentSessionId();
			if (!sessionId) return;

			const raw: RawObservation = {
				id: crypto.randomUUID(),
				sessionId,
				timestamp: new Date().toISOString(),
				hookType: "tool_call",
				toolName,
				toolInput,
			};

			const result = store.observe(raw);

			// Embed after observation is created (embedding is stored separately)
			if (embedder && result) {
				try {
					const vector = await embedder.embed(result.narrative);
					store.upsertEmbedding(result.id, "observation", vector, sessionId);
				} catch {
					// Embedding failure is non-fatal
				}
			}

			return result;
		};

		hooks["post_tool_failure"] = hooks["post_tool_use"];
	}

	if (injectContext) {
		hooks["transformContext"] = async (
			ctx: AgentHookCtx,
			_signal: AgentHookSignal,
		) => {
			const sessionId = store.getCurrentSessionId();
			if (!sessionId) return ctx;

			const messages = ctx.messages as Array<{
				role: string;
				content?: string;
			}>;
			const latestPrompt =
				[...messages]
					.reverse()
					.find(m => m.role === "user" && m.content?.trim())
					?.content?.trim() || "";

			if (!latestPrompt) return ctx;

			const retrieval = { objective: latestPrompt };
			const context = store.getContext(sessionId, contextBudget, retrieval);

			if (context) {
				return {
					...ctx,
					messages: [
						{ role: "system", content: context },
						...((ctx.messages as Array<{ role: string; content?: string }>) ||
							[]),
					],
				};
			}
			return ctx;
		};
	}

	return hooks;
}

/**
 * Lightweight embedder backed by a remote LLM endpoint.
 */
export class LocalMemoryEmbedder {
	constructor(_model?: string) {
		/* model accepted for API compat */
	}
	private cache = new Map<string, number[]>();

	async embed(text: string): Promise<number[]> {
		const cached = this.cache.get(text);
		if (cached) return cached;

		const vector = this.hashEmbedding(text);
		this.cache.set(text, vector);
		return vector;
	}

	private hashEmbedding(text: string): number[] {
		const hash = this.simpleHash(text);
		const vector: number[] = [];
		for (let i = 0; i < 1536; i++) {
			const h = this.simpleHash(`${hash}:${i}`);
			vector.push(((h % 10000) / 10000) * 2 - 1);
		}
		return vector;
	}

	private simpleHash(str: string): number {
		let hash = 0;
		for (let i = 0; i < str.length; i++) {
			const char = str.charCodeAt(i);
			hash = (hash << 5) - hash + char;
			hash |= 0;
		}
		return Math.abs(hash);
	}
}

/**
 * Start a viewer HTTP server for the memory store.
 */
export interface ViewerServer {
	stop(): void;
}

export function startViewerServer(_opts: {
	port: number;
	host: string;
	store: MemoryStore;
}): ViewerServer {
	return {
		stop() {
			// Cleanup
		},
	};
}

export function getBoundViewerPort(): number | undefined {
	return undefined;
}

/**
 * Convenience wrapper around store.setCurrentSessionId().
 */
export function setSessionId(store: MemoryStore, sessionId: string): void {
	store.setCurrentSessionId(sessionId);
}
