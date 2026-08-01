// ── Hook Adapter: Wired into Logician's HookBus ──────────────────────────────
import type { MemoryStore } from "./types.js";

/**
 * Register memory hooks on a HookBus.
 * 
 * - beforeCompact → recall relevant memories and append to compacted context
 * - beforeAgentStart → warm up recent memories from store
 */
export function registerMemoryHooks(
  bus: any, // HookBus instance from agent-core
  store: MemoryStore,
  options?: { maxRecentMemories?: number },
): () => void {
  const maxRecent = options?.maxRecentMemories ?? 20;

  // beforeAgentStart: warm up recent memories as initial context
  bus.on("agentStart", (ctx: any, signal?: AbortSignal) => {
    const recent = store.list({ limit: maxRecent });
    if (!recent.length) return ctx;

    const summary = `# Previous Session Memories\n\n${recent
      .map((m: any) => `- [${m.importance}/10] ${m.content.slice(0, 200)}`)
      .join("\n")}\n`;

    if (ctx.initialPrompt && typeof ctx.initialPrompt === "string") {
      return {
        ...ctx,
        initialPrompt: summary + "\n\n---\n\n" + ctx.initialPrompt,
      };
    }
    return ctx;
  });

  // beforeCompact: recall relevant memories to inject into compacted context
  bus.on("beforeCompact", (ctx: any, signal?: AbortSignal) => {
    const relevant = store.list({ limit: 10 });
    if (!relevant.length) return ctx;

    const memorySection = `# Agent Memories\n\n${relevant
      .map((m: any) => `## ${m.source || "memory"} [${m.importance}/10]\n\n${m.content}`)
      .join("\n\n")}\n`;

    if (ctx.summary && typeof ctx.summary === "string") {
      return {
        ...ctx,
        summary: ctx.summary + "\n\n---\n\n" + memorySection,
      };
    }
    return ctx;
  });

  // Return cleanup function
  return () => {
    store.close();
  };
}

/**
 * Manual remember tool: save a memory entry.
 * Returns the created memory ID for reference.
 */
export function remember(
  store: MemoryStore,
  content: string,
  source: string = "manual",
  sessionId: string = "",
  importance?: number,
  tags?: string[],
): string {
  const entry = store.create(content, {
    source,
    sessionId,
    importance,
    tags,
    autoTags: true,
  });
  return entry.id;
}

/**
 * Manual recall tool: search and format memories.
 */
export function recall(
  store: MemoryStore,
  query: string,
  limit: number = 10,
  format?: "text" | "markdown",
): string {
  const result = store.recall(
    { search: query, limit },
    { format: format || "text" },
  );
  return result || `No memories found matching "${query}"`;
}

/**
 * List tool: enumerate memories with optional filters.
 */
export function listMemories(
  store: MemoryStore,
  query?: { tags?: string[]; source?: string; minImportance?: number },
  limit: number = 20,
): string {
  const memories = store.list({ ...query, limit });

  if (!memories.length) return "No memories match the query.";

  return memories
    .map(
      (m: any) =>
        `[${m.importance}/10] ${m.source} | ${m.createdAt.slice(0, 10)}\n${m.content.slice(0, 300)}`,
    )
    .join("\n\n---\n\n");
}

/**
 * Forget tool: delete a memory by ID.
 */
export function forget(
  store: MemoryStore,
  id: string,
): string {
  const deleted = store.delete(id);
  return deleted ? `Memory ${id} deleted.` : `Memory ${id} not found.`;
}
