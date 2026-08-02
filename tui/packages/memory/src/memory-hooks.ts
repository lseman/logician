// ── @logician/memory — Hook Factory ──────────────────────────────────────────
// Creates AgentHooks for the Logician agent — observation capture + context injection.
// Returns undefined if memory is disabled.

import type { AgentHooks } from "@logician/agent-core";
import type { MemoryStore } from "./types.js";

export interface MemoryHooksConfig {
  /** Whether to capture tool observations. Default: true */
  captureTools?: boolean;
  /** Whether to inject context into agent messages. Default: true */
  injectContext?: boolean;
  /** Token budget for context injection. Default: 4000 */
  contextBudget?: number;
}

/**
 * Create memory hooks from a store. Returns an AgentHooks object (or undefined
 * if memory is disabled). The hooks are safe to call repeatedly — the store is
 * read-only during hook execution and never blocks the turn.
 */
export function createMemoryHooks(
  store: MemoryStore,
  config: MemoryHooksConfig = {},
): AgentHooks {
  const captureTools = config.captureTools ?? true;
  const injectContext = config.injectContext ?? true;
  const contextBudget = config.contextBudget ?? 4000;

  // ── afterToolCall: capture observations ─────────────────────────────

  const afterToolCall = captureTools
    ? (ctx: {
        toolCall: { name: string; id?: string; arguments?: string };
        args: Record<string, unknown>;
        result: string;
        isError: boolean;
      }) => {
        const toolName =
          ctx.toolCall.name ||
          (ctx.args.tool_name as string) ||
          (ctx.args.name as string) ||
          "unknown";

        // Derive workspace from store's current workspace
        const workspace = store.getCurrentWorkspace();

        // Build a synthetic observation for capture
        store.observe({
          id: ctx.toolCall.id || crypto.randomUUID(),
          sessionId: "memory", // session ID is managed externally
          timestamp: new Date().toISOString(),
          hookType: ctx.isError ? "post_tool_failure" : "post_tool_use",
          toolName,
          toolInput: ctx.args,
          toolOutput: ctx.result,
          workspace,
          raw: {
            tool_name: toolName,
            tool_input: ctx.args,
            tool_output: ctx.result,
            ...(ctx.isError ? { error: ctx.result } : {}),
          },
        });

        return undefined;
      }
    : undefined;

  // ── transformContext: inject session context into messages ──────────

  const transformContext = injectContext
    ? (ctx: { messages: any[] }) => {
        const sessionContext = store.getContext("memory", contextBudget);
        if (!sessionContext) return undefined;

        const messages = [
          ...ctx.messages,
          {
            role: "system" as const,
            content: sessionContext,
          },
        ];

        return { messages };
      }
    : undefined;

  // ── Hook composition ───────────────────────────────────────────────

  const hooks: AgentHooks = {};

  if (afterToolCall) hooks.afterToolCall = afterToolCall;
  if (transformContext) hooks.transformContext = transformContext;

  return hooks;
}
