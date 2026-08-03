// ── @logician/memory — Hook Factory ──────────────────────────────────────────
// Creates AgentHooks for the Logician agent — observation capture + context injection.
// Returns undefined if memory is disabled.

import type { AgentHooks } from "@logician/agent-core";
import type { CompressedObservation, Memory, MemoryStore, RawObservation } from "./types.js";
import type { ExplicitTaskState } from "@logician/agent-core";

export interface MemoryHooksConfig {
  /** Whether to capture tool observations. Default: true */
  captureTools?: boolean;
  /** Whether to capture user prompts. Default: true */
  capturePrompts?: boolean;
  /** Whether to inject context into agent messages. Default: true */
  injectContext?: boolean;
  /** Token budget for context injection. Default: 4000 */
  contextBudget?: number;
  /** Called synchronously after an observation has been persisted. */
  onObservationSaved?: (observation: CompressedObservation) => void;
  /** Deduplicate equivalent observations for five minutes. Default: true. */
  deduplicate?: boolean;
  /** Consolidate high-signal observations at turn/compaction boundaries. */
  autoConsolidate?: boolean;
  /** Called after automatic consolidation creates or evolves memories. */
  onMemoriesSaved?: (memories: Memory[]) => void;
}

function saveObservation(
  store: MemoryStore,
  raw: Omit<RawObservation, "sessionId" | "timestamp" | "workspace"> &
    Partial<Pick<RawObservation, "timestamp" | "workspace">>,
  onSaved?: (observation: CompressedObservation) => void,
  deduplicate: boolean = true,
): CompressedObservation | null {
  const sessionId = store.getCurrentSessionId();
  if (!sessionId) return null;

  const dedupName = raw.toolName || raw.hookType;
  const dedupInput = raw.toolInput ?? raw.userPrompt ?? raw.raw;
  if (deduplicate && store.dedupCheck(sessionId, dedupName, dedupInput)) {
    return null;
  }

  const observation = store.observe({
    ...raw,
    sessionId,
    timestamp: raw.timestamp || new Date().toISOString(),
    workspace: raw.workspace ?? store.getCurrentWorkspace(),
  });
  if (observation && deduplicate) {
    store.dedupRecord(sessionId, dedupName, dedupInput);
  }
  if (observation && onSaved) {
    try { onSaved(observation); } catch {}
  }
  return observation;
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
  const capturePrompts = config.capturePrompts ?? true;
  const injectContext = config.injectContext ?? true;
  const contextBudget = config.contextBudget ?? 4000;
  const onObservationSaved = config.onObservationSaved;
  const deduplicate = config.deduplicate ?? true;
  const autoConsolidate = config.autoConsolidate ?? true;
  let latestPrompt = "";

  const consolidate = () => {
    const sessionId = store.getCurrentSessionId();
    if (!sessionId) return;
    const memories = store.consolidate(sessionId);
    if (memories.length && config.onMemoriesSaved) {
      try { config.onMemoriesSaved(memories); } catch {}
    }
  };

  // Capture the user's request at the start of every turn. AgentMemory treats
  // prompts as first-class observations because they preserve intent even when
  // a turn never reaches a tool call.
  const beforeAgentStart = capturePrompts
    ? (ctx: { prompt: string }) => {
        const prompt = ctx.prompt?.trim();
        if (!prompt) return undefined;
        latestPrompt = prompt;
        saveObservation(store, {
          id: crypto.randomUUID(),
          hookType: "prompt_submit",
          userPrompt: prompt,
          raw: { prompt },
        }, onObservationSaved, deduplicate);
        return undefined;
      }
    : undefined;

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

        // AgentMemory intentionally ignores interrupted failures: they are
        // user/runtime control flow, not evidence about the project.
        if (ctx.isError && /^(?:cancelled|canceled|aborted|interrupted)\b/i.test(ctx.result.trim())) {
          return undefined;
        }

        saveObservation(store, {
          id: `${ctx.toolCall.id || crypto.randomUUID()}:post`,
          hookType: ctx.isError ? "post_tool_failure" : "post_tool_use",
          toolName,
          toolInput: ctx.args,
          toolOutput: ctx.result,
          raw: {
            tool_name: toolName,
            tool_input: ctx.args,
            tool_output: ctx.result,
            ...(ctx.isError ? { error: ctx.result } : {}),
          },
        }, onObservationSaved, deduplicate);

        return undefined;
      }
    : undefined;

  // ── transformContext: inject session context into messages ──────────

  const transformContext = injectContext
    ? (ctx: { messages: any[]; taskState?: ExplicitTaskState }) => {
        const sessionId = store.getCurrentSessionId();
        if (!sessionId) return undefined;
        const sessionContext = store.getContext(
          sessionId,
          contextBudget,
          ctx.taskState
            ? {
                objective: ctx.taskState.objective || latestPrompt,
                phase: ctx.taskState.phase,
                changedFiles: ctx.taskState.changedFiles,
                recentEvidence: ctx.taskState.evidence.slice(-6).map((item) => item.summary),
                toolFailures: ctx.taskState.toolFailures,
              }
            : latestPrompt,
        );
        if (!sessionContext) return undefined;

        const messages = [
          ...ctx.messages.filter(
            (message) =>
              !(message?.role === "system" &&
                typeof message.content === "string" &&
                message.content.startsWith("# Agent Context\n")),
          ),
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

  if (beforeAgentStart) hooks.beforeAgentStart = beforeAgentStart;
  if (afterToolCall) hooks.afterToolCall = afterToolCall;
  if (transformContext) hooks.transformContext = transformContext;
  if (autoConsolidate) {
    hooks.shouldStopAfterTurn = () => {
      consolidate();
      return undefined;
    };
    hooks.beforeCompact = () => {
      consolidate();
      return undefined;
    };
  }

  return hooks;
}
