import type { Tool } from "@logician/agent-core";
import type { MemoryStore } from "@logician/memory";

export function createMemoryGetTool(getStore: () => MemoryStore | null): Tool {
  return {
    name: "memory_get",
    label: "Expand Memory",
    readOnly: true,
    executionMode: "parallel",
    description: "Expand compact memory or observation IDs from Agent Context into their complete stored details.",
    promptSnippet: "Expand compact memory IDs when their full rationale or evidence is needed",
    promptGuidelines: ["Use memory_get only for IDs shown in Agent Context; avoid expanding every result."],
    parameters: {
      type: "object",
      properties: {
        ids: {
          type: "array",
          items: { type: "string" },
          maxItems: 20,
          description: "Memory or observation IDs to expand",
        },
      },
      required: ["ids"],
    },
    execute: async (args) => {
      const store = getStore();
      if (!store) return "Memory is disabled.";
      const ids = Array.isArray(args.ids)
        ? args.ids.filter((id): id is string => typeof id === "string").slice(0, 20)
        : [];
      if (!ids.length) return "No valid memory IDs provided.";
      const entries = store.expandEntries(ids);
      if (!entries.length) return "No matching memory entries found in this workspace.";
      const found = new Set(entries.map((entry) => entry.id));
      const missing = ids.filter((id) => !found.has(id));
      return [
        ...entries.map((entry) => [
          `## ${entry.kind} ${entry.id}: ${entry.title}`,
          `Type: ${entry.type}`,
          entry.files.length ? `Files: ${entry.files.join(", ")}` : "",
          entry.content,
        ].filter(Boolean).join("\n\n")),
        missing.length ? `Missing or out of scope: ${missing.join(", ")}` : "",
      ].filter(Boolean).join("\n\n---\n\n");
    },
  };
}
