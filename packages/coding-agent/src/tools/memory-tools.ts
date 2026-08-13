import type { Tool } from "@logician/agent-core";
import type { MemoryStore } from "@logician/memory";

function brief(value: string, maxLength: number = 220): string {
	const normalized = value.replace(/\s+/g, " ").trim();
	if (normalized.length <= maxLength) return normalized;
	const slice = normalized.slice(0, maxLength - 1);
	const boundary = slice.lastIndexOf(" ");
	return `${slice.slice(0, boundary > maxLength * 0.65 ? boundary : undefined).trimEnd()}…`;
}

export function createMemorySearchTool(
	getStore: () => MemoryStore | null,
): Tool {
	return {
		name: "memory_search",
		label: "Search Memory",
		readOnly: true,
		executionMode: "parallel",
		description:
			"Search this workspace's durable memories and observations, returning compact summaries with stable IDs.",
		promptSnippet:
			"Search memory indexes for relevant IDs before expanding full records",
		promptGuidelines: [
			"Use memory_search when Agent Context does not contain enough relevant history.",
			"Filter the compact results, then batch only the relevant IDs into one memory_get call.",
		],
		parameters: {
			type: "object",
			properties: {
				query: {
					type: "string",
					description: "Words or concepts to find in workspace memory",
				},
				limit: {
					type: "number",
					minimum: 1,
					maximum: 20,
					description: "Maximum results of each kind (default 8)",
				},
			},
			required: ["query"],
		},
		execute: async args => {
			const store = getStore();
			if (!store) return "Memory is disabled.";
			const query = typeof args.query === "string" ? args.query.trim() : "";
			if (!query) return "No valid memory search query provided.";
			const limit =
				typeof args.limit === "number" && Number.isFinite(args.limit)
					? Math.max(1, Math.min(20, Math.floor(args.limit)))
					: 8;
			const memories = store.list({ search: query, limit });
			const observations = store.searchObservations(query, limit);
			const results = [
				...memories.map(
					memory =>
						`- [${memory.id}] Memory · ${memory.type} · ${memory.title} — ${brief(memory.content)}`,
				),
				...observations.map(({ observation }) => {
					const description = brief(
						[observation.narrative, ...observation.facts]
							.filter(Boolean)
							.join(" "),
					);
					return `- [${observation.id}] Observation · ${observation.type} · ${observation.title}${description ? ` — ${description}` : ""}`;
				}),
			];
			if (!results.length)
				return `No memory entries matched "${query}" in this workspace.`;
			return [
				`# Memory search: ${query}`,
				...results,
				"Use one `memory_get` call with only the relevant bracketed IDs to retrieve complete records.",
			].join("\n\n");
		},
	};
}

export function createMemoryGetTool(getStore: () => MemoryStore | null): Tool {
	return {
		name: "memory_get",
		label: "Expand Memory",
		readOnly: true,
		executionMode: "parallel",
		description:
			"Expand compact memory or observation IDs from Agent Context into their complete stored details.",
		promptSnippet:
			"Expand compact memory IDs when their full rationale or evidence is needed",
		promptGuidelines: [
			"Use memory_get only for relevant IDs shown by Agent Context or memory_search.",
			"Batch related IDs into one call; avoid expanding every result.",
		],
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
		execute: async args => {
			const store = getStore();
			if (!store) return "Memory is disabled.";
			const ids = Array.isArray(args.ids)
				? args.ids
						.filter((id): id is string => typeof id === "string")
						.slice(0, 20)
				: [];
			if (!ids.length) return "No valid memory IDs provided.";
			const entries = store.expandEntries(ids);
			if (!entries.length)
				return "No matching memory entries found in this workspace.";
			const found = new Set(entries.map(entry => entry.id));
			const missing = ids.filter(id => !found.has(id));
			return [
				...entries.map(entry =>
					[
						`## ${entry.kind} ${entry.id}: ${entry.title}`,
						`Type: ${entry.type}`,
						entry.files.length ? `Files: ${entry.files.join(", ")}` : "",
						entry.content,
					]
						.filter(Boolean)
						.join("\n\n"),
				),
				missing.length ? `Missing or out of scope: ${missing.join(", ")}` : "",
			]
				.filter(Boolean)
				.join("\n\n---\n\n");
		},
	};
}
