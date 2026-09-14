// ── retain tool — write durable facts into long-term memory ────────────────
// Wires the model to MemoriamGateway.createMemory. Only registered when
// Memoriam is enabled (config.memoriam.mode === "sdk") — see default-tools.ts.

import type { Tool } from "@logician/log-core";
import type { MemoriamGateway } from "./memoriam-gateway.ts";

export interface RetainToolOptions {
	gateway: MemoriamGateway;
	sessionId: string;
}

export function createRetainTool(opts: RetainToolOptions): Tool {
	const { gateway, sessionId } = opts;
	return {
		readOnly: true,
		executionMode: "sequential",
		name: "retain",
		label: "Retain",
		hookAliases: ["Retain"],
		description:
			"Store one or more facts in long-term memory, for future sessions to draw on. Use " +
			"for durable, reusable knowledge — user preferences, project decisions, " +
			"architectural choices; anything that would improve a future response. Not for " +
			"ephemeral task state. Each item should be specific and self-contained (who, what, " +
			"when, why). Batch related facts into one call; memories are deduplicated and " +
			"consolidated over time.",
		promptSnippet: "Store durable facts in long-term memory",
		parameters: {
			type: "object",
			properties: {
				items: {
					type: "array",
					description: "Memories to retain.",
					items: {
						type: "object",
						properties: {
							content: {
								type: "string",
								description: "Information to remember.",
							},
							context: {
								type: "string",
								description: "Source context (optional).",
							},
						},
						required: ["content"],
					},
				},
			},
			required: ["items"],
		},
		execute: async (args: Record<string, unknown>): Promise<string> => {
			const rawItems = Array.isArray(args.items) ? args.items : [];
			const items = rawItems.flatMap(item => {
				if (!item || typeof item !== "object") return [];
				const obj = item as Record<string, unknown>;
				const content = String(obj.content || "").trim();
				if (!content) return [];
				const context = String(obj.context || "").trim();
				return [{ content, context: context || undefined }];
			});
			if (!items.length) {
				return "Error: retain requires at least one item with non-empty 'content'.";
			}
			for (const item of items) {
				await gateway.createMemory(
					item.context ? `${item.content}\n\nContext: ${item.context}` : item.content,
					{ type: "fact", sessionIds: [sessionId] },
				);
			}
			const noun = items.length === 1 ? "memory" : "memories";
			return `${items.length} ${noun} stored.`;
		},
	};
}
