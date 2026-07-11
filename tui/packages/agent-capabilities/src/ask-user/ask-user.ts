// ── ask_user tool — agent Q&A ─────────────────────────────────────────────
// The agent calls this tool when it needs to ask the user a question with
// multiple-choice options. Execution blocks until the user selects or dismisses.

import type { Tool, ToolContext } from "@logician/agent-core/core/types.ts";

export const ask_user: Tool = {
	readOnly: true,
	name: "ask_user",
	label: "Ask User",
	hookAliases: ["AskUser"],
	description:
		"Ask the user a question with selectable options. Execution blocks " +
		"until the user responds. Use this when the agent needs user input to " +
		"proceed — e.g. clarifying requirements, choosing between alternatives, " +
		"or getting confirmation.",
	promptSnippet: "Ask the user structured questions with options",
	parameters: {
		type: "object",
		properties: {
			question: {
				type: "string",
				description: "The question to ask the user. Keep it concise and clear.",
			},
			choices: {
				type: "array",
				description:
					"List of selectable options. Each option has a 'value' (sent back to the agent) and a 'label' (shown to the user).",
				items: {
					type: "object",
					properties: {
						value: {
							type: "string",
							description: "The value returned when this option is selected.",
						},
						label: {
							type: "string",
							description: "Display text for this option.",
						},
					},
					required: ["value", "label"],
				},
			},
		},
		required: ["question", "choices"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (typeof raw === "string") {
			try {
				return JSON.parse(raw);
			} catch {
				return {};
			}
		}
		if (!raw || typeof raw !== "object") return {};
		const args = raw as Record<string, unknown>;
		return {
			question: String(args.question || ""),
			choices: args.choices ?? args.options ?? args.answers,
		};
	},
	execute: async (
		args: Record<string, unknown>,
		ctx?: ToolContext,
	): Promise<string> => {
		const question = String(args.question || "Please answer:");
		const rawChoices = args.choices ?? [];
		const choices: Array<{ value: string; label: string }> = [];

		if (Array.isArray(rawChoices)) {
			for (const item of rawChoices) {
				if (!item || typeof item !== "object") continue;
				const obj = item as Record<string, unknown>;
				const value = String(obj.value || String(obj.label || "")).trim();
				const label = String(obj.label || value).trim();
				if (value && label) {
					choices.push({ value, label });
				}
			}
		}

		if (choices.length === 0) {
			return "Error: ask_user requires at least one choice with 'value' and 'label'.";
		}

		if (ctx?.onQuestionRequest) {
			const answer = await ctx.onQuestionRequest({ question, choices });
			return `User responded: ${answer}`;
		}

		return "Error: ask_user requires a question request handler (not available in non-interactive context).";
	},
};
