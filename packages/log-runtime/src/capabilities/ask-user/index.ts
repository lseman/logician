// ── ask_user tool — agent Q&A ─────────────────────────────────────────────
// The agent calls this tool when it needs to ask the user a question with
// multiple-choice options. Execution blocks until the user selects or dismisses.

import type { Tool, ToolContext } from "@logician/log-core";

export const ask_user: Tool = {
	readOnly: true,
	executionMode: "sequential",
	name: "ask_user",
	label: "Ask User",
	hookAliases: ["AskUser"],
	description:
		"Ask the user one or more tabbed questions with selectable options. Execution blocks " +
		"until the user responds. Use this when the agent needs user input to " +
		"proceed — e.g. clarifying requirements, choosing between alternatives, " +
		"or getting confirmation.",
	promptSnippet: "Ask the user structured questions with options",
	parameters: {
		type: "object",
		properties: {
			questions: {
				type: "array",
				description:
					"One or more questions shown as tabs. Use stable, unique ids and concise headers.",
				items: {
					type: "object",
					properties: {
						id: { type: "string", description: "Stable answer key." },
						header: { type: "string", description: "Short tab label." },
						question: { type: "string", description: "Question text." },
						choices: {
							type: "array",
							items: {
								type: "object",
								properties: {
									value: { type: "string" },
									label: { type: "string" },
									description: { type: "string" },
								},
								required: ["value", "label"],
							},
						},
					},
					required: ["id", "question", "choices"],
				},
			},
		},
		required: ["questions"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (typeof raw === "string") {
			try {
				return JSON.parse(raw);
			} catch (_e: unknown) {
				return {};
			}
		}
		if (!raw || typeof raw !== "object") return {};
		const args = raw as Record<string, unknown>;
		return { questions: args.questions };
	},
	execute: async (
		args: Record<string, unknown>,
		ctx?: ToolContext,
	): Promise<string> => {
		const normalizeChoices = (
			rawChoices: unknown,
		): Array<{ value: string; label: string; description?: string }> => {
			const choices: Array<{
				value: string;
				label: string;
				description?: string;
			}> = [];
			if (!Array.isArray(rawChoices)) return choices;
			for (const item of rawChoices) {
				if (!item || typeof item !== "object") continue;
				const obj = item as Record<string, unknown>;
				const value = String(obj.value || "").trim();
				const label = String(obj.label || "").trim();
				if (value && label) {
					const description = String(obj.description || "").trim();
					choices.push({
						value,
						label,
						...(description ? { description } : {}),
					});
				}
			}
			return choices;
		};

		const questions = Array.isArray(args.questions)
			? args.questions.flatMap(item => {
					if (!item || typeof item !== "object") return [];
					const obj = item as Record<string, unknown>;
					const question = String(obj.question || "").trim();
					const choices = normalizeChoices(obj.choices);
					if (!question || !choices.length) return [];
					return [
						{
							id: String(obj.id || "").trim(),
							header: String(obj.header || "").trim() || undefined,
							question,
							choices,
						},
					];
				})
			: [];

		if (
			questions.length === 0 ||
			questions.some(item => !item.id || !item.choices.length)
		) {
			return "Error: ask_user requires at least one choice with 'value' and 'label'.";
		}
		if (new Set(questions.map(item => item.id)).size !== questions.length) {
			return "Error: ask_user question ids must be unique.";
		}

		if (ctx?.onQuestionRequest) {
			const answer = await ctx.onQuestionRequest({ questions });
			return `User responded: ${answer}`;
		}

		return "Error: ask_user requires a question request handler (not available in non-interactive context).";
	},
};
