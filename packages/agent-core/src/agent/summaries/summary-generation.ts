// ── LLM-backed summary generation ─────────────────────────────────────────
// Shared by compaction (condense old messages) and branch summarization
// (structured goal/progress/decisions summary for an abandoned branch).

import type { LLMBackend } from "../backend.ts";
import { convertToChatFormat } from "../messages.ts";
import type { Message, ThinkingLevel } from "../types.ts";
import {
	computeFileLists,
	extractFileOpsFromMessages,
	type FileOperations,
	formatFileOperations,
	parseBranchSummary,
	serializeMessages,
} from "./branch-summarization.ts";
import type { BranchProgress, BranchSummaryData } from "./types.ts";

/** Condense older messages into a short plain-text summary for compaction. */
export async function generateCompactionSummary(
	backend: LLMBackend,
	messages: Message[],
	systemMessages: Message[],
	options: {
		temperature?: number;
		maxTokens?: number;
		thinkingLevel?: ThinkingLevel;
	},
): Promise<string | null> {
	try {
		const chatMessages = [
			...systemMessages,
			{
				role: "user" as const,
				content:
					"Summarize the following conversation history concisely. " +
					"Focus on key decisions, actions taken, files modified, " +
					"and any important context that should be retained. " +
					"Be brief but preserve all actionable information.",
			},
			...messages,
		];

		const response = await backend.generate(convertToChatFormat(chatMessages), {
			temperature: options.temperature ?? 0.3,
			maxTokens: Math.min(2048, (options.maxTokens ?? 4096) / 2),
			thinkingLevel: options.thinkingLevel,
		});

		return response.content?.trim() || null;
	} catch (_e: unknown) {
		return null;
	}
}

/**
 * Generate a structured branch summary via LLM: goal, constraints, progress,
 * key decisions, next steps. Falls back to a basic summary on any failure.
 */
export async function generateBranchSummaryText(
	backend: LLMBackend,
	messages: Message[],
	options: {
		customInstructions?: string;
		fileOps?: FileOperations;
		maxTokens?: number;
		thinkingLevel?: ThinkingLevel;
	} = {},
): Promise<BranchSummaryData | null> {
	if (messages.length === 0) return null;

	const extractedOps = options.fileOps ?? extractFileOpsFromMessages(messages);
	const { readFiles, modifiedFiles } = computeFileLists(extractedOps);

	try {
		const conversationText = serializeMessages(messages);

		let instructions = `Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[One sentence: what was the user trying to accomplish]

## Constraints & Preferences
- [constraint 1]
- [constraint 2]
- (none) if none were mentioned

## Progress
### Done
- [x] [completed task/change]
### In Progress
- [ ] [started but unfinished]
### Blocked
- [issue preventing progress, if any]

## Key Decisions
- **[decision]**: [brief rationale]

## Next Steps
1. [first step to continue]
2. [second step, if any]

Preserve exact file paths, function names, and error messages. Be concise.`;

		if (options.customInstructions) {
			instructions += `\n\nAdditional focus: ${options.customInstructions}`;
		}

		const response = await backend.generate(
			[
				{
					role: "user",
					content: `<conversation>\n${conversationText}\n</conversation>\n\n${instructions}`,
				},
			] as { role: string; content: string }[],
			{
				temperature: 0.3,
				maxTokens: Math.min(2048, (options.maxTokens ?? 4096) / 2),
				thinkingLevel: options.thinkingLevel,
			},
		);

		const summaryText = response.content?.trim();
		if (!summaryText) return null;

		const parsed = parseBranchSummary(summaryText);
		const fileOpsText = formatFileOperations(readFiles, modifiedFiles);
		const full = `${summaryText}\n${fileOpsText}`;

		return {
			goal: parsed.goal || "Branch conversation",
			constraints: parsed.constraints || [],
			progress: (parsed.progress as BranchProgress) || {
				done: [],
				inProgress: [],
				blocked: [],
			},
			keyDecisions: parsed.keyDecisions || [],
			nextSteps: parsed.nextSteps || [],
			full,
		};
	} catch (_e: unknown) {
		const fallback = `Branch ${messages.length} messages explored.`;
		return {
			goal: "Branch exploration",
			constraints: [],
			progress: { done: [], inProgress: [], blocked: [] },
			keyDecisions: [],
			nextSteps: [],
			full: fallback,
		};
	}
}
