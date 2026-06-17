// ── Branch summarization ──────────────────────────────────────────────────────
// When the agent explores a conversation branch (e.g., via session navigation or
// parallel subagent work) and then returns to the main thread, the abandoned
// branch content needs to be summarized so context isn't lost.
//
// This is Pi's branch summarization system, adapted for Logician's message API.

import type { AgentMessage, CompactableMessage } from "../core/types.ts";
import { estimateCompressableTokens } from "./compaction";
import {
	COMPUTE_FILE_LISTS,
	CREATE_FILE_OPS,
	EXTRACT_FILE_OPS_FROM_MESSAGE,
	FORMAT_FILE_OPERATIONS,
	serializeConversation,
} from "./utils";

// ============================================================================
// Types
// ============================================================================

/** File-operation details stored on generated branch summary entries. */
export interface BranchSummaryDetails {
	readFiles: string[];
	modifiedFiles: string[];
}

/** Prepared branch content for summarization. */
interface BranchPreparation {
	messages: CompactableMessage[];
	fileOps: ReturnType<typeof CREATE_FILE_OPS>;
	totalTokens: number;
}

/** Options for generating a branch summary. */
export interface GenerateBranchSummaryOptions {
	customInstructions?: string;
	replaceInstructions?: boolean;
	reserveTokens?: number;
}

export const DEFAULT_BRANCH_SUMMARY_OPTIONS: GenerateBranchSummaryOptions = {
	reserveTokens: 16384,
};

// ============================================================================
// Collect branch entries
// ============================================================================

/**
 * Collect messages from a branch that diverged from the main thread.
 * In Logician's model, this is used when a subagent or parallel task explores
 * a different topic and returns to the main conversation.
 */
export function collectBranchMessages(
	branchMessages: CompactableMessage[],
	contextWindow: number,
	reserveTokens: number,
): BranchPreparation {
	const tokenBudget = contextWindow - reserveTokens;
	const messages: CompactableMessage[] = [];
	const fileOps = CREATE_FILE_OPS();
	let totalTokens = 0;

	// Walk backwards from newest, respecting budget
	for (let i = branchMessages.length - 1; i >= 0; i--) {
		const msg = branchMessages[i];
		const tokens = estimateCompressableTokens(msg as AgentMessage | CompactableMessage);

		if (totalTokens + tokens > tokenBudget && tokenBudget > 0) {
			// Allow compacted/summary entries to pass with some slack
			if (msg.role === "compactionSummary" || msg.role === "branchSummary") {
				if (totalTokens < tokenBudget * 0.9) {
					messages.unshift(msg);
					totalTokens += tokens;
				}
			}
			break;
		}

		messages.unshift(msg);
		totalTokens += tokens;

		// Extract file operations
		EXTRACT_FILE_OPS_FROM_MESSAGE(msg, fileOps);
	}

	return { messages, fileOps, totalTokens };
}

// ============================================================================
// Summarization prompts
// ============================================================================

const BRANCH_SUMMARY_PREAMBLE =
	"The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

const BRANCH_SUMMARY_PROMPT = `Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

// ============================================================================
// Generate branch summary
// ============================================================================

/** Generate a summary for abandoned branch entries. */
export async function generateBranchSummary(
	branchMessages: CompactableMessage[],
	options: GenerateBranchSummaryOptions = DEFAULT_BRANCH_SUMMARY_OPTIONS,
): Promise<{
	summary: string;
	readFiles: string[];
	modifiedFiles: string[];
}> {
	const {
		customInstructions,
		replaceInstructions,
		reserveTokens = DEFAULT_BRANCH_SUMMARY_OPTIONS.reserveTokens,
	} = options;

	const { messages, fileOps } = collectBranchMessages(
		branchMessages,
		128000, // default context window
		reserveTokens ?? 16384,
	);

	if (messages.length === 0) {
		return {
			summary: "No content to summarize",
			readFiles: [],
			modifiedFiles: [],
		};
	}

	const llmMessages = convertMessagesToLlmFormat(messages);
	const conversationText = serializeConversation(
		llmMessages as Array<{
			role: string;
			content: string | Array<{ type: string; text?: string }>;
		}>,
	);

	let instructions: string;
	if (replaceInstructions && customInstructions) {
		instructions = customInstructions;
	} else if (customInstructions) {
		instructions = `${BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: ${customInstructions}`;
	} else {
		instructions = BRANCH_SUMMARY_PROMPT;
	}

	const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${instructions}`;

	// Placeholder for actual LLM call
	const summary = `${BRANCH_SUMMARY_PREAMBLE}[Branch summary would be generated here by calling the LLM with the conversation above.\n\nPrompt length: ${promptText.length} chars]`;

	const { readFiles, modifiedFiles } = COMPUTE_FILE_LISTS(fileOps);
	const fullSummary =
		summary + FORMAT_FILE_OPERATIONS(readFiles, modifiedFiles);

	return {
		summary: fullSummary || "No summary generated",
		readFiles,
		modifiedFiles,
	};
}

function convertMessagesToLlmFormat(messages: CompactableMessage[]): Array<{
	role: string;
	content: string | Array<{ type: string; text?: string }>;
}> {
	const result: Array<{
		role: string;
		content: string | Array<{ type: string; text?: string }>;
	}> = [];

	for (const msg of messages) {
		if (msg.role === "assistant" && Array.isArray(msg.content)) {
			const textParts = msg.content
				.filter(
					(b: unknown) =>
						typeof b === "object" &&
						b !== null &&
						"type" in b &&
						b.type === "text",
				)
				.map((b: unknown) => ({
					type: "text" as const,
					text: (b as { text: string }).text,
				}));
			result.push({
				role: msg.role,
				content: textParts.length > 0 ? textParts : "",
			});
		} else {
			result.push({
				role: msg.role,
				content: typeof msg.content === "string" ? msg.content : "",
			});
		}
	}

	return result;
}
