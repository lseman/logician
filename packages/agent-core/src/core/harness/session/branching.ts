// ── Branch/checkpoint operations for AgentHarness ─────────────────────────
// Pure(ish) helpers operating on explicit state passed in by the harness.
// The harness owns the mutable fields (history/branches/checkpoints) and
// applies the returned results; these functions contain the logic that used
// to live directly on the AgentHarness class.

import type { LLMBackend } from "../../provider/backend.ts";
import { collectMessagesForBranchSummary } from "../../../runtime/summaries/branch-summarization.ts";
import { generateBranchSummaryText } from "../../../runtime/summaries/summary-generation.ts";
import type {
	BranchInfo,
	BranchSummaryData,
} from "../../../runtime/summaries/types.ts";
import type { Message, ThinkingLevel } from "../../types/index.ts";

export interface Branch {
	id: string;
	parent: Message[];
	forkedAt: number;
	summary: BranchSummaryData | null;
	/** Durable session entry selected at the fork point. */
	sessionLeafId?: string;
}

/** Fork the current history into a new branch. Mutates `branches` in place, returns the new branch id. */
export function forkBranch(
	_branches: Branch[],
	branchSeq: number,
	currentHistory: Message[],
	customSummary?: BranchSummaryData,
	sessionLeafId?: string,
): { branch: Branch; nextBranchSeq: number } {
	const branch: Branch = {
		id: `branch_${branchSeq + 1}`,
		parent: currentHistory,
		forkedAt: currentHistory.length,
		summary: customSummary ?? null,
		sessionLeafId,
	};
	return { branch, nextBranchSeq: branchSeq + 1 };
}

export interface BranchSummaryOutcome {
	/** New active history after the operation. */
	history: Message[];
	/** Full summary text, or null if the branch was empty / generation failed. */
	summaryText: string | null;
}

/**
 * Summarize the active branch's diverged messages and merge the summary
 * back into the parent history. Returns null (with parent history restored)
 * if nothing diverged.
 */
export async function summarizeAndMergeBranch(
	backend: LLMBackend,
	branch: Branch,
	currentHistory: Message[],
	options: {
		customInstructions?: string;
		contextWindowTokens?: number;
		maxTokens?: number;
		thinkingLevel?: ThinkingLevel;
	} = {},
): Promise<BranchSummaryOutcome> {
	const diverged = currentHistory.slice(branch.forkedAt);
	if (!diverged.length) {
		return { history: branch.parent, summaryText: null };
	}

	const tokenBudget = options.contextWindowTokens
		? Math.floor(options.contextWindowTokens * 0.5)
		: 0;
	const collection = collectMessagesForBranchSummary(
		currentHistory,
		branch.parent,
		branch.forkedAt,
		tokenBudget,
	);

	const summary = await generateBranchSummaryText(
		backend,
		collection.messages,
		{
			customInstructions: options.customInstructions,
			fileOps: collection.fileOps,
			maxTokens: options.maxTokens,
			thinkingLevel: options.thinkingLevel,
		},
	);

	branch.summary = summary;

	if (!summary) {
		return { history: branch.parent, summaryText: null };
	}

	const summaryEntry: Message = {
		role: "assistant",
		content: summary.full,
		tool_calls: [],
	};

	return {
		history: [...branch.parent, summaryEntry],
		summaryText: summary.full,
	};
}

/** Render the branch tree as an ASCII summary string. */
export function renderBranchTree(branches: Branch[]): string {
	if (branches.length === 0) {
		return "No active branches.";
	}

	const lines: string[] = [];
	lines.push(`Branches (${branches.length}):`);

	for (const branch of branches) {
		const depth = branches.indexOf(branch);
		const prefix = "  ".repeat(depth) + (depth > 0 ? "└─ " : "");
		lines.push(`${prefix}[${branch.id}] forked at message ${branch.forkedAt}`);

		if (branch.summary) {
			const goal = branch.summary.goal;
			const preview = goal.length > 60 ? `${goal.slice(0, 60)}...` : goal;
			lines.push(`${"  ".repeat(depth + 1)}Goal: ${preview}`);

			if (branch.summary.progress.done.length > 0) {
				lines.push(
					`${"  ".repeat(depth + 1)}Done: ${branch.summary.progress.done.length} items`,
				);
			}
			if (branch.summary.progress.inProgress.length > 0) {
				lines.push(
					`${"  ".repeat(depth + 1)}In Progress: ${branch.summary.progress.inProgress.length} items`,
				);
			}
			if (branch.summary.progress.blocked.length > 0) {
				lines.push(
					`${"  ".repeat(depth + 1)}Blocked: ${branch.summary.progress.blocked.length} items`,
				);
			}
		}
	}

	return lines.join("\n");
}

export function listBranches(branches: Branch[]): BranchInfo[] {
	return branches.map((b, i) => ({
		id: b.id,
		depth: i + 1,
		summary: b.summary,
		forkedAt: b.forkedAt,
	}));
}
