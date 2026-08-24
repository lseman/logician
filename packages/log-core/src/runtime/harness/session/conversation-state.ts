// ── ConversationState ────────────────────────────────────────────────────
// Owns the active message history, rewind checkpoints, and branch stack for
// one AgentSession instance. File-frame coordination (beginFileFrame /
// restoreFileFrame / clearFileFrames) is paired here so a conversation
// checkpoint and its workspace snapshot can never drift apart — every method
// that pushes/pops/clears history does the matching file-frame operation in
// the same call.
//
// Durable persistence is attached by AgentSession. ConversationState remains
// storage-agnostic and only maintains the in-memory conversation shape.

import {
	beginFileFrame,
	clearFileFrames,
	restoreFileFrame,
} from "../../../capabilities/session/file-checkpoints.ts";
import type {
	BranchInfo,
	BranchSummaryData,
} from "../../../capabilities/session/summaries/types.ts";
import {
	ThreadLedger,
	type ThreadReplacementReason,
} from "../../../capabilities/session/thread-ledger.ts";
import type { Message } from "../../../system/types/types-messages.ts";
import {
	type Branch,
	forkBranch,
	listBranches as listBranchesHelper,
} from "./branching.ts";

// Bounded ring (newest last) so rewind has a bounded number of prior turns
// to restore from without growing memory unboundedly across a long session.
const MAX_CHECKPOINTS = 20;

export class ConversationState {
	private readonly ledger = new ThreadLedger();
	private checkpoints: Message[][] = [];
	private branches: Branch[] = [];
	private branchSeq = 0;

	get history(): Message[] {
		return this.ledger.messages;
	}

	set history(messages: Message[]) {
		this.ledger.replace(messages, "run-commit");
	}

	get items() {
		return this.ledger.items();
	}

	private replaceHistory(
		messages: readonly Message[],
		reason: ThreadReplacementReason,
	): void {
		this.ledger.replace(messages, reason);
	}

	/** Push a checkpoint (conversation + file frame) before a turn begins. */
	checkpoint(): void {
		this.checkpoints.push(this.history);
		if (this.checkpoints.length > MAX_CHECKPOINTS) {
			this.checkpoints.shift();
		}
		beginFileFrame();
	}

	/** Restore the most recent checkpoint's conversation and files. */
	rewind(): { messages: number; filesRestored: number } | null {
		const snapshot = this.checkpoints.pop();
		if (!snapshot) return null;
		this.branches = [];
		this.replaceHistory(snapshot, "rewind");
		const filesRestored = restoreFileFrame() ?? 0;
		return { messages: snapshot.length, filesRestored };
	}

	/** Drop all history, checkpoints, branches, and file frames. */
	clear(): void {
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this.replaceHistory([], "clear");
	}

	/** Replace history wholesale (resume/switch), dropping branches/checkpoints/frames. */
	replace(messages: Message[]): void {
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this.replaceHistory(
			messages.filter((message): message is Message =>
				Boolean(message && message.role !== "system"),
			),
			"restore",
		);
	}

	/** Append messages without touching checkpoints/branches. */
	append(messages: Message[]): Message[] {
		return this.ledger.append(
			messages.filter((message): message is Message =>
				Boolean(message && message.role !== "system"),
			),
		);
	}

	// ── Branching ────────────────────────────────────────────────────────

	fork(customSummary?: BranchSummaryData, sessionLeafId?: string): string {
		const current = this.history;
		const { branch, nextBranchSeq } = forkBranch(
			this.branches,
			this.branchSeq,
			current,
			customSummary,
			sessionLeafId,
		);
		this.branchSeq = nextBranchSeq;
		this.branches.push(branch);
		return branch.id;
	}

	/** The active branch, if any (fork target for branchSummary/discardBranch). */
	activeBranch(): Branch | undefined {
		return this.branches.at(-1);
	}

	/** Pop the active branch and restore its parent history without merging. */
	discardBranch(): Branch | undefined {
		const branch = this.branches.pop();
		if (!branch) return undefined;
		this.replaceHistory(branch.parent, "branch-discard");
		return branch;
	}

	/** Pop the active branch without restoring history (caller sets the merged result). */
	popBranch(): Branch | undefined {
		return this.branches.pop();
	}

	listBranches(): BranchInfo[] {
		return listBranchesHelper(this.branches);
	}
}
