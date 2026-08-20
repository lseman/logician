// ── ConversationState ────────────────────────────────────────────────────
// Owns the active message history, rewind checkpoints, and branch stack for
// one AgentHarness instance. File-frame coordination (beginFileFrame /
// restoreFileFrame / clearFileFrames) is paired here so a conversation
// checkpoint and its workspace snapshot can never drift apart — every method
// that pushes/pops/clears history does the matching file-frame operation in
// the same call.
//
// Session persistence (the durable on-disk log) is a separate concern owned
// by the harness: it depends on hook-lifecycle state (session id, transcript
// path) that has nothing to do with in-memory conversation shape.

import {
	beginFileFrame,
	clearFileFrames,
	restoreFileFrame,
} from "../../session/file-checkpoints.ts";
import type {
	BranchInfo,
	BranchSummaryData,
} from "../../../runtime/summaries/types.ts";
import type { Message } from "../../types/index.ts";
import {
	type Branch,
	forkBranch,
	listBranches as listBranchesHelper,
} from "./branching.ts";

// Bounded ring (newest last) so rewind has a bounded number of prior turns
// to restore from without growing memory unboundedly across a long session.
const MAX_CHECKPOINTS = 20;

export class ConversationState {
	private _history: Message[] = [];
	private checkpoints: Message[][] = [];
	private branches: Branch[] = [];
	private branchSeq = 0;

	get history(): Message[] {
		return this._history;
	}

	set history(messages: Message[]) {
		this._history = messages;
	}

	/** Push a checkpoint (conversation + file frame) before a turn begins. */
	checkpoint(): void {
		this.checkpoints.push([...this._history]);
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
		this._history = snapshot;
		const filesRestored = restoreFileFrame() ?? 0;
		return { messages: snapshot.length, filesRestored };
	}

	/** Drop all history, checkpoints, branches, and file frames. */
	clear(): void {
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this._history = [];
	}

	/** Replace history wholesale (resume/switch), dropping branches/checkpoints/frames. */
	replace(messages: Message[]): void {
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this._history = messages.filter(
			(m): m is Message => m != null && m.role !== "system",
		);
	}

	/** Append messages without touching checkpoints/branches. */
	append(messages: Message[]): Message[] {
		const toAdd = messages.filter(
			(m): m is Message => m != null && m.role !== "system",
		);
		if (toAdd.length) this._history = [...this._history, ...toAdd];
		return toAdd;
	}

	// ── Branching ────────────────────────────────────────────────────────

	fork(customSummary?: BranchSummaryData, sessionLeafId?: string): string {
		const current = this._history;
		const { branch, nextBranchSeq } = forkBranch(
			this.branches,
			this.branchSeq,
			current,
			customSummary,
			sessionLeafId,
		);
		this.branchSeq = nextBranchSeq;
		this.branches.push(branch);
		this._history = [...current];
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
		this._history = branch.parent;
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
