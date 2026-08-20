import type { Message } from "../../types/index.ts";
import {
	type Branch,
	forkBranch,
	listBranches,
	renderBranchTree,
} from "../branching.ts";
import type { BranchInfo, BranchSummaryData } from "../summaries/types.ts";
import {
	beginFileFrame,
	clearFileFrames,
	restoreFileFrame,
} from "../utils/file-checkpoints.ts";

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

	checkpoint(): void {
		this.checkpoints.push([...this._history]);
		if (this.checkpoints.length > MAX_CHECKPOINTS) this.checkpoints.shift();
		beginFileFrame();
	}

	rewind(): { messages: number; filesRestored: number } | null {
		const snapshot = this.checkpoints.pop();
		if (!snapshot) return null;
		this.branches = [];
		this._history = snapshot;
		const filesRestored = restoreFileFrame() ?? 0;
		return { messages: snapshot.length, filesRestored };
	}

	clear(): void {
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this._history = [];
	}

	replace(messages: Message[]): void {
		this.branches = [];
		this.checkpoints = [];
		clearFileFrames();
		this._history = messages.filter(
			(m): m is Message => m != null && m.role !== "system",
		);
	}

	append(messages: Message[]): Message[] {
		const toAdd = messages.filter(
			(m): m is Message => m != null && m.role !== "system",
		);
		if (toAdd.length) this._history = [...this._history, ...toAdd];
		return toAdd;
	}

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

	activeBranch(): Branch | undefined {
		return this.branches.at(-1);
	}

	discardBranch(): Branch | undefined {
		const branch = this.branches.pop();
		if (!branch) return undefined;
		this._history = branch.parent;
		return branch;
	}

	popBranch(): Branch | undefined {
		return this.branches.pop();
	}

	branchTree(): string {
		return renderBranchTree(this.branches);
	}

	listBranches(): BranchInfo[] {
		return listBranches(this.branches);
	}
}
