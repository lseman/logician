/** Mutable state belonging to one interactive agent session. */
import type { SessionStore } from "../../../capabilities/session/session-store.ts";
import type {
	BranchInfo,
	BranchSummaryData,
} from "../../../capabilities/session/summaries/types.ts";
import type { ThreadItem } from "../../../capabilities/session/thread-ledger.ts";
import type { QueueMode } from "../../../system/types/types-config.ts";
import type { Message } from "../../../system/types/types-messages.ts";
import { ConversationState } from "../session/conversation-state.ts";
import type { AbortResult, HarnessQueues } from "../types.ts";
import { HarnessQueueController } from "./queue-controller.ts";

interface SessionStateOptions {
	steeringMode?: QueueMode;
	followUpMode?: QueueMode;
	onQueueChange: (queues: HarnessQueues) => void;
}

export class SessionState {
	readonly conversation = new ConversationState();
	readonly queue: HarnessQueueController;
	store?: SessionStore;
	id?: string;
	transcriptPath?: string;
	hasStarted = false;
	pendingContinuation = false;
	repositoryQuery?: string;

	constructor(options: SessionStateOptions) {
		this.queue = new HarnessQueueController(
			{
				steeringMode: options.steeringMode,
				followUpMode: options.followUpMode,
			},
			options.onQueueChange,
		);
	}

	attachStore(store: SessionStore): void {
		this.store = store;
		this.id = store.getMeta().id;
	}

	setId(id: string): void {
		this.id = id;
	}

	get messages(): Message[] {
		return this.conversation.history;
	}

	get threadItems(): readonly ThreadItem[] {
		return this.conversation.items;
	}

	checkpoint(): void {
		this.conversation.checkpoint();
	}

	clearHistory(): void {
		this.conversation.clear();
		this.hasStarted = false;
	}

	replaceHistory(messages: Message[]): void {
		this.conversation.replace(messages);
		this.hasStarted = false;
	}

	appendMessages(messages: Message[]): Message[] {
		return this.conversation.append(messages);
	}

	rewind(): { messages: number; filesRestored: number } | null {
		return this.conversation.rewind();
	}

	fork(customSummary?: BranchSummaryData): string {
		return this.conversation.fork(customSummary, this.store?.getLeafEntryId());
	}

	discardBranch(): boolean {
		const branch = this.conversation.discardBranch();
		if (!branch) return false;
		this.store?.checkout(branch.sessionLeafId);
		return true;
	}

	listBranches(): BranchInfo[] {
		return this.conversation.listBranches();
	}

	steer(text: string, interrupt: boolean, abort: () => void): void {
		this.queue.steer(text, interrupt, abort);
	}

	flushSteering(abort: () => void): number {
		return this.queue.flushSteering(abort);
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.queue.drop(displayIndex);
	}

	followUp(text: string): void {
		this.queue.followUp(text);
	}

	nextTurn(text: string): void {
		this.queue.nextTurn(text);
	}

	abortQueues(): AbortResult {
		return this.queue.abortSnapshot();
	}

	getQueues(): HarnessQueues {
		return this.queue.snapshot();
	}

	clearQueues(): HarnessQueues {
		return this.queue.clear();
	}

	setQueueMode(type: "steering" | "followUp", mode: QueueMode): void {
		this.queue.setMode(type, mode);
	}

	getQueueMode(type: "steering" | "followUp"): QueueMode {
		return this.queue.getMode(type);
	}

	setPendingContinuation(value: boolean): void {
		this.pendingContinuation = value;
	}

	takePendingContinuation(): boolean {
		if (!this.pendingContinuation) return false;
		this.pendingContinuation = false;
		return true;
	}

	setRepositoryQuery(query: string | undefined): void {
		this.repositoryQuery = query;
	}

	getRepositoryQuery(): string | undefined {
		return this.repositoryQuery;
	}
}
