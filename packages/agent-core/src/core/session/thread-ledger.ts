import type { Message } from "../types/types-messages.ts";

export type ThreadReplacementReason =
	| "restore"
	| "rewind"
	| "branch-discard"
	| "branch-merge"
	| "compaction"
	| "clear"
	| "run-commit";

export type ThreadItem =
	| {
			id: string;
			sequence: number;
			type: "message";
			message: Message;
	  }
	| {
			id: string;
			sequence: number;
			type: "projection";
			reason: ThreadReplacementReason;
			messages: readonly Message[];
	  };

function cloneMessage(message: Message): Message {
	return {
		...message,
		tool_calls: message.tool_calls?.map(call => ({ ...call })),
	};
}

/**
 * Append-only record of conversation changes with a current message projection.
 * Rewinds and compactions append projection items instead of erasing provenance.
 */
export class ThreadLedger {
	private readonly entries: ThreadItem[] = [];
	private projection: Message[] = [];
	private sequence = 0;

	get messages(): Message[] {
		return this.projection.map(cloneMessage);
	}

	items(): readonly ThreadItem[] {
		return this.entries.map(item =>
			item.type === "message"
				? { ...item, message: cloneMessage(item.message) }
				: { ...item, messages: item.messages.map(cloneMessage) },
		);
	}

	append(messages: readonly Message[]): Message[] {
		const accepted = messages
			.filter((message): message is Message => Boolean(message))
			.map(cloneMessage);
		for (const message of accepted) {
			const sequence = ++this.sequence;
			this.entries.push({
				id: `item-${sequence}`,
				sequence,
				type: "message",
				message,
			});
		}
		this.projection = [...this.projection, ...accepted];
		return accepted.map(cloneMessage);
	}

	replace(messages: readonly Message[], reason: ThreadReplacementReason): void {
		const next = messages
			.filter((message): message is Message => Boolean(message))
			.map(cloneMessage);
		const sequence = ++this.sequence;
		this.entries.push({
			id: `item-${sequence}`,
			sequence,
			type: "projection",
			reason,
			messages: next,
		});
		this.projection = next;
	}
}
