export interface TaskLedgerEntry {
	id: string | number;
	subject: string;
	status: string;
}

/** Read-only task state supplied by an optional capability package. */
export interface TaskLedger {
	snapshot(): readonly TaskLedgerEntry[];
}

export const EMPTY_TASK_LEDGER: TaskLedger = {
	snapshot: () => [],
};
