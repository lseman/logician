// ── Staged edits singleton ─────────────────────────────────────────────────────
// Module-level store for staged edits produced by ast_edit and hashline edit.
// Both xd://resolve and xd://reject access this store.

export interface StagedFile {
	path: string;
	content: string;
}

export interface StagedEdit {
	tool: string;
	files: StagedFile[];
}

let currentStagedEdit: StagedEdit | null = null;

/**
 * Set the current staged edits.
 */
export function setStagedEdit(edit: StagedEdit): void {
	currentStagedEdit = edit;
}

/**
 * Get the current staged edits.
 */
export function getStagedEdit(): StagedEdit | null {
	return currentStagedEdit;
}

/**
 * Clear the current staged edits.
 */
export function clearStagedEdit(): void {
	currentStagedEdit = null;
}
