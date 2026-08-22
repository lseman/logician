/** Durable background semantic-extraction job queue. Jobs survive crashes
 * and are reclaimed via lease/fencing on the next claim rather than
 * delaying interactive turns. */

import type { Database } from "bun:sqlite";
import type { ExtractionJob, ExtractionJobStatus } from "../types.js";
import {
	normalizeWorkspacePath,
	now,
	sanitizePayload,
	sanitizeString,
} from "./module-helpers.ts";

type ExtractionJobRow = {
	id: string;
	session_id: string;
	workspace: string;
	payload: string;
	status: ExtractionJobStatus;
	attempts: number;
	created_at: string;
	updated_at: string;
	next_attempt_at: string;
	last_error: string | null;
	owner_id: string | null;
	lease_until: string | null;
	fencing_token: number;
};

const rowToExtractionJob = (row: ExtractionJobRow): ExtractionJob => ({
	id: row.id,
	sessionId: row.session_id,
	workspace: row.workspace,
	payload: row.payload,
	status: row.status,
	attempts: row.attempts,
	createdAt: row.created_at,
	updatedAt: row.updated_at,
	nextAttemptAt: row.next_attempt_at,
	lastError: row.last_error || undefined,
	ownerId: row.owner_id || undefined,
	leaseUntil: row.lease_until || undefined,
	fencingToken: row.fencing_token || 0,
});

const DEFAULT_EXTRACTION_LEASE_MS = 30_000;

/** Random per-process worker identity, distinguishing this process's leased
 * jobs from another process's when multiple hosts share one database. */
const extractionWorkerId = crypto.randomUUID();

export function enqueueExtractionJob(
	db: Database,
	sessionId: string,
	workspace: string,
	payload: string,
): ExtractionJob {
	const id = crypto.randomUUID();
	const timestamp = now();
	const safePayload = (() => {
		try {
			return JSON.stringify(sanitizePayload(JSON.parse(payload)));
		} catch {
			return JSON.stringify({ invalidPayload: sanitizeString(payload) });
		}
	})();
	db.prepare(
		`INSERT INTO extraction_jobs
      (id, session_id, workspace, payload, status, attempts, created_at, updated_at, next_attempt_at)
      VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?)`,
	).run(
		id,
		sessionId,
		normalizeWorkspacePath(workspace),
		safePayload,
		timestamp,
		timestamp,
		timestamp,
	);
	db.prepare(
		"DELETE FROM extraction_jobs WHERE status = 'completed' AND updated_at < ?",
	).run(new Date(Date.now() - 7 * 86_400_000).toISOString());
	return rowToExtractionJob(
		db
			.prepare("SELECT * FROM extraction_jobs WHERE id = ?")
			.get(id) as ExtractionJobRow,
	);
}

export function claimExtractionJob(
	db: Database,
	getWorkspace: () => string,
	leaseMs: number = DEFAULT_EXTRACTION_LEASE_MS,
): ExtractionJob | null {
	const timestamp = now();
	const leaseUntil = new Date(
		Date.now() + Math.max(0, leaseMs),
	).toISOString();
	const row = db
		.prepare(
			`UPDATE extraction_jobs
		  SET status = 'running', attempts = attempts + 1, updated_at = ?,
		      owner_id = ?, lease_until = ?, fencing_token = fencing_token + 1
		  WHERE id = (
		    SELECT id FROM extraction_jobs
		    WHERE workspace = ? AND next_attempt_at <= ?
		      AND (status = 'pending' OR (status = 'running' AND lease_until <= ?))
		    ORDER BY created_at ASC LIMIT 1
		  )
		  RETURNING *`,
		)
		.get(
			timestamp,
			extractionWorkerId,
			leaseUntil,
			getWorkspace(),
			timestamp,
			timestamp,
		) as ExtractionJobRow | undefined;
	return row ? rowToExtractionJob(row) : null;
}

export function completeExtractionJob(db: Database, id: string): void {
	const timestamp = now();
	db.prepare(
		"UPDATE extraction_jobs SET status = 'completed', updated_at = ?, last_error = NULL, lease_until = NULL WHERE id = ? AND status = 'running' AND owner_id = ?",
	).run(timestamp, id, extractionWorkerId);
}

export function renewExtractionJob(
	db: Database,
	id: string,
	leaseMs: number = DEFAULT_EXTRACTION_LEASE_MS,
): boolean {
	const timestamp = now();
	const leaseUntil = new Date(
		Date.now() + Math.max(1, leaseMs),
	).toISOString();
	const updated = db
		.prepare(
			`UPDATE extraction_jobs
		  SET updated_at = ?, lease_until = ?
		  WHERE id = ? AND status = 'running' AND owner_id = ?`,
		)
		.run(timestamp, leaseUntil, id, extractionWorkerId);
	return updated.changes === 1;
}

export function failExtractionJob(
	db: Database,
	id: string,
	error: string,
	retryDelayMs: number = 1_000,
): void {
	const row = db
		.prepare(
			"SELECT attempts FROM extraction_jobs WHERE id = ? AND status = 'running' AND owner_id = ?",
		)
		.get(id, extractionWorkerId) as { attempts: number } | undefined;
	if (!row) return;
	const terminal = row.attempts >= 3;
	const timestamp = now();
	const nextAttempt = new Date(
		Date.now() + Math.max(0, retryDelayMs),
	).toISOString();
	db.prepare(
		`UPDATE extraction_jobs SET status = ?, updated_at = ?, next_attempt_at = ?, last_error = ?, lease_until = NULL
		  WHERE id = ? AND status = 'running' AND owner_id = ?`,
	).run(
		terminal ? "failed" : "pending",
		timestamp,
		nextAttempt,
		error.slice(0, 1000),
		id,
		extractionWorkerId,
	);
}

export function listExtractionJobs(
	db: Database,
	getWorkspace: () => string,
	status?: ExtractionJobStatus,
): ExtractionJob[] {
	const rows = status
		? db
				.prepare(
					"SELECT * FROM extraction_jobs WHERE workspace = ? AND status = ? ORDER BY created_at",
				)
				.all(getWorkspace(), status)
		: db
				.prepare(
					"SELECT * FROM extraction_jobs WHERE workspace = ? ORDER BY created_at",
				)
				.all(getWorkspace());
	return (rows as ExtractionJobRow[]).map(rowToExtractionJob);
}
