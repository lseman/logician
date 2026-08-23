/** Creates the store's tables/indices/triggers and runs its migrations.
 * Pure db operations — no workspace or other closure state. */

import type { Database } from "bun:sqlite";
import { normalizeWorkspacePath } from "./module-helpers.ts";

export function initMemoryStoreSchema(db: Database): void {
	db.exec(`
    -- Sessions: lifecycle tracking
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      name TEXT,
      project TEXT NOT NULL DEFAULT '',
      cwd TEXT NOT NULL DEFAULT '',
      workspace TEXT NOT NULL DEFAULT '',
      started_at TEXT NOT NULL,
      ended_at TEXT,
      status TEXT NOT NULL DEFAULT 'active',
      observation_count INTEGER NOT NULL DEFAULT 0,
      model TEXT,
      tags TEXT NOT NULL DEFAULT '[]',
      first_prompt TEXT,
      summary TEXT,
      commit_shas TEXT NOT NULL DEFAULT '[]'
    );

    -- Observations: raw + compressed
    CREATE TABLE IF NOT EXISTS observations (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      hook_type TEXT NOT NULL,
      type TEXT NOT NULL DEFAULT 'other',
      title TEXT NOT NULL DEFAULT '',
      subtitle TEXT,
      narrative TEXT NOT NULL DEFAULT '',
      facts TEXT NOT NULL DEFAULT '[]',
      concepts TEXT NOT NULL DEFAULT '[]',
      files TEXT NOT NULL DEFAULT '[]',
      importance INTEGER NOT NULL DEFAULT 5,
      workspace TEXT NOT NULL DEFAULT '',
      consolidated INTEGER NOT NULL DEFAULT 0,
      claims TEXT NOT NULL DEFAULT '[]',
      provenance TEXT,
      raw_data TEXT,
      FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    -- Memories: long-term structured knowledge
    CREATE TABLE IF NOT EXISTS memories (
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      type TEXT NOT NULL DEFAULT 'fact',
      title TEXT NOT NULL DEFAULT '',
      content TEXT NOT NULL,
      concepts TEXT NOT NULL DEFAULT '[]',
      files TEXT NOT NULL DEFAULT '[]',
      session_ids TEXT NOT NULL DEFAULT '[]',
      strength INTEGER NOT NULL DEFAULT 5,
      version INTEGER NOT NULL DEFAULT 1,
      parent_id TEXT,
      related_ids TEXT NOT NULL DEFAULT '[]',
      source_observation_ids TEXT NOT NULL DEFAULT '[]',
      is_latest INTEGER NOT NULL DEFAULT 1,
      project TEXT,
      workspace TEXT NOT NULL DEFAULT '',
      access_count INTEGER NOT NULL DEFAULT 0,
      last_accessed TEXT,
      working_tier TEXT NOT NULL DEFAULT 'cold',
      supersedes TEXT NOT NULL DEFAULT '[]'
    );

    -- Dedup: hash-based deduplication within a time window
    CREATE TABLE IF NOT EXISTS dedup (
      hash TEXT PRIMARY KEY,
      created_at TEXT NOT NULL
    );

    -- Indices for efficient querying
    CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id);
    CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(type);
    CREATE INDEX IF NOT EXISTS idx_observations_timestamp ON observations(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_observations_importance ON observations(importance DESC);
    CREATE INDEX IF NOT EXISTS idx_observations_narrative ON observations(narrative);
    CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
    CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength DESC);
    CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
    CREATE INDEX IF NOT EXISTS idx_memories_is_latest ON memories(is_latest);
    CREATE INDEX IF NOT EXISTS idx_memories_working_tier ON memories(working_tier);
    CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count DESC);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
    CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);

    -- Memory Relations
    CREATE TABLE IF NOT EXISTS relations (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL CHECK(type IN ('supersedes', 'contradicts', 'related_to', 'supports', 'extends')),
      source_id TEXT NOT NULL,
      target_id TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 0.5,
      created_at TEXT NOT NULL,
      FOREIGN KEY (source_id) REFERENCES memories(id),
      FOREIGN KEY (target_id) REFERENCES memories(id)
    );
    CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
    CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
    CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(type);

    -- Durable background semantic-extraction queue. Jobs survive crashes and
    -- are reclaimed on the next startup without delaying interactive turns.
    CREATE TABLE IF NOT EXISTS extraction_jobs (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      workspace TEXT NOT NULL DEFAULT '',
      payload TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending',
      attempts INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      next_attempt_at TEXT NOT NULL,
      last_error TEXT,
      owner_id TEXT,
      lease_until TEXT,
      fencing_token INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_extraction_jobs_ready
      ON extraction_jobs(workspace, status, next_attempt_at, created_at);

    -- Append-only truth layer derived from immutable observations. Claim rows
    -- are never updated in place except to close validity or link a successor.
    CREATE TABLE IF NOT EXISTS claims (
      id TEXT PRIMARY KEY,
      workspace TEXT NOT NULL,
      observation_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      text TEXT NOT NULL,
      status TEXT NOT NULL CHECK(status IN ('tentative', 'verified', 'invalidated')),
      confidence REAL NOT NULL,
      operation TEXT NOT NULL CHECK(operation IN ('ADD', 'SUPERSEDE', 'INVALIDATE', 'NOOP')),
      valid_from TEXT NOT NULL,
      valid_to TEXT,
      transaction_time TEXT NOT NULL,
      source TEXT NOT NULL CHECK(source IN ('model', 'deterministic')),
      trust TEXT NOT NULL CHECK(trust IN ('trusted_local', 'external', 'untrusted')),
      extractor_version TEXT NOT NULL,
      schema_version INTEGER NOT NULL,
      supersedes_claim_id TEXT,
      superseded_by_claim_id TEXT,
      tombstoned_at TEXT,
	  lifecycle TEXT NOT NULL DEFAULT 'probationary',
	  validity_predicates TEXT NOT NULL DEFAULT '[]',
	  evidence_certificate TEXT NOT NULL DEFAULT '{}',
      FOREIGN KEY (observation_id) REFERENCES observations(id) ON DELETE CASCADE,
      FOREIGN KEY (supersedes_claim_id) REFERENCES claims(id),
      FOREIGN KEY (superseded_by_claim_id) REFERENCES claims(id)
    );
    CREATE TABLE IF NOT EXISTS claim_evidence (
      claim_id TEXT NOT NULL,
      observation_id TEXT NOT NULL,
      evidence_event_id TEXT NOT NULL,
      PRIMARY KEY (claim_id, evidence_event_id),
      FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE,
      FOREIGN KEY (observation_id) REFERENCES observations(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_claims_workspace_status
      ON claims(workspace, status, transaction_time DESC);
    CREATE INDEX IF NOT EXISTS idx_claims_observation ON claims(observation_id);
    CREATE INDEX IF NOT EXISTS idx_claim_evidence_observation
      ON claim_evidence(observation_id);

    CREATE TABLE IF NOT EXISTS memory_embeddings (
      entity_id TEXT PRIMARY KEY,
      entity_kind TEXT NOT NULL CHECK(entity_kind IN ('observation', 'memory')),
      session_id TEXT,
      workspace TEXT NOT NULL DEFAULT '',
      dimensions INTEGER NOT NULL,
	  model TEXT NOT NULL DEFAULT 'unknown',
	  content_hash TEXT NOT NULL DEFAULT '',
	  creation_version INTEGER NOT NULL DEFAULT 1,
	  vector_bucket TEXT NOT NULL DEFAULT '',
      vector TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memory_embeddings_workspace
      ON memory_embeddings(workspace, dimensions, entity_kind);
    CREATE INDEX IF NOT EXISTS idx_memory_embeddings_recency
      ON memory_embeddings(workspace, dimensions, updated_at DESC);

	CREATE TABLE IF NOT EXISTS retrieval_traces (
	  id TEXT PRIMARY KEY,
	  workspace TEXT NOT NULL,
	  session_id TEXT NOT NULL,
	  objective TEXT NOT NULL,
	  created_at TEXT NOT NULL,
	  latency_ms REAL NOT NULL,
	  budget INTEGER NOT NULL,
	  tokens INTEGER NOT NULL,
	  abstained INTEGER NOT NULL,
	  reason TEXT,
	  candidate_counts TEXT NOT NULL,
	  selected TEXT NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_retrieval_traces_workspace
	  ON retrieval_traces(workspace, created_at DESC);

	CREATE TABLE IF NOT EXISTS memory_outcome_receipts (
	  id TEXT PRIMARY KEY,
	  workspace TEXT NOT NULL,
	  retrieval_trace_id TEXT NOT NULL,
	  task_id TEXT NOT NULL,
	  trial_id TEXT,
	  created_at TEXT NOT NULL,
	  selected_ids TEXT NOT NULL,
	  policy_version INTEGER NOT NULL,
	  outcome TEXT NOT NULL,
	  reward REAL NOT NULL,
	  FOREIGN KEY (retrieval_trace_id) REFERENCES retrieval_traces(id) ON DELETE CASCADE
	);
	CREATE INDEX IF NOT EXISTS idx_memory_receipts_workspace
	  ON memory_outcome_receipts(workspace, created_at DESC);
	CREATE TABLE IF NOT EXISTS memory_policy_state (
	  workspace TEXT PRIMARY KEY,
	  version INTEGER NOT NULL,
	  mode TEXT NOT NULL CHECK(mode IN ('deterministic', 'shadow')),
	  weights TEXT NOT NULL,
	  samples INTEGER NOT NULL,
	  updated_at TEXT NOT NULL
	);
  `);

	runMigrations(db);
}

function runMigrations(db: Database): void {
	for (const [column, definition] of [
		["owner_id", "TEXT"],
		["lease_until", "TEXT"],
		["fencing_token", "INTEGER NOT NULL DEFAULT 0"],
	] as const) {
		const columns = db
			.prepare("PRAGMA table_info(extraction_jobs)")
			.all() as Array<{ name: string }>;
		if (!columns.some(item => item.name === column))
			db.exec(`ALTER TABLE extraction_jobs ADD COLUMN ${column} ${definition}`);
	}
	for (const [column, definition] of [
		["lifecycle", "TEXT NOT NULL DEFAULT 'probationary'"],
		["validity_predicates", "TEXT NOT NULL DEFAULT '[]'"],
		["evidence_certificate", "TEXT NOT NULL DEFAULT '{}'"],
	] as const) {
		const columns = db.prepare("PRAGMA table_info(claims)").all() as Array<{
			name: string;
		}>;
		if (!columns.some(item => item.name === column))
			db.exec(`ALTER TABLE claims ADD COLUMN ${column} ${definition}`);
	}
	// Add workspace columns to existing databases that were created before
	// the workspace scoping feature.
	for (const table of ["sessions", "observations", "memories"]) {
		try {
			const cols = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{
				name: string;
			}>;
			if (!cols.find(c => c.name === "workspace")) {
				db.exec(
					`ALTER TABLE ${table} ADD COLUMN workspace TEXT NOT NULL DEFAULT ''`,
				);
			}
			try {
				db.exec(
					`CREATE INDEX IF NOT EXISTS idx_${table}_workspace ON ${table}(workspace)`,
				);
			} catch {}
		} catch {}
	}
	for (const [column, definition] of [
		["model", "TEXT NOT NULL DEFAULT 'unknown'"],
		["content_hash", "TEXT NOT NULL DEFAULT ''"],
		["creation_version", "INTEGER NOT NULL DEFAULT 1"],
		["vector_bucket", "TEXT NOT NULL DEFAULT ''"],
	] as const) {
		const columns = db
			.prepare("PRAGMA table_info(memory_embeddings)")
			.all() as Array<{ name: string }>;
		if (!columns.some(item => item.name === column))
			db.exec(
				`ALTER TABLE memory_embeddings ADD COLUMN ${column} ${definition}`,
			);
	}
	db.exec(`CREATE INDEX IF NOT EXISTS idx_memory_embeddings_bucket
	  ON memory_embeddings(workspace, dimensions, vector_bucket)`);
	try {
		const sessionCols = db
			.prepare("PRAGMA table_info(sessions)")
			.all() as Array<{ name: string }>;
		if (!sessionCols.some(column => column.name === "name")) {
			db.exec("ALTER TABLE sessions ADD COLUMN name TEXT");
		}
	} catch {}
	// Add consolidated column to observations
	try {
		const obsCols = db
			.prepare(`PRAGMA table_info(observations)`)
			.all() as Array<{ name: string }>;
		if (!obsCols.find(c => c.name === "consolidated")) {
			db.exec(
				`ALTER TABLE observations ADD COLUMN consolidated INTEGER NOT NULL DEFAULT 0`,
			);
		}
		if (!obsCols.find(c => c.name === "claims")) {
			db.exec(
				"ALTER TABLE observations ADD COLUMN claims TEXT NOT NULL DEFAULT '[]'",
			);
		}
		if (!obsCols.find(c => c.name === "provenance")) {
			db.exec("ALTER TABLE observations ADD COLUMN provenance TEXT");
		}
	} catch {}
	// Create consolidated index
	try {
		db.exec(
			`CREATE INDEX IF NOT EXISTS idx_observations_consolidated ON observations(consolidated)`,
		);
	} catch {}

	// Enforce at most one is_latest=1 memory per (workspace, title). Without
	// this, two concurrent consolidate() calls (e.g. the extraction worker and
	// a direct turn-end consolidation) can both read "no existing memory with
	// this title" before either writes, and both insert instead of one
	// superseding the other. Existing databases may already have duplicates
	// from before this constraint existed, so they're resolved (keep the most
	// recently updated row as latest) before the index is created — otherwise
	// CREATE UNIQUE INDEX would fail outright on those rows.
	try {
		db.exec(`
      UPDATE memories SET is_latest = 0
      WHERE is_latest = 1 AND id NOT IN (
        SELECT id FROM (
          SELECT id, ROW_NUMBER() OVER (
            PARTITION BY workspace, title ORDER BY updated_at DESC, rowid DESC
          ) AS rn
          FROM memories WHERE is_latest = 1
        ) WHERE rn = 1
      )
    `);
		db.exec(`CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_latest_title
      ON memories(workspace, title) WHERE is_latest = 1`);
	} catch {}

	// FTS5 indices are kept separate from the source tables so existing
	// databases can be upgraded in place. Triggers keep them current.
	db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
      id UNINDEXED, title, narrative, facts, concepts, files,
      tokenize = 'unicode61 remove_diacritics 2'
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
      id UNINDEXED, title, content, concepts, files,
      tokenize = 'unicode61 remove_diacritics 2'
    );
    CREATE TRIGGER IF NOT EXISTS observations_fts_insert AFTER INSERT ON observations BEGIN
      INSERT INTO observations_fts(id, title, narrative, facts, concepts, files)
      VALUES (new.id, new.title, new.narrative, new.facts, new.concepts, new.files);
    END;
    CREATE TRIGGER IF NOT EXISTS observations_fts_delete AFTER DELETE ON observations BEGIN
      DELETE FROM observations_fts WHERE id = old.id;
    END;
    CREATE TRIGGER IF NOT EXISTS observations_fts_update AFTER UPDATE ON observations BEGIN
      DELETE FROM observations_fts WHERE id = old.id;
      INSERT INTO observations_fts(id, title, narrative, facts, concepts, files)
      VALUES (new.id, new.title, new.narrative, new.facts, new.concepts, new.files);
    END;
    CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
      INSERT INTO memories_fts(id, title, content, concepts, files)
      VALUES (new.id, new.title, new.content, new.concepts, new.files);
    END;
    CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
      DELETE FROM memories_fts WHERE id = old.id;
    END;
    CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
      DELETE FROM memories_fts WHERE id = old.id;
      INSERT INTO memories_fts(id, title, content, concepts, files)
      VALUES (new.id, new.title, new.content, new.concepts, new.files);
    END;
  `);
	// Incrementally backfill databases created before FTS5 support.
	db.exec(`
    INSERT INTO observations_fts(id, title, narrative, facts, concepts, files)
      SELECT o.id, o.title, o.narrative, o.facts, o.concepts, o.files
      FROM observations o
      WHERE NOT EXISTS (SELECT 1 FROM observations_fts f WHERE f.id = o.id);
    INSERT INTO memories_fts(id, title, content, concepts, files)
      SELECT m.id, m.title, m.content, m.concepts, m.files
      FROM memories m
      WHERE NOT EXISTS (SELECT 1 FROM memories_fts f WHERE f.id = m.id);
  `);

	// Recover path scope for databases written before workspace became
	// mandatory. Session cwd is authoritative; observations inherit it.
	try {
		const legacySessions = db
			.prepare(
				"SELECT id, cwd FROM sessions WHERE workspace = '' AND cwd != ''",
			)
			.all() as Array<{ id: string; cwd: string }>;
		const updateSessionWorkspace = db.prepare(
			"UPDATE sessions SET workspace = ? WHERE id = ?",
		);
		for (const session of legacySessions) {
			updateSessionWorkspace.run(
				normalizeWorkspacePath(session.cwd),
				session.id,
			);
		}
		db.exec(`
      UPDATE observations
      SET workspace = COALESCE(
        (SELECT sessions.workspace FROM sessions WHERE sessions.id = observations.session_id),
        ''
      )
      WHERE workspace = ''
    `);
		db.exec(`
      UPDATE memories
      SET workspace = COALESCE((
        SELECT sessions.workspace
        FROM json_each(memories.session_ids)
        JOIN sessions ON sessions.id = json_each.value
        WHERE sessions.workspace != ''
        LIMIT 1
      ), '')
      WHERE workspace = '' AND json_valid(session_ids)
    `);

		for (const table of ["sessions", "observations", "memories"]) {
			const scopes = db
				.prepare(
					`SELECT DISTINCT workspace FROM ${table} WHERE workspace != ''`,
				)
				.all() as Array<{ workspace: string }>;
			const normalizeStoredScope = db.prepare(
				`UPDATE ${table} SET workspace = ? WHERE workspace = ?`,
			);
			for (const scope of scopes) {
				const normalized = normalizeWorkspacePath(scope.workspace);
				if (normalized !== scope.workspace) {
					normalizeStoredScope.run(normalized, scope.workspace);
				}
			}
		}
	} catch {}

	// Remove deprecated phase column from retrieval_traces (SQLite needs
	// the rename-table workaround because ALTER TABLE DROP COLUMN is not
	// supported until 3.35.0).
	try {
		const rtCols = db
			.prepare("PRAGMA table_info(retrieval_traces)")
			.all() as Array<{ name: string }>;
		if (rtCols.find(c => c.name === "phase")) {
			db.exec("DROP TABLE IF EXISTS retrieval_traces_v2");
			db.exec(`
				CREATE TABLE retrieval_traces_v2 (
					id TEXT PRIMARY KEY,
					workspace TEXT NOT NULL,
					session_id TEXT NOT NULL,
					objective TEXT NOT NULL,
					created_at TEXT NOT NULL,
					latency_ms REAL NOT NULL,
					budget INTEGER NOT NULL,
					tokens INTEGER NOT NULL,
					abstained INTEGER NOT NULL,
					reason TEXT,
					candidate_counts TEXT NOT NULL,
					selected TEXT NOT NULL
				)
			`);
			db.exec(
				"INSERT INTO retrieval_traces_v2 SELECT id, workspace, session_id, objective, created_at, latency_ms, budget, tokens, abstained, reason, candidate_counts, selected FROM retrieval_traces",
			);
			db.exec("DROP TABLE retrieval_traces");
			db.exec("ALTER TABLE retrieval_traces_v2 RENAME TO retrieval_traces");
			db.exec(
				"CREATE INDEX IF NOT EXISTS idx_retrieval_traces_workspace ON retrieval_traces(workspace, created_at DESC)",
			);
			// Clear all traces — their phase-weighted scores are no longer valid.
			db.exec("DELETE FROM retrieval_traces");
			db.exec("DELETE FROM memory_outcome_receipts");
		}
	} catch {}
}
