// ── @logician/memory — SQLite-backed Store ───────────────────────────────────
// Implements the observation→compression→memory pipeline.

import { Database } from "bun:sqlite";
import { mkdirSync } from "node:fs";
import { dirname, normalize, resolve } from "node:path";
import type {
  CompressedObservation,
  ContextBlock,
  ContextRetrievalQuery,
  CreateMemoryOptions,
  DecayConfig,
  DecayConfigInput,
  ExportData,
  ExtractionJob,
  ExtractionJobStatus,
  ExpandedMemoryEntry,
  FileContextEntry,
  ImportData,
  ImportResult,
  Memory,
  MemoryQuery,
  MemoryRelation,
  MemoryRelationType,
  MemoryStore,
  MemoryType,
  ObservationType,
  RawObservation,
  RecallOptions,
  RetentionScore,
  SearchResult,
  SemanticSearchResult,
  Session,
  WorkingMemoryTier,
} from "./types.js";

function generateId(): string {
  return crypto.randomUUID();
}

function now(): string {
  return new Date().toISOString();
}

function normalizeWorkspacePath(workspace: string): string {
  const value = workspace.trim();
  return normalize(resolve(value || process.cwd()));
}

const REDACTED = "[REDACTED]";
const MAX_RAW_STRING = 8_000;

function sanitizeString(value: string): string {
  return value
    .replace(/-----BEGIN[^-]*PRIVATE KEY-----[\s\S]*?-----END[^-]*PRIVATE KEY-----/gi, REDACTED)
    .replace(/\bBearer\s+[A-Za-z0-9._~+\/-]{12,}/gi, `Bearer ${REDACTED}`)
    .replace(/\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,})\b/g, REDACTED)
    .replace(/\b(api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd|secret)\s*[:=]\s*([^\s,;]+)/gi, `$1=${REDACTED}`)
    .slice(0, MAX_RAW_STRING);
}

function sanitizePayload(value: unknown, depth = 0): unknown {
  if (depth > 8) return "[TRUNCATED]";
  if (typeof value === "string") return sanitizeString(value);
  if (typeof value !== "object" || value === null) return value;
  if (Array.isArray(value)) return value.slice(0, 100).map((item) => sanitizePayload(item, depth + 1));
  const result: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value).slice(0, 100)) {
    result[key] = /(?:authorization|cookie|api[_-]?key|token|secret|password|passwd|private[_-]?key)/i.test(key)
      ? REDACTED
      : sanitizePayload(item, depth + 1);
  }
  return result;
}

function toFtsQuery(query: string): string {
  const terms = query.normalize("NFKC").match(/[\p{L}\p{N}_]+/gu) || [];
  return [...new Set(terms.slice(0, 12).map((term) => term.toLowerCase()))]
    .map((term) => `"${term.replace(/"/g, "\"\"")}"${term.length > 1 ? "*" : ""}`)
    .join(" AND ");
}

function toFtsAnyQuery(query: string): string {
  const stop = new Set(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "this", "to", "with"]);
  const terms = query.normalize("NFKC").match(/[\p{L}\p{N}_]+/gu) || [];
  return [...new Set(terms.map((term) => term.toLowerCase()).filter((term) => term.length > 1 && !stop.has(term)).slice(0, 12))]
    .map((term) => `"${term.replace(/"/g, "\"\"")}"*`)
    .join(" OR ");
}

// ── Synthetic Compression (zero-LLM) ─────────────────────────────────────────

function inferType(payload: unknown, hookType: string): ObservationType {
  // post_tool_failure is always an error, regardless of payload
  if (hookType === "post_tool_failure") return "error";
  if (hookType === "prompt_submit") return "conversation";
  if (hookType === "notification") return "notification";

  if (typeof payload === "object" && payload !== null) {
    const d = payload as Record<string, unknown>;
    const name = ((d as any).tool_name || (d as any).name || "") as string;
    const lower = name.toLowerCase();

    if (lower.includes("read") || lower.includes("cat")) return "file_read";
    if (lower.includes("write") || lower.includes("append") || lower.includes("overwrite")) return "file_write";
    if (lower.includes("edit")) return "file_edit";
    if (lower.includes("bash") || lower.includes("shell") || lower.includes("exec") || lower.includes("run")) return "command_run";
    if (lower.includes("search") || lower.includes("grep")) return "search";
    if (lower.includes("fetch") || lower.includes("curl") || lower.includes("http")) return "web_fetch";
  }
  return "other";
}

function buildSyntheticCompression(raw: RawObservation): CompressedObservation {
  const { id, sessionId, timestamp, hookType, raw: data } = raw;
  let type = inferType(data, hookType);

  let title = "Observation";
  let narrative = "";
  const facts: string[] = [];
  const concepts: string[] = [];
  const files: string[] = [];
  let importance = 5;

  if (typeof data === "object" && data !== null) {
    const d = data as Record<string, unknown>;
    const toolName = ((d as any).tool_name || (d as any).name || hookType) as string;

    // Extract file references
    const filePatterns = [
      /(?:file_path|path|file|filename|target_file|output_file)["']?\s*[:=]\s*["']?([^\s"'`,]+)/gi,
      /(?:read|write|edit|open)\s+["']?([^\s"'`,]+\.[\w.]+)/gi,
    ];
    for (const pattern of filePatterns) {
      let match;
      const str = JSON.stringify(d);
      while ((match = pattern.exec(str)) !== null) {
        if (match[1] && match[1].includes("/")) files.push(match[1].slice(0, 300));
      }
    }

    // Extract concepts from keywords
    const conceptKeywords = [
      "error", "bug", "fix", "crash", "panic", "timeout", "retry",
      "config", "setting", "env", "environment",
      "auth", "permission", "access", "token",
      "database", "schema", "migration", "query", "connection",
      "api", "endpoint", "route", "middleware", "login", "auth",
      "test", "unit", "integration", "mock", "stub",
      "build", "deploy", "pipeline", "ci", "cd",
      "refactor", "optimize", "performance", "memory", "cpu",
      "security", "vulnerability", "sanitize", "escape",
    ];
    const lowerStr = JSON.stringify(d).toLowerCase();
    for (const kw of conceptKeywords) {
      if (lowerStr.includes(kw)) concepts.push(kw);
    }

    // Build title and narrative
    const output = (d as any).tool_output || (d as any).output || (d as any).result || "";
    const error = (d as any).error || "";
    const input = (d as any).tool_input || (d as any).input || (d as any).arguments || {};
    const inputStr = typeof input === "string" ? input : JSON.stringify(input).slice(0, 500);

    // For failures, error field takes precedence
    const effectiveOutput = error || output;

    if (typeof effectiveOutput === "string" && effectiveOutput.length > 0) {
      const truncated = effectiveOutput.slice(0, 1000);
      title = truncate(truncated, 80) || toolName;
      narrative = `${toolName}: ${truncated.slice(0, 300)}`;
      facts.push(truncated.slice(0, 500));
    } else if (typeof inputStr === "string" && inputStr.length > 0) {
      title = `${toolName}: ${inputStr.slice(0, 80)}`;
      narrative = `${toolName}(input)`;
      facts.push(inputStr.slice(0, 500));
    } else {
      title = `${toolName}`;
      narrative = `${hookType} → ${toolName}`;
    }

    // Boost importance for errors
    const outputStr = typeof output === "string" ? output.toLowerCase() : "";
    const errorStr = typeof error === "string" ? error.toLowerCase() : "";
    const combinedErr = outputStr + " " + errorStr;
    if (type === "error") {
      importance = 8;
    } else if (type === "file_write" || type === "file_edit") {
      importance = 7;
    } else if (type === "file_read" || type === "search" || type === "web_fetch") {
      importance = 3;
    } else if (type === "command_run") {
      if (/\b(?:error|fail(?:ed|ure)?|panic|crash|exception|timeout|refused)\b/.test(combinedErr)) {
        importance = 8;
      } else {
        importance = /\b(?:pass(?:ed)?|success|built|compiled|deployed)\b/.test(outputStr) ? 6 : 4;
      }
    } else if (type === "other") {
      importance = 4;
    }

    // Add files to facts
    const uniqueFiles = [...new Set(files)];
    if (uniqueFiles.length > 0) {
      facts.push(`Files: ${uniqueFiles.slice(0, 5).join(", ")}`);
    }

    // Add concepts to narrative
    if (concepts.length > 3) {
      facts.push(`Concepts: ${concepts.slice(0, 5).join(", ")}`);
    }

    if (hookType === "prompt_submit") {
      const prompt = typeof d.prompt === "string"
        ? d.prompt.trim()
        : typeof d.userPrompt === "string"
          ? d.userPrompt.trim()
          : raw.userPrompt?.trim() || "";
      if (prompt) {
        title = `User request: ${truncate(prompt, 100)}`;
        narrative = prompt.slice(0, 2000);
        facts.length = 0;
        facts.push(prompt.slice(0, 1000));
        if (/^(?:hi|hello|hey|thanks|thank you|ok|okay)[!. ]*$/i.test(prompt)) {
          importance = 1;
        } else if (/\b(?:decide|decision|must|requirement|architecture|security|breaking|never|always)\b/i.test(prompt)) {
          type = "decision";
          importance = 7;
        } else if (prompt.length < 20) {
          importance = 2;
        } else {
          importance = 5;
        }
      }
    }
  } else if (typeof data === "string") {
    title = truncate(data, 80) || hookType;
    narrative = data.slice(0, 500);
    facts.push(data.slice(0, 500));
  }

  // Extract concepts from narrative
  const extractedConcepts = new Set(concepts);
  const conceptPatterns = [
    /#[\w]+/g,  // hashtags
    /\b([A-Z][a-z]+(?:[A-Z][a-z]+)*\w*)\b/g,  // camelCase/PascalCase words
  ];
  for (const pattern of conceptPatterns) {
    let match;
    const str = narrative + " " + facts.join(" ");
    while ((match = pattern.exec(str)) !== null) {
      const word = match[0].replace(/#/g, "");
      if (word.length >= 3 && !extractedConcepts.has(word)) {
        extractedConcepts.add(word);
      }
      if (extractedConcepts.size >= 10) break;
    }
  }

  const uniqueFiles = [...new Set(files)];

  return {
    id,
    sessionId,
    timestamp,
    type,
    title: title.slice(0, 200),
    narrative: narrative.slice(0, 2000),
    facts,
    concepts: [...extractedConcepts].slice(0, 10),
    files: uniqueFiles,
    importance: Math.max(1, Math.min(10, importance)),
    consolidated: false,
  };
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 3) + "...";
}

// ── DB Helpers ───────────────────────────────────────────────────────────────

function safeParseJsonArray(val: unknown): string[] {
  if (Array.isArray(val)) return val.map(String);
  if (typeof val === "string") {
    try { return JSON.parse(val); } catch { return []; }
  }
  return [];
}

function safeParseJson(val: string): unknown {
  try { return JSON.parse(val); } catch { return null; }
}

// ── Store Factory ────────────────────────────────────────────────────────────

export function createMemoryStore(dbPath: string): MemoryStore {
  const resolved = dbPath
    .replace(/^~(?=\/|$)/, process.env.HOME || "")
    .replace(/^~/, process.env.HOME || "");
  mkdirSync(dirname(resolved), { recursive: true });
  const db = new Database(resolved);
  let currentWorkspace = normalizeWorkspacePath(process.cwd());

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
      last_error TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_extraction_jobs_ready
      ON extraction_jobs(workspace, status, next_attempt_at, created_at);

    CREATE TABLE IF NOT EXISTS memory_embeddings (
      entity_id TEXT PRIMARY KEY,
      entity_kind TEXT NOT NULL CHECK(entity_kind IN ('observation', 'memory')),
      session_id TEXT,
      workspace TEXT NOT NULL DEFAULT '',
      dimensions INTEGER NOT NULL,
      vector TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memory_embeddings_workspace
      ON memory_embeddings(workspace, dimensions, entity_kind);
    CREATE INDEX IF NOT EXISTS idx_memory_embeddings_recency
      ON memory_embeddings(workspace, dimensions, updated_at DESC);
  `);

  // A process may exit after claiming but before acknowledging a job. Requeue
  // those leases on startup; writes are idempotent because observation IDs are
  // derived from the job ID by the hook worker.
  db.prepare("UPDATE extraction_jobs SET status = 'pending' WHERE status = 'running'").run();

  // ── Schema migrations ──────────────────────────────────────────────────
  // Add workspace columns to existing databases that were created before
  // the workspace scoping feature.
  for (const table of ["sessions", "observations", "memories"]) {
    try {
      const cols = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
      if (!cols.find(c => c.name === "workspace")) {
        db.exec(`ALTER TABLE ${table} ADD COLUMN workspace TEXT NOT NULL DEFAULT ''`);
      }
      try { db.exec(`CREATE INDEX IF NOT EXISTS idx_${table}_workspace ON ${table}(workspace)`); } catch {}
    } catch {}
  }
  try {
    const sessionCols = db.prepare("PRAGMA table_info(sessions)").all() as Array<{ name: string }>;
    if (!sessionCols.some((column) => column.name === "name")) {
      db.exec("ALTER TABLE sessions ADD COLUMN name TEXT");
    }
  } catch {}
  // Add consolidated column to observations
  try {
    const obsCols = db.prepare(`PRAGMA table_info(observations)`).all() as Array<{ name: string }>;
    if (!obsCols.find(c => c.name === "consolidated")) {
      db.exec(`ALTER TABLE observations ADD COLUMN consolidated INTEGER NOT NULL DEFAULT 0`);
    }
  } catch {}
  // Create consolidated index
  try { db.exec(`CREATE INDEX IF NOT EXISTS idx_observations_consolidated ON observations(consolidated)`); } catch {}

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
    const legacySessions = db.prepare(
      "SELECT id, cwd FROM sessions WHERE workspace = '' AND cwd != ''",
    ).all() as Array<{ id: string; cwd: string }>;
    const updateSessionWorkspace = db.prepare("UPDATE sessions SET workspace = ? WHERE id = ?");
    for (const session of legacySessions) {
      updateSessionWorkspace.run(normalizeWorkspacePath(session.cwd), session.id);
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
      const scopes = db.prepare(
        `SELECT DISTINCT workspace FROM ${table} WHERE workspace != ''`,
      ).all() as Array<{ workspace: string }>;
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

  // ── Sessions ───────────────────────────────────────────────────────────

  function createSession(id: string, data: Partial<Session>): Session {
    const ts = now();
    const sessionCwd = data.cwd ? normalizeWorkspacePath(data.cwd) : currentWorkspace;
    const sessionWorkspace = normalizeWorkspacePath(data.workspace || sessionCwd);
    db.prepare(`
      INSERT OR IGNORE INTO sessions (id, name, project, cwd, workspace, started_at, status, observation_count,
                                      model, tags, first_prompt, summary, commit_shas)
      VALUES (?, ?, COALESCE(?, ''), COALESCE(?, ''), COALESCE(?, ''), ?, 'active', 0, ?, ?, ?, ?, ?)
    `).run(
      id,
      data.name || null,
      data.project || "",
      sessionCwd,
      sessionWorkspace,
      ts,
      data.model || null,
      JSON.stringify(data.tags || []),
      data.firstPrompt || null,
      data.summary || null,
      JSON.stringify(data.commitShas || []),
    );
    return {
      id,
      name: data.name,
      project: data.project || "",
      cwd: sessionCwd,
      workspace: sessionWorkspace,
      startedAt: ts,
      status: "active",
      observationCount: 0,
      model: data.model,
      tags: data.tags || [],
      firstPrompt: data.firstPrompt,
      summary: data.summary,
      commitShas: data.commitShas || [],
    };
  }

  function getSession(id: string): Session | null {
    const row = db.prepare(`SELECT * FROM sessions WHERE id = ?`).get(id) as any;
    if (!row) return null;
    return {
      id: row.id,
      name: row.name || undefined,
      project: row.project || "",
      cwd: row.cwd || "",
      workspace: row.workspace || "",
      startedAt: row.started_at,
      endedAt: row.ended_at,
      status: row.status || "active",
      observationCount: row.observation_count || 0,
      model: row.model,
      tags: safeParseJsonArray(row.tags),
      firstPrompt: row.first_prompt,
      summary: row.summary,
      commitShas: safeParseJsonArray(row.commit_shas),
    };
  }

  function listSessions(query?: { status?: string; project?: string; workspace?: string }): Session[] {
    const conditions: string[] = [];
    const params: any[] = [];

    if (query?.status) {
      conditions.push("status = ?");
      params.push(query.status);
    }
    if (query?.project) {
      conditions.push("project = ?");
      params.push(query.project);
    }
    if (query?.workspace) {
      conditions.push("workspace = ?");
      params.push(normalizeWorkspacePath(query.workspace));
    } else {
      conditions.push("workspace = ?");
      params.push(currentWorkspace);
    }

    const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
    const rows = db.prepare(`SELECT * FROM sessions ${where} ORDER BY started_at DESC`).all(...params) as any[];
    return rows.map((r) => ({
      id: r.id,
      name: r.name || undefined,
      project: r.project || "",
      cwd: r.cwd || "",
      workspace: r.workspace || "",
      startedAt: r.started_at,
      endedAt: r.ended_at,
      status: r.status || "active",
      observationCount: r.observation_count || 0,
      model: r.model,
      tags: safeParseJsonArray(r.tags),
      firstPrompt: r.first_prompt,
      summary: r.summary,
      commitShas: safeParseJsonArray(r.commit_shas),
    }));
  }

  function updateSession(id: string, updates: Partial<Session>): Session | null {
    const sets: string[] = [];
    const params: any[] = [];

    if (updates.name !== undefined) { sets.push("name = ?"); params.push(updates.name || null); }
    if (updates.project !== undefined) { sets.push("project = ?"); params.push(updates.project); }
    if (updates.cwd !== undefined) { sets.push("cwd = ?"); params.push(updates.cwd); }
    if (updates.status !== undefined) { sets.push("status = ?"); params.push(updates.status); }
    if (updates.endedAt !== undefined) { sets.push("ended_at = ?"); params.push(updates.endedAt); }
    if (updates.observationCount !== undefined) { sets.push("observation_count = ?"); params.push(updates.observationCount); }
    if (updates.model !== undefined) { sets.push("model = ?"); params.push(updates.model); }
    if (updates.tags !== undefined) { sets.push("tags = ?"); params.push(JSON.stringify(updates.tags)); }
    if (updates.firstPrompt !== undefined) { sets.push("first_prompt = ?"); params.push(updates.firstPrompt); }
    if (updates.summary !== undefined) { sets.push("summary = ?"); params.push(updates.summary); }

    if (!sets.length) return getSession(id);

    params.push(id);
    db.prepare(`UPDATE sessions SET ${sets.join(", ")} WHERE id = ?`).run(...params);
    return getSession(id);
  }

  function clearSessions(keepSessionId?: string): { sessions: number; observations: number } {
    // Completely unscoped rows predate workspace support and can never be
    // shown or attributed safely. Treat them as legacy garbage when the user
    // explicitly asks to clean sessions.
    const rows = db.prepare(`
      SELECT id FROM sessions
      WHERE (workspace = ? OR (workspace = '' AND cwd = ''))
        AND (? IS NULL OR id != ?)
    `).all(currentWorkspace, keepSessionId || null, keepSessionId || null) as Array<{ id: string }>;
    if (!rows.length) return { sessions: 0, observations: 0 };
    const sessionIds = new Set(rows.map((session) => session.id));
    let observations = 0;
    const countObservations = db.prepare("SELECT COUNT(*) AS count FROM observations WHERE session_id = ?");
    const deleteObservations = db.prepare("DELETE FROM observations WHERE session_id = ?");
    const deleteSession = db.prepare("DELETE FROM sessions WHERE id = ?");
    for (const session of rows) {
      observations += (countObservations.get(session.id) as { count: number }).count;
      deleteObservations.run(session.id);
      deleteSession.run(session.id);
    }
    const memories = db.prepare("SELECT id, session_ids FROM memories WHERE json_valid(session_ids)")
      .all() as Array<{ id: string; session_ids: string }>;
    const updateSources = db.prepare("UPDATE memories SET session_ids = ? WHERE id = ?");
    for (const memory of memories) {
      const retained = safeParseJsonArray(memory.session_ids).filter((id) => !sessionIds.has(id));
      updateSources.run(JSON.stringify(retained), memory.id);
    }
    return { sessions: rows.length, observations };
  }

  function discardEmptySession(id: string): boolean {
    const session = db.prepare(`
      SELECT id FROM sessions
      WHERE id = ? AND observation_count = 0
        AND NOT EXISTS (SELECT 1 FROM observations WHERE session_id = sessions.id)
        AND NOT EXISTS (
          SELECT 1 FROM memories m, json_each(m.session_ids) je
          WHERE json_valid(m.session_ids) AND je.value = sessions.id
        )
    `).get(id) as { id: string } | undefined;
    if (!session) return false;
    return db.prepare("DELETE FROM sessions WHERE id = ?").run(id).changes > 0;
  }

  // ── Observations ───────────────────────────────────────────────────────

  function observe(raw: RawObservation, compressed?: CompressedObservation): CompressedObservation | null {
    const ts = now();
    const safeRaw = {
      ...raw,
      toolInput: sanitizePayload(raw.toolInput),
      toolOutput: sanitizePayload(raw.toolOutput),
      userPrompt: raw.userPrompt ? sanitizeString(raw.userPrompt) : undefined,
      raw: sanitizePayload(raw.raw),
    };
    const generated = compressed || buildSyntheticCompression(safeRaw);
    const comp: CompressedObservation = {
      ...generated,
      title: sanitizeString(generated.title).slice(0, 200),
      subtitle: generated.subtitle ? sanitizeString(generated.subtitle).slice(0, 300) : undefined,
      narrative: sanitizeString(generated.narrative).slice(0, 2000),
      facts: generated.facts.map(sanitizeString).slice(0, 20),
      concepts: generated.concepts.map(sanitizeString).slice(0, 20),
      files: generated.files.map(sanitizeString).slice(0, 20),
    };
    // Derive workspace from observation data or current workspace
    const obsWorkspace = normalizeWorkspacePath(raw.workspace || currentWorkspace);

    db.prepare(`
      INSERT INTO observations (id, session_id, timestamp, hook_type, type, title, subtitle,
                                narrative, facts, concepts, files, importance, workspace, raw_data)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      comp.id,
      raw.sessionId,
      raw.timestamp || ts,
      raw.hookType,
      comp.type,
      comp.title,
      comp.subtitle || null,
      comp.narrative,
      JSON.stringify(comp.facts),
      JSON.stringify(comp.concepts),
      JSON.stringify(comp.files),
      comp.importance,
      obsWorkspace,
      JSON.stringify(safeRaw.raw).slice(0, 32_000),
    );

    // Update session observation count
    db.prepare(`
      UPDATE sessions SET observation_count = observation_count + 1 WHERE id = ?
    `).run(raw.sessionId);

    // Apply sliding window cap (enforce max observations per session)
    slidingWindowCap(raw.sessionId, 200);

    return comp;
  }

  function getObservation(id: string, sessionId: string): CompressedObservation | null {
    const row = db.prepare(
      `SELECT * FROM observations WHERE id = ? AND session_id = ?`
    ).get(id, sessionId) as any;
    if (!row) return null;
    return {
      id: row.id,
      sessionId: row.session_id,
      timestamp: row.timestamp,
      type: row.type as ObservationType,
      title: row.title || "",
      subtitle: row.subtitle,
      facts: safeParseJsonArray(row.facts),
      narrative: row.narrative || "",
      concepts: safeParseJsonArray(row.concepts),
      files: safeParseJsonArray(row.files),
      importance: row.importance ?? 5,
      consolidated: row.consolidated === 1 || row.consolidated === true,
      workspace: row.workspace || "",
    };
  }

  function rowToObservation(row: any): CompressedObservation {
    return {
      id: row.id,
      sessionId: row.session_id,
      timestamp: row.timestamp,
      type: row.type as ObservationType,
      title: row.title || "",
      subtitle: row.subtitle,
      facts: safeParseJsonArray(row.facts),
      narrative: row.narrative || "",
      concepts: safeParseJsonArray(row.concepts),
      files: safeParseJsonArray(row.files),
      importance: row.importance ?? 5,
      consolidated: row.consolidated === 1 || row.consolidated === true,
      workspace: row.workspace || "",
    };
  }

  function listObservations(sessionId: string, limit: number = 50): CompressedObservation[] {
    const rows = db.prepare(
      `SELECT * FROM observations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?`
    ).all(sessionId, limit) as any[];
    return rows.map(rowToObservation);
  }

  function listRecentObservations(limit: number = 50, type?: ObservationType): CompressedObservation[] {
    const conditions: string[] = [];
    const params: Array<string | number> = [];
    conditions.push("workspace = ?");
    params.push(currentWorkspace);
    if (type) {
      conditions.push("type = ?");
      params.push(type);
    }
    const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
    params.push(Math.max(1, Math.min(limit, 1000)));
    const rows = db.prepare(
      `SELECT * FROM observations ${where} ORDER BY timestamp DESC, rowid DESC LIMIT ?`,
    ).all(...params) as any[];
    return rows.map(rowToObservation);
  }

  function searchObservations(query: string, limit: number = 20): SearchResult[] {
    const ftsQuery = toFtsQuery(query);
    if (!ftsQuery) return [];
    const rows = db.prepare(`
      SELECT o.*, bm25(observations_fts, 0, 8, 4, 2, 3, 3) AS lexical_rank
      FROM observations_fts
      JOIN observations o ON o.id = observations_fts.id
      WHERE observations_fts MATCH ? AND o.workspace = ?
      ORDER BY lexical_rank ASC, o.importance DESC, o.timestamp DESC
      LIMIT ?
    `).all(ftsQuery, currentWorkspace, Math.max(1, Math.min(limit, 1000))) as any[];

    return rows.map((r) => ({
      observation: {
        id: r.id,
        sessionId: r.session_id,
        timestamp: r.timestamp,
        type: r.type as ObservationType,
        title: r.title || "",
        subtitle: r.subtitle,
        facts: safeParseJsonArray(r.facts),
        narrative: r.narrative || "",
        concepts: safeParseJsonArray(r.concepts),
        files: safeParseJsonArray(r.files),
        importance: r.importance ?? 5,
        consolidated: r.consolidated === 1 || r.consolidated === true,
        workspace: r.workspace || "",
      },
      score: Number((Math.max(0, -Number(r.lexical_rank || 0)) + r.importance / 10).toFixed(4)),
      sessionId: r.session_id,
    }));
  }

  function expandEntries(ids: string[]): ExpandedMemoryEntry[] {
    const uniqueIds = [...new Set(ids.map((id) => id.trim()).filter(Boolean))].slice(0, 20);
    const entries = new Map<string, ExpandedMemoryEntry>();
    const observationStatement = db.prepare("SELECT * FROM observations WHERE id = ? AND workspace = ?");
    const memoryStatement = db.prepare("SELECT * FROM memories WHERE id = ? AND workspace = ?");
    for (const id of uniqueIds) {
      const observationRow = observationStatement.get(id, currentWorkspace) as any;
      if (observationRow) {
        const observation = rowToObservation(observationRow);
        entries.set(id, {
          id,
          kind: "observation",
          title: observation.title,
          content: [observation.narrative, ...observation.facts].filter(Boolean).join("\n"),
          type: observation.type,
          files: observation.files,
          concepts: observation.concepts,
          timestamp: observation.timestamp,
          sessionIds: [observation.sessionId],
        });
        continue;
      }
      const memoryRow = memoryStatement.get(id, currentWorkspace) as any;
      if (memoryRow) {
        const memory = rowToMemory(memoryRow);
        trackAccess(memory.id);
        entries.set(id, {
          id,
          kind: "memory",
          title: memory.title,
          content: memory.content,
          type: memory.type,
          files: memory.files,
          concepts: memory.concepts,
          timestamp: memory.updatedAt,
          sessionIds: memory.sessionIds,
        });
      }
    }
    return uniqueIds.flatMap((id) => entries.get(id) ? [entries.get(id)!] : []);
  }

  function clearObservations(): number {
    const { count } = db.prepare("SELECT COUNT(*) AS count FROM observations WHERE workspace = ?")
      .get(currentWorkspace) as { count: number };
    db.prepare("DELETE FROM observations WHERE workspace = ?").run(currentWorkspace);
    db.prepare(`
      UPDATE sessions
      SET observation_count = (
        SELECT COUNT(*) FROM observations WHERE observations.session_id = sessions.id
      )
      WHERE workspace = ?
    `).run(currentWorkspace);
    return count;
  }

  // ── Memories ───────────────────────────────────────────────────────────

  function create(content: string, options: CreateMemoryOptions = {}): Memory {
    const id = generateId();
    const ts = now();

    // Auto-extract concepts from content
    const concepts = options.concepts || extractConcepts(content);
    const files = options.files || extractFiles(content);

    // Auto-assign strength
    const strength = options.strength ?? assignStrength(content);

    // Derive workspace from options or current workspace
    const memoryWorkspace = normalizeWorkspacePath(options.workspace || currentWorkspace);

    db.prepare(`
      INSERT INTO memories (id, created_at, updated_at, type, title, content,
                            concepts, files, session_ids, strength, version,
                            parent_id, related_ids, source_observation_ids, is_latest, project, workspace,
                            access_count, last_accessed, working_tier)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, '[]', '[]', 1, ?, ?, 0, NULL, 'cold')
    `).run(
      id,
      ts,
      ts,
      options.type || "fact",
      content.slice(0, 200),
      content,
      JSON.stringify(concepts),
      JSON.stringify(files),
      JSON.stringify(options.sessionIds || []),
      strength,
      options.parentId || null,
      options.project || null,
      memoryWorkspace,
    );

    return {
      id,
      createdAt: ts,
      updatedAt: ts,
      type: options.type || "fact",
      title: content.slice(0, 200),
      content,
      concepts,
      files,
      sessionIds: options.sessionIds || [],
      strength,
      version: 1,
      parentId: options.parentId,
      relatedIds: [],
      sourceObservationIds: [],
      isLatest: true,
      project: options.project,
      workspace: memoryWorkspace,
    };
  }

  function get(id: string): Memory | null {
    const row = db.prepare(
      "SELECT * FROM memories WHERE id = ? AND is_latest = 1 AND workspace = ?",
    ).get(id, currentWorkspace) as any;
    if (!row) return null;
    return {
      id: row.id,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      type: row.type as MemoryType,
      title: row.title || "",
      content: row.content,
      concepts: safeParseJsonArray(row.concepts),
      files: safeParseJsonArray(row.files),
      sessionIds: safeParseJsonArray(row.session_ids),
      strength: row.strength ?? 5,
      version: row.version ?? 1,
      parentId: row.parent_id,
      supersedes: safeParseJsonArray(row.supersedes),
      relatedIds: safeParseJsonArray(row.related_ids),
      sourceObservationIds: safeParseJsonArray(row.source_observation_ids),
      isLatest: true,
      project: row.project,
      workspace: row.workspace || "",
    };
  }

  function getAny(id: string): Memory | null {
    const row = db.prepare("SELECT * FROM memories WHERE id = ? AND workspace = ?").get(id, currentWorkspace) as any;
    if (!row) return null;
    return {
      id: row.id,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      type: row.type as MemoryType,
      title: row.title || "",
      content: row.content,
      concepts: safeParseJsonArray(row.concepts),
      files: safeParseJsonArray(row.files),
      sessionIds: safeParseJsonArray(row.session_ids),
      strength: row.strength ?? 5,
      version: row.version ?? 1,
      parentId: row.parent_id,
      supersedes: safeParseJsonArray(row.supersedes),
      relatedIds: safeParseJsonArray(row.related_ids),
      sourceObservationIds: safeParseJsonArray(row.source_observation_ids),
      isLatest: row.is_latest === 1,
      project: row.project,
      workspace: row.workspace || "",
    };
  }

  function rowToMemory(row: any): Memory {
    return {
      id: row.id,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      type: row.type as MemoryType,
      title: row.title || "",
      content: row.content,
      concepts: safeParseJsonArray(row.concepts),
      files: safeParseJsonArray(row.files),
      sessionIds: safeParseJsonArray(row.session_ids),
      strength: row.strength ?? 5,
      version: row.version ?? 1,
      parentId: row.parent_id,
      supersedes: safeParseJsonArray(row.supersedes),
      relatedIds: safeParseJsonArray(row.related_ids),
      sourceObservationIds: safeParseJsonArray(row.source_observation_ids),
      isLatest: row.is_latest === 1 || row.is_latest === true,
      project: row.project,
      workspace: row.workspace || "",
    };
  }

  // ── Dedup ──────────────────────────────────────────────────────────────

  const dedupWindowMs = 5 * 60 * 1000; // 5 minutes

  function computeDedupHash(sessionId: string, toolName: string, toolInput: unknown): string {
    const inputStr = typeof toolInput === "string" ? toolInput : JSON.stringify(toolInput ?? "").slice(0, 500);
    const raw = `${sessionId}:${toolName}:${inputStr}`;
    // Simple hash: use node:crypto if available, otherwise fallback
    try {
      const { createHash } = require("node:crypto") as typeof import("node:crypto");
      return createHash("sha256").update(raw).digest("hex").slice(0, 16);
    } catch {
      // Fallback: simple string hash
      let hash = 0;
      for (let i = 0; i < raw.length; i++) {
        const char = raw.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
      }
      return Math.abs(hash).toString(36);
    }
  }

  function dedupCheck(sessionId: string, toolName: string, toolInput: unknown): boolean {
    const hash = computeDedupHash(sessionId, toolName, toolInput);
    const row = db.prepare("SELECT created_at FROM dedup WHERE hash = ?").get(hash) as { created_at: string } | undefined;
    if (!row) return false;
    const age = Date.now() - new Date(row.created_at).getTime();
    return age < dedupWindowMs;
  }

  function dedupRecord(sessionId: string, toolName: string, toolInput: unknown): void {
    const hash = computeDedupHash(sessionId, toolName, toolInput);
    db.prepare(`
      INSERT INTO dedup (hash, created_at) VALUES (?, ?)
      ON CONFLICT(hash) DO UPDATE SET created_at = excluded.created_at
    `).run(hash, now());
    // Clean up old entries
    db.prepare("DELETE FROM dedup WHERE created_at < ?").run(new Date(Date.now() - dedupWindowMs * 2).toISOString());
  }

  // ── Sliding Window ─────────────────────────────────────────────────────

  function slidingWindowCap(sessionId: string, cap: number = 200): number {
    const excess = db.prepare(
      "SELECT COUNT(*) as cnt FROM observations WHERE session_id = ? AND id NOT IN (SELECT id FROM observations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?)"
    ).get(sessionId, sessionId, cap) as { cnt: number };
    if (!excess || excess.cnt <= 0) return 0;

    db.prepare(`
      DELETE FROM observations WHERE session_id = ? AND id NOT IN (
        SELECT id FROM observations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?
      )
    `).run(sessionId, sessionId, cap);

    return excess.cnt;
  }

  // ── Access Tracker ─────────────────────────────────────────────────────

  function trackAccess(entityId: string): void {
    db.prepare(`
      UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?
    `).run(now(), entityId);
  }

  function getAccessStats(entityId: string): { lastAccessed: string; accessCount: number } | null {
    const row = db.prepare(
      "SELECT last_accessed, access_count FROM memories WHERE id = ?"
    ).get(entityId) as { last_accessed: string; access_count: number } | undefined;
    if (!row) return null;
    return {
      lastAccessed: row.last_accessed || "",
      accessCount: row.access_count || 0,
    };
  }

  // ── Working Memory Tiers ───────────────────────────────────────────────

  function getWorkingMemoryTier(entityId: string): WorkingMemoryTier {
    const row = db.prepare("SELECT working_tier FROM memories WHERE id = ?").get(entityId) as { working_tier: string } | undefined;
    return (row?.working_tier as WorkingMemoryTier) || "cold";
  }

  function setWorkingMemoryTier(entityId: string, tier: WorkingMemoryTier): void {
    db.prepare("UPDATE memories SET working_tier = ? WHERE id = ?").run(tier, entityId);
  }

  function autoTierMemories(): Record<string, WorkingMemoryTier> {
    const tiered: Record<string, WorkingMemoryTier> = {};
    const nowMs = Date.now();
    const oneHour = 60 * 60 * 1000;
    const oneDay = 24 * oneHour;

    const rows = db.prepare(
      "SELECT id, last_accessed FROM memories WHERE is_latest = 1 AND last_accessed IS NOT NULL AND workspace = ?"
    ).all(currentWorkspace) as { id: string; last_accessed: string }[];

    for (const row of rows) {
      const accessMs = new Date(row.last_accessed).getTime();
      const age = nowMs - accessMs;
      let tier: WorkingMemoryTier;
      if (age < oneHour) tier = "hot";
      else if (age < oneDay) tier = "warm";
      else tier = "cold";

      db.prepare("UPDATE memories SET working_tier = ? WHERE id = ?").run(tier, row.id);
      tiered[row.id] = tier;
    }

    // Mark memories with no access as archived
    db.prepare(
      "UPDATE memories SET working_tier = 'archived' WHERE is_latest = 1 AND last_accessed IS NULL AND workspace = ?",
    ).run(currentWorkspace);

    return tiered;
  }

  // ── Auto-Forget ────────────────────────────────────────────────────────

  interface AutoForgetResult {
    deleted: number;
    details: string[];
  }

  function autoForget(ttlMs: number = 30 * 24 * 60 * 60 * 1000, minImportance: number = 3, maxDeletes: number = 100): AutoForgetResult {
    const cutoff = new Date(Date.now() - ttlMs).toISOString();
    const result: AutoForgetResult = { deleted: 0, details: [] };

    // Find old, low-importance observations
    const oldObs = db.prepare(
      "SELECT id, session_id, importance, timestamp FROM observations WHERE workspace = ? AND timestamp < ? AND importance < ? LIMIT ?"
    ).all(currentWorkspace, cutoff, minImportance, maxDeletes) as { id: string; session_id: string; importance: number; timestamp: string }[];

    for (const obs of oldObs) {
      db.prepare("DELETE FROM observations WHERE id = ?").run(obs.id);
      result.deleted++;
      result.details.push(`Deleted obs ${obs.id.slice(0, 8)} from session ${obs.session_id.slice(0, 8)} (${obs.importance}/10)`);
    }

    return result;
  }

  function list(query: MemoryQuery = {}): Memory[] {
    const workspace = normalizeWorkspacePath(query.workspace || currentWorkspace);
    const conditions: string[] = [];
    const params: any[] = [];
    const ftsQuery = query.search ? toFtsQuery(query.search) : "";
    const from = ftsQuery
      ? "memories_fts JOIN memories m ON m.id = memories_fts.id"
      : "memories m";
    if (ftsQuery) {
      conditions.push("memories_fts MATCH ?");
      params.push(ftsQuery);
    }
    conditions.push("m.workspace = ?", "m.is_latest = 1");
    params.push(workspace);

    if (query.type) {
      conditions.push("m.type = ?");
      params.push(query.type);
    }

    if (query.project) {
      conditions.push("m.project = ?");
      params.push(query.project);
    }

    if (query.minStrength !== undefined) {
      conditions.push("m.strength >= ?");
      params.push(query.minStrength);
    }

    const baseWhere = conditions.join(" AND ");

    // Subquery for concept AND filtering: all concepts must match
    const conceptSub = query.concepts?.length
      ? `(SELECT COUNT(DISTINCT je.value) FROM json_each(m.concepts) je WHERE je.value IN (${Array(query.concepts!.length).fill("?").join(", ")})) >= ${query.concepts!.length}`
      : "1=1";

    // Subquery for session matching
    const sessionSub = query.sessionId
      ? `(SELECT COUNT(*) FROM json_each(m.session_ids) je WHERE je.value = ?) >= 1`
      : "1=1";

    // Add params for subqueries
    if (query.concepts?.length) {
      for (const c of query.concepts!) params.push(c);
    }
    if (query.sessionId) {
      params.push(query.sessionId);
    }

    const limit = query.limit ?? 10;

    const sql = `
      SELECT m.*${ftsQuery ? ", bm25(memories_fts, 0, 8, 4, 2, 2) AS lexical_rank" : ""}
      FROM ${from}
      WHERE ${baseWhere}
        AND ${conceptSub}
        AND ${sessionSub}
      ORDER BY ${ftsQuery ? "lexical_rank ASC," : ""} m.strength DESC, m.updated_at DESC
      LIMIT ?
    `;
    params.push(limit);

    return db.prepare(sql).all(...params).map(rowToMemory);
  }

  function deleteEntry(id: string): boolean {
    // Only delete if still latest (prevent double-delete)
    const result = db.prepare("UPDATE memories SET is_latest = 0 WHERE id = ? AND is_latest = 1").run(id);
    return result.changes > 0;
  }

  function clearMemories(): number {
    const ids = db.prepare("SELECT id FROM memories WHERE workspace = ?").all(currentWorkspace) as Array<{ id: string }>;
    if (!ids.length) return 0;
    const removeRelations = db.prepare("DELETE FROM relations WHERE source_id = ? OR target_id = ?");
    for (const { id } of ids) removeRelations.run(id, id);
    db.prepare("DELETE FROM memories WHERE workspace = ?").run(currentWorkspace);
    return ids.length;
  }

  function update(
    id: string,
    updates: Partial<Pick<Memory, "content" | "concepts" | "strength" | "title">>,
  ): Memory | null {
    const sets: string[] = [];
    const params: any[] = [];

    if (updates.content !== undefined) { sets.push("content = ?"); params.push(updates.content); }
    if (updates.title !== undefined) { sets.push("title = ?"); params.push(updates.title); }
    if (updates.concepts !== undefined) { sets.push("concepts = ?"); params.push(JSON.stringify(updates.concepts)); }
    if (updates.strength !== undefined) { sets.push("strength = ?"); params.push(updates.strength); }

    if (!sets.length) return get(id);

    sets.push("updated_at = ?");
    params.push(now());
    params.push(id);

    db.prepare(`UPDATE memories SET ${sets.join(", ")} WHERE id = ?`).run(...params);
    return get(id);
  }

  function recall(query: MemoryQuery, options: RecallOptions = {}): string {
    const memories = list(query);
    if (!memories.length) return "";

    const format = options.format || "text";
    const template = options.template || "{{title}}: {{content}}";

    if (format === "markdown") {
      return memories
        .map((m) => `## ${m.title} [${m.strength}/10]\n\n${m.content}\n\n${m.concepts.length ? " Concepts: " + m.concepts.join(", ") : ""}`)
        .join("\n\n---\n\n");
    }

    if (format === "system-prompt") {
      return memories
        .map((m) => `## ${m.type} [${m.strength}/10]\n\n${m.content}`)
        .join("\n\n");
    }

    // text
    return memories
      .map((m) => template
        .replace("{{content}}", m.content)
        .replace("{{title}}", m.title)
        .replace("{{strength}}", String(m.strength)))
      .join("\n\n");
  }

  function upsertEmbedding(
    id: string,
    kind: "observation" | "memory",
    vector: number[],
    sessionId?: string,
  ): void {
    if (!vector.length || vector.some((value) => !Number.isFinite(value))) return;
    db.prepare(`INSERT INTO memory_embeddings
      (entity_id, entity_kind, session_id, workspace, dimensions, vector, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(entity_id) DO UPDATE SET
        entity_kind = excluded.entity_kind,
        session_id = excluded.session_id,
        workspace = excluded.workspace,
        dimensions = excluded.dimensions,
        vector = excluded.vector,
        updated_at = excluded.updated_at`)
      .run(id, kind, sessionId || null, currentWorkspace, vector.length, JSON.stringify(vector), now());
  }

  // Brute-force cosine similarity has no index to lean on, so it's bounded to
  // the most recently embedded rows rather than the whole workspace history.
  // This runs synchronously on every turn's context injection (once the
  // embedder is warm), so an unbounded scan would make per-turn latency grow
  // linearly with total memory size. Recency is a reasonable proxy here: the
  // context-ranking pass in getContext() already privileges recent and
  // lexically/file-matched candidates over exhaustive semantic recall, so
  // capping the semantic candidate pool trades a small amount of recall on
  // very old entries for bounded per-turn cost.
  const SEMANTIC_SEARCH_SCAN_CAP = 4_000;

  function searchEmbeddings(vector: number[], limit: number = 40): SemanticSearchResult[] {
    if (!vector.length || vector.some((value) => !Number.isFinite(value))) return [];
    const rows = db.prepare(`SELECT entity_id, entity_kind, session_id, vector
      FROM memory_embeddings WHERE workspace = ? AND dimensions = ?
      ORDER BY updated_at DESC LIMIT ?`)
      .all(currentWorkspace, vector.length, SEMANTIC_SEARCH_SCAN_CAP) as Array<{
        entity_id: string; entity_kind: "observation" | "memory";
        session_id: string | null; vector: string;
      }>;
    let queryNorm = 0;
    for (const value of vector) queryNorm += value * value;
    queryNorm = Math.sqrt(queryNorm);
    if (!queryNorm) return [];
    const results: SemanticSearchResult[] = [];
    for (const row of rows) {
      const candidate = safeParseJson(row.vector);
      if (!Array.isArray(candidate) || candidate.length !== vector.length) continue;
      let dot = 0;
      let norm = 0;
      for (let index = 0; index < vector.length; index++) {
        const value = Number(candidate[index]);
        if (!Number.isFinite(value)) { norm = 0; break; }
        dot += vector[index] * value;
        norm += value * value;
      }
      const score = norm ? dot / (queryNorm * Math.sqrt(norm)) : 0;
      if (score <= 0) continue;
      results.push({
        id: row.entity_id,
        kind: row.entity_kind,
        sessionId: row.session_id || undefined,
        score,
      });
    }
    return results.sort((a, b) => b.score - a.score).slice(0, Math.max(1, Math.min(limit, 200)));
  }

  function hasEmbedding(id: string): boolean {
    return Boolean(db.prepare("SELECT 1 AS found FROM memory_embeddings WHERE entity_id = ? AND workspace = ?")
      .get(id, currentWorkspace));
  }

  // ── Consolidation ──────────────────────────────────────────────────────

  function consolidate(sessionId: string): Memory[] {
    const pendingRows = db.prepare(
      `SELECT * FROM observations
       WHERE session_id = ? AND workspace = ? AND consolidated = 0
         AND importance >= 5
       ORDER BY timestamp ASC LIMIT 100`
    ).all(sessionId, currentWorkspace) as any[];

    // Semantic episodes are complete grounded units. When present, consolidate
    // those rather than mechanically merging their underlying tool telemetry.
    const semanticRows = pendingRows.filter((row) => row.hook_type === "stop");
    const rows = semanticRows.length ? semanticRows : pendingRows;

    if (rows.length < 1 || (!semanticRows.length && rows.length < 2)) return [];

    const observations = rows.map((r) => ({
      id: r.id,
      sessionId: r.session_id,
      timestamp: r.timestamp,
      type: r.type as ObservationType,
      title: r.title || "",
      subtitle: r.subtitle,
      facts: safeParseJsonArray(r.facts),
      narrative: r.narrative || "",
      concepts: safeParseJsonArray(r.concepts),
      files: safeParseJsonArray(r.files),
      importance: r.importance ?? 5,
      consolidated: r.consolidated === 1,
      workspace: r.workspace || currentWorkspace,
    }));

    // Group by the most concrete topic available. A file or concept is much
    // more useful than broad buckets such as "command_run".
    const groups: Record<string, typeof observations> = {};
    for (const obs of observations) {
      const key = obs.files[0]
        ? `file:${obs.files[0]}`
        : obs.concepts[0]
          ? `concept:${obs.concepts[0].toLowerCase()}`
          : `type:${obs.type}`;
      if (!groups[key]) groups[key] = [];
      groups[key].push(obs);
    }

    const memories: Memory[] = [];
    const usedObservationIds: string[] = [];

    // A single transaction for the whole consolidation pass: it makes each
    // group's read-existing/supersede/insert sequence atomic with respect to
    // other writers on this connection (the extraction worker also calls
    // consolidate() after every job), and turns what was one commit per
    // group/observation into a single commit for the entire pass.
    const applyConsolidation = db.transaction(() => {
      for (const [topic, group] of Object.entries(groups)) {
        if (group.length < 2 && !semanticRows.length) continue;
        const dominantType = group
          .map((item) => item.type)
          .sort((a, b) => group.filter((item) => item.type === b).length - group.filter((item) => item.type === a).length)[0];
        const allFacts = [...new Set(group.flatMap((o) => o.facts.length ? o.facts : [o.narrative]).filter(Boolean))].slice(0, 8);
        const allConcepts = [...new Set(group.flatMap(o => o.concepts))].slice(0, 10);
        const allFiles = [...new Set(group.flatMap(o => o.files))].slice(0, 10);
        const avgStrength = Math.round(group.reduce((s, o) => s + o.importance, 0) / group.length);

        const typeNames: Record<string, MemoryType> = {
          file_read: "fact",
          file_write: "pattern",
          file_edit: "pattern",
          command_run: "workflow",
          search: "fact",
          web_fetch: "fact",
          conversation: "pattern",
          error: "bug",
          decision: "pattern",
          discovery: "fact",
          implementation: "architecture",
          bugfix: "bug",
          notification: "fact",
          other: "fact",
        };

        const label = topic.replace(/^(?:file|concept|type):/, "");
        const title = `${label} — ${dominantType.replace(/_/g, " ")}`.slice(0, 200);
        const content = allFacts.join("\n");
        const sourceIds = group.map((o) => o.id);
        const strength = Math.min(10, Math.max(1, avgStrength + 1));

        // Writes the group as either a fresh memory or a superseding version
        // of `existingRow`. idx_memories_latest_title enforces at most one
        // is_latest=1 row per (workspace, title); the caller retries this
        // once against a fresh read if that constraint fires, which only
        // happens when a separate process consolidates the same title
        // between the SELECT and this INSERT (writes on this connection are
        // already serialized by the enclosing transaction).
        const writeMemory = (existingRow: any) => {
          const ts = now();
          if (existingRow) {
            const existing = rowToMemory(existingRow);
            const id = generateId();
            const mergedContent = [...new Set([...existing.content.split("\n"), ...allFacts])].filter(Boolean).slice(-12).join("\n");
            const mergedConcepts = [...new Set([...existing.concepts, ...allConcepts])].slice(0, 15);
            const mergedFiles = [...new Set([...existing.files, ...allFiles])].slice(0, 15);
            const mergedSessions = [...new Set([...existing.sessionIds, sessionId])];
            const mergedSources = [...new Set([...(existing.sourceObservationIds || []), ...sourceIds])].slice(-100);
            db.prepare("UPDATE memories SET is_latest = 0 WHERE id = ?").run(existing.id);
            db.prepare(`INSERT INTO memories
                (id, created_at, updated_at, type, title, content, concepts, files, session_ids,
                 strength, version, parent_id, related_ids, source_observation_ids, is_latest,
                 project, workspace, supersedes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)`)
                .run(id, existing.createdAt, ts, existing.type, title, mergedContent,
                  JSON.stringify(mergedConcepts), JSON.stringify(mergedFiles), JSON.stringify(mergedSessions),
                  Math.min(10, Math.max(existing.strength, strength)), existing.version + 1, existing.id,
                  JSON.stringify(existing.relatedIds || []), JSON.stringify(mergedSources), existing.project || null,
                  currentWorkspace, JSON.stringify([existing.id]));
            db.prepare(`INSERT INTO relations (id, type, source_id, target_id, confidence, created_at)
              VALUES (?, 'supersedes', ?, ?, 1, ?)`)
              .run(generateId(), id, existing.id, ts);
            return id;
          }
          const id = generateId();
          db.prepare(`INSERT INTO memories
            (id, created_at, updated_at, type, title, content, concepts, files, session_ids,
             strength, version, parent_id, related_ids, source_observation_ids, is_latest, project, workspace)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, '[]', ?, 1, NULL, ?)`)
            .run(id, ts, ts, typeNames[dominantType] || "fact", title, content,
              JSON.stringify(allConcepts), JSON.stringify(allFiles), JSON.stringify([sessionId]),
              strength, JSON.stringify(sourceIds), currentWorkspace);
          return id;
        };

        const existingRow = db.prepare(
          "SELECT * FROM memories WHERE workspace = ? AND is_latest = 1 AND title = ? LIMIT 1",
        ).get(currentWorkspace, title) as any;

        let newId: string;
        try {
          newId = writeMemory(existingRow);
        } catch (error) {
          if (!(error instanceof Error) || !/UNIQUE constraint failed/.test(error.message)) throw error;
          const retryRow = db.prepare(
            "SELECT * FROM memories WHERE workspace = ? AND is_latest = 1 AND title = ? LIMIT 1",
          ).get(currentWorkspace, title) as any;
          newId = writeMemory(retryRow);
        }
        memories.push(rowToMemory(db.prepare("SELECT * FROM memories WHERE id = ?").get(newId)));
        usedObservationIds.push(...sourceIds);
      }

      if (usedObservationIds.length) {
        const mark = db.prepare("UPDATE observations SET consolidated = 1 WHERE id = ?");
        usedObservationIds.forEach((id) => mark.run(id));
      }
    });
    applyConsolidation();

    return memories;
  }

  // ── Context Injection ──────────────────────────────────────────────────

  function getContext(
    sessionId: string,
    budget: number = 4000,
    query: string | ContextRetrievalQuery = "",
  ): string {
    const retrieval = typeof query === "string" ? { objective: query } : query;
    const objective = retrieval.objective?.trim() || "";
    const phase = retrieval.phase || "orient";
    const changedFiles = retrieval.changedFiles || [];
    const queryText = [objective, ...changedFiles, ...(retrieval.recentEvidence || [])].join(" ");
    const queryTokens = contextTokens(queryText);
    const fileTokens = new Set(changedFiles.flatMap((file) => [...contextTokens(file)]));
    const nowMs = Date.now();
    const estimateTokens = (text: string) => Math.ceil(text.length / 3);
    type Candidate = ContextBlock & {
      id: string;
      score: number;
      sourceKey: string;
      memoryId?: string;
    };
    const candidates: Candidate[] = [];
    const episodicFallbackCandidates: Candidate[] = [];
    const briefDescription = (value: string, maxLength: number = 220): string => {
      const normalized = value.replace(/\s+/g, " ").trim();
      if (normalized.length <= maxLength) return normalized;
      const slice = normalized.slice(0, maxLength - 1);
      const boundary = slice.lastIndexOf(" ");
      return `${slice.slice(0, boundary > maxLength * 0.65 ? boundary : undefined).trimEnd()}…`;
    };

    const overlapScore = (text: string): number => {
      if (!queryTokens.size) return 0;
      const candidateTokens = contextTokens(text);
      let overlap = 0;
      for (const token of queryTokens) if (candidateTokens.has(token)) overlap++;
      return overlap / Math.sqrt(Math.max(1, queryTokens.size * candidateTokens.size));
    };
    const fileScore = (files: string[] | undefined, text: string): number => {
      if (!fileTokens.size) return 0;
      const candidateTokens = contextTokens([...(files || []), text].join(" "));
      let matches = 0;
      for (const token of fileTokens) if (candidateTokens.has(token)) matches++;
      return matches / fileTokens.size;
    };
    const recencyScore = (timestamp: string | undefined): number => {
      const ageDays = Math.max(0, nowMs - Date.parse(timestamp || "")) / 86_400_000;
      return Number.isFinite(ageDays) ? 1 / (1 + ageDays / 14) : 0;
    };
    const phaseScore = (type: string): number => {
      if (phase === "investigate" && /error|file_read|search|fact/.test(type)) return 1;
      if (phase === "implement" && /file_write|file_edit|decision|pattern/.test(type)) return 1;
      if (phase === "verify" && /command_run|error|test/.test(type)) return 1;
      if (phase === "blocked" && /error|decision/.test(type)) return 1;
      return 0;
    };

    const session = getSession(sessionId);
    if (session?.summary) {
      const content = `# Session Summary\n\n${session.summary}`;
      candidates.push({
        id: `summary:${sessionId}`,
        type: "summary",
        content,
        tokens: estimateTokens(content),
        recency: nowMs,
        score: 8 + overlapScore(session.summary) * 12,
        sourceKey: sessionId,
      });
    }

    // Generate candidates in SQLite so relevant older knowledge is not hidden
    // behind a fixed recent-item window. RRF makes lexical rank comparable to
    // task, file, recency, and salience signals without score calibration.
    const ftsQuery = toFtsAnyQuery(queryText);
    const lexicalObservationRank = new Map<string, number>();
    const lexicalMemoryRank = new Map<string, number>();
    const semanticObservationRank = new Map<string, number>();
    const semanticMemoryRank = new Map<string, number>();
    const lexicalObservations = ftsQuery
      ? db.prepare(`SELECT o.*, bm25(observations_fts, 0, 8, 4, 2, 3, 4) AS rank
          FROM observations_fts JOIN observations o ON o.id = observations_fts.id
          WHERE observations_fts MATCH ? AND o.workspace = ?
          ORDER BY rank ASC LIMIT 80`).all(ftsQuery, currentWorkspace) as any[]
      : [];
    lexicalObservations.forEach((row, index) => lexicalObservationRank.set(row.id, index + 1));
    const lexicalMemories = ftsQuery
      ? db.prepare(`SELECT m.*, bm25(memories_fts, 0, 8, 4, 3, 4) AS rank
          FROM memories_fts JOIN memories m ON m.id = memories_fts.id
          WHERE memories_fts MATCH ? AND m.workspace = ? AND m.is_latest = 1
          ORDER BY rank ASC LIMIT 80`).all(ftsQuery, currentWorkspace) as any[]
      : [];
    lexicalMemories.forEach((row, index) => lexicalMemoryRank.set(row.id, index + 1));
    const semanticResults = retrieval.semanticVector?.length
      ? searchEmbeddings(retrieval.semanticVector, 80)
      : [];
    semanticResults.forEach((result, index) => {
      const target = result.kind === "observation" ? semanticObservationRank : semanticMemoryRank;
      target.set(result.id, index + 1);
    });
    const rrfBoost = (rank: number | undefined, weight: number): number =>
      rank ? weight * (60 / (60 + rank)) : 0;

    const recentObservations = listRecentObservations(60);
    const observationPool = new Map<string, CompressedObservation>();
    recentObservations.forEach((obs) => observationPool.set(obs.id, obs));
    lexicalObservations.forEach((row) => observationPool.set(row.id, rowToObservation(row)));
    for (const result of semanticResults) {
      if (result.kind !== "observation" || observationPool.has(result.id)) continue;
      const row = db.prepare("SELECT * FROM observations WHERE id = ? AND workspace = ?")
        .get(result.id, currentWorkspace) as any;
      if (row) observationPool.set(result.id, rowToObservation(row));
    }
    const latestEpisodeBySession = new Map<string, number>();
    for (const obs of observationPool.values()) {
      if (!obs.id.startsWith("episode:")) continue;
      latestEpisodeBySession.set(
        obs.sessionId,
        Math.max(latestEpisodeBySession.get(obs.sessionId) || 0, Date.parse(obs.timestamp) || 0),
      );
    }
    for (const obs of observationPool.values()) {
      // The active transcript already contains current-session events. Adding
      // them again wastes context and can make stale tool output look current.
      if (obs.sessionId === sessionId) continue;
      const isEpisode = obs.id.startsWith("episode:");
      const coveredByEpisode = !isEpisode &&
        (latestEpisodeBySession.get(obs.sessionId) || 0) >= (Date.parse(obs.timestamp) || 0);
      // Completed semantic episodes supersede the low-level telemetry that
      // produced them. Newer, not-yet-synthesized events remain available.
      if (coveredByEpisode) continue;
      const body = `${obs.title} ${obs.narrative} ${(obs.facts || []).join(" ")} ${(obs.concepts || []).join(" ")}`;
      const relevance = overlapScore(body);
      const files = fileScore(obs.files, body);
      const score =
        relevance * 18 +
        files * 10 +
        phaseScore(obs.type) * 3 +
        (obs.importance / 10) * 4 +
        recencyScore(obs.timestamp) * 2 +
        (isEpisode ? 5 : 0) +
        rrfBoost(lexicalObservationRank.get(obs.id), 12) +
        rrfBoost(semanticObservationRank.get(obs.id), 10);
      // Episodic evidence is a fallback, not a peer of durable memory. Without
      // a task/file/semantic match, raw observations stay available through
      // memory_get and explicit observation search instead of entering every
      // prompt.
      if (!queryTokens.size || (relevance === 0 && files === 0 && !semanticObservationRank.has(obs.id))) continue;
      const label = isEpisode ? "Prior episode" : "Prior observation";
      const fileLabel = obs.files.length ? ` · ${obs.files.slice(0, 3).join(", ")}` : "";
      const description = briefDescription([obs.narrative, ...(obs.facts || [])].filter(Boolean).join(" "));
      const content = `- [${obs.id}] ${label} · ${obs.type} · ${obs.title}${description ? ` — ${description}` : ""}${fileLabel}`;
      episodicFallbackCandidates.push({
        id: `observation:${obs.id}`,
        type: "observation",
        content,
        tokens: estimateTokens(content),
        recency: Date.parse(obs.timestamp),
        score,
        sourceKey: obs.sessionId,
      });
    }

    const memoryPool = new Map<string, Memory>();
    let memoryCandidateCount = 0;
    list({ limit: 50, minStrength: 4 }).forEach((memory) => memoryPool.set(memory.id, memory));
    lexicalMemories.forEach((row) => memoryPool.set(row.id, rowToMemory(row)));
    for (const result of semanticResults) {
      if (result.kind !== "memory" || memoryPool.has(result.id)) continue;
      const row = db.prepare("SELECT * FROM memories WHERE id = ? AND workspace = ? AND is_latest = 1")
        .get(result.id, currentWorkspace) as any;
      if (row) memoryPool.set(result.id, rowToMemory(row));
    }
    for (const mem of memoryPool.values()) {
      const body = `${mem.title} ${mem.content} ${(mem.concepts || []).join(" ")} ${(mem.files || []).join(" ")}`;
      const relevance = overlapScore(body);
      const files = fileScore(mem.files, body);
      if (queryTokens.size && relevance === 0 && files === 0 && !semanticMemoryRank.has(mem.id)) continue;
      const sources = mem.sourceObservationIds?.length || 0;
      const fileLabel = mem.files.length ? ` · ${mem.files.slice(0, 3).join(", ")}` : "";
      const description = briefDescription(mem.content);
      const sourceLabel = sources ? ` · ${sources} sources` : "";
      const content = `- [${mem.id}] Memory · ${mem.type} · ${mem.title}${description ? ` — ${description}` : ""}${fileLabel}${sourceLabel}`;
      candidates.push({
        id: `memory:${mem.id}`,
        type: "memory",
        content,
        tokens: estimateTokens(content),
        recency: Date.parse(mem.updatedAt),
        score:
          relevance * 20 +
          files * 12 +
          phaseScore(mem.type) * 3 +
          (mem.strength / 10) * 6 +
          recencyScore(mem.updatedAt) +
          rrfBoost(lexicalMemoryRank.get(mem.id), 14) +
          rrfBoost(semanticMemoryRank.get(mem.id), 12),
        sourceKey: mem.sessionIds[0] || `memory:${mem.type}`,
        memoryId: mem.id,
      });
      memoryCandidateCount++;
    }

    // Prefer consolidated semantic memory. Only when retrieval finds no
    // relevant durable memory do we surface a small amount of prior episodic
    // evidence, which the agent can expand by stable ID if needed.
    if (memoryCandidateCount === 0) {
      episodicFallbackCandidates
        .sort((a, b) => b.score - a.score || b.recency - a.recency)
        .slice(0, 3)
        .forEach((candidate) => candidates.push(candidate));
    }

    candidates.sort((a, b) =>
      b.score - a.score ||
      (b.score / Math.max(1, b.tokens)) - (a.score / Math.max(1, a.tokens)) ||
      b.recency - a.recency,
    );

    const diversified: Candidate[] = [];
    const deferred: Candidate[] = [];
    const sourceCounts = new Map<string, number>();
    for (const candidate of candidates) {
      const count = sourceCounts.get(candidate.sourceKey) || 0;
      if (count >= 2) deferred.push(candidate);
      else {
        diversified.push(candidate);
        sourceCounts.set(candidate.sourceKey, count + 1);
      }
    }
    diversified.push(...deferred);

    const blocks: Candidate[] = [];
    let tokenCount = 0;
    for (const candidate of diversified) {
      if (blocks.length >= 40) break;
      if (blocks.some((block) => block.id === candidate.id)) continue;
      if (tokenCount + candidate.tokens > budget) continue;
      blocks.push(candidate);
      tokenCount += candidate.tokens;
    }
    if (!blocks.length) return "";
    for (const memoryId of new Set(blocks.flatMap((block) => block.memoryId ? [block.memoryId] : []))) {
      trackAccess(memoryId);
    }

    const includesEpisodicFallback = blocks.some((block) => block.type === "observation");
    const retrievalMode = includesEpisodicFallback ? "episodic fallback" : "semantic memory";
    const retrievalNote = objective
      ? `_Task-aware retrieval: ${phase}; ${retrievalMode} compact index; ${blocks.length} items; ~${tokenCount}/${budget} tokens._`
      : `_Semantic memory compact index: ${blocks.length} items; ~${tokenCount}/${budget} tokens._`;
    const expansionNote = "Each bracketed value is a stable ID. These entries are summaries, not complete records. Call `memory_get` once with the relevant IDs when full rationale, evidence, or details are needed.";
    return `# Agent Context\n\n${retrievalNote}\n\n${expansionNote}\n\n${blocks.map((block) => block.content).join("\n")}\n`;
  }

  function contextTokens(value: string): Set<string> {
    const stop = new Set(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "this", "to", "with"]);
    return new Set(
      (value.normalize("NFKC").toLowerCase().match(/[\p{L}\p{N}_-]{2,}/gu) || [])
        .map((token) => token.replace(/(?:ing|ed|es|s)$/i, ""))
        .filter((token) => token.length > 1 && !stop.has(token)),
    );
  }

  // ── Helpers ────────────────────────────────────────────────────────────

  function extractConcepts(content: string): string[] {
    const concepts = new Set<string>();
    const keywords = [
      "error", "bug", "fix", "crash", "panic", "timeout", "retry",
      "config", "setting", "env", "environment",
      "auth", "permission", "access", "token",
      "database", "schema", "migration", "query", "connection",
      "api", "endpoint", "route", "middleware", "login",
      "test", "unit", "integration", "mock", "stub",
      "build", "deploy", "pipeline", "ci", "cd",
      "refactor", "optimize", "performance", "memory", "cpu",
      "security", "vulnerability", "sanitize", "escape",
      "cache", "index", "search", "filter", "sort",
      "async", "promise", "callback", "event", "listener",
      "state", "store", "redux", "context", "hook",
      "type", "interface", "class", "module", "package",
    ];
    const lower = content.toLowerCase();
    for (const kw of keywords) {
      if (lower.includes(kw)) concepts.add(kw);
    }
    // Hashtags
    const hashtags = content.match(/#(\w+)/g);
    if (hashtags) {
      for (const h of hashtags) concepts.add(h.slice(1));
    }
    return [...concepts].slice(0, 10);
  }

  function extractFiles(content: string): string[] {
    const files = new Set<string>();
    // Match file paths: src/foo.ts, ./lib/bar.js, ../test/baz.py
    const patterns = [
      /(?:src|lib|pkg|test|app|src|vendor|node_modules|dist|build)\//g,
      /\/[\w.-]+\.(ts|js|tsx|jsx|py|rs|go|rb|java|c|h|cpp|json|yaml|yml|toml|md|css|scss|html|sh|bash)/g,
    ];
    for (const pattern of patterns) {
      let match;
      const str = content;
      while ((match = pattern.exec(str)) !== null) {
        const path = content.slice(Math.max(0, match.index - 50), match.index + match[0].length);
        if (path.includes("/")) files.add(path.slice(0, 300));
      }
    }
    return [...files].slice(0, 10);
  }

  function assignStrength(content: string): number {
    const lower = content.toLowerCase();
    if (/^fix|^bug|error|panic|crash|exception/i.test(lower)) return 8;
    if (/^decid|^architect|^design|^pattern/i.test(lower)) return 7;
    if (/^todo|^next|^future|suggestion/i.test(lower)) return 4;
    return 5;
  }

  // ── Memory Relations ─────────────────────────────────────────────────

  function relate(
    sourceId: string,
    targetId: string,
    type: MemoryRelationType,
    confidence: number = 0.5,
  ): MemoryRelation | null {
    // Validate both memories exist
    const source = get(sourceId);
    const target = get(targetId);
    if (!source || !target) return null;

    const relationId = generateId();
    const ts = now();
    const clampedConf = Math.max(0, Math.min(1, confidence || computeRelationConfidence(source, target, type)));

    db.prepare(`
      INSERT INTO relations (id, type, source_id, target_id, confidence, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(relationId, type, sourceId, targetId, clampedConf, ts);

    // Update related_ids on both memories
    db.prepare(`UPDATE memories SET related_ids = json_insert(related_ids, '$', ?) WHERE id IN (?, ?)`).run(
      targetId, sourceId, sourceId,
    );
    db.prepare(`UPDATE memories SET related_ids = json_insert(related_ids, '$', ?) WHERE id IN (?, ?)`).run(
      sourceId, targetId, targetId,
    );

    return { id: relationId, type, sourceId, targetId, confidence: clampedConf, createdAt: ts };
  }

  function computeRelationConfidence(
    source: Memory,
    target: Memory,
    relationType: MemoryRelationType,
  ): number {
    let score = 0.5;

    // Shared sessions boost confidence
    const sharedSessions = source.sessionIds.filter((sid) => target.sessionIds.includes(sid));
    score += Math.min(sharedSessions.length * 0.1, 0.3);

    // Recency boost
    const now = Date.now();
    const sourceAge = now - new Date(source.updatedAt).getTime();
    const targetAge = now - new Date(target.updatedAt).getTime();
    const sevenDays = 7 * 24 * 60 * 60 * 1000;
    const ninetyDays = 90 * 24 * 60 * 60 * 1000;

    if (sourceAge < sevenDays && targetAge < sevenDays) score += 0.1;
    else if (sourceAge > ninetyDays && targetAge > ninetyDays) score -= 0.1;

    // Relation-type adjustments
    if (relationType === "supersedes") score += 0.1;
    if (relationType === "contradicts") score -= 0.05;

    return Math.max(0, Math.min(1, score));
  }

  function getRelations(memoryId: string): MemoryRelation[] {
    const rows = db.prepare(
      `SELECT * FROM relations WHERE source_id = ? OR target_id = ? ORDER BY created_at DESC`
    ).all(memoryId, memoryId) as any[];

    return rows.map((r) => ({
      id: r.id,
      type: r.type as MemoryRelationType,
      sourceId: r.source_id,
      targetId: r.target_id,
      confidence: r.confidence ?? 0.5,
      createdAt: r.created_at,
    }));
  }

  function getRelatedMemories(
    memoryId: string,
    maxHops: number = 2,
    minConfidence: number = 0,
  ): Array<{ memory: Memory; hop: number; confidence: number }> {
    const allRelations = db.prepare(
      `SELECT * FROM relations`
    ).all() as any[];

    const visited = new Set<string>([memoryId]);
    const result: Array<{ memory: Memory; hop: number; confidence: number }> = [];
    const queue: Array<{ id: string; hop: number }> = [{ id: memoryId, hop: 0 }];
    const MAX_VISITED = 500;

    while (queue.length > 0 && visited.size < MAX_VISITED) {
      const current = queue.shift()!;
      if (current.hop >= maxHops) continue;
      visited.add(current.id);

      const memory = get(current.id);
      if (!memory) continue;

      // Find relations involving this memory
      const relatedRelations = allRelations.filter(
        (r) => r.source_id === current.id || r.target_id === current.id,
      );

      // Get the target memory IDs from these relations
      for (const rel of relatedRelations) {
        const targetId = rel.source_id === current.id ? rel.target_id : rel.source_id;
        if (visited.has(targetId)) continue;

        const targetMemory = get(targetId);
        if (!targetMemory) continue;

        visited.add(targetId);
        const confidence = rel.confidence ?? 0.5;

        if (current.hop >= 0 && confidence >= minConfidence) {
          result.push({ memory: targetMemory, hop: current.hop + 1, confidence });
        }

        queue.push({ id: targetId, hop: current.hop + 1 });
      }
    }

    result.sort((a, b) => b.confidence - a.confidence);
    return result;
  }

  function evolve(
    memoryId: string,
    newContent: string,
    newTitle?: string,
  ): { memory: Memory; previousId: string } | null {
    // First get the existing memory (must be latest)
    const existing = get(memoryId);
    if (!existing) return null;

    const ts = now();
    const evolved: Memory = {
      ...existing,
      id: generateId(),
      createdAt: ts,
      updatedAt: ts,
      title: newTitle || existing.title,
      content: newContent,
      version: (existing.version || 1) + 1,
      parentId: existing.id,
      supersedes: [existing.id, ...(existing.supersedes || [])],
      isLatest: true,
    };

    // Insert new version FIRST (so we can still reference it)
    db.prepare(`
      INSERT INTO memories (id, created_at, updated_at, type, title, content,
                            concepts, files, session_ids, strength, version,
                            parent_id, related_ids, source_observation_ids, is_latest, project,
                            workspace, access_count, last_accessed, working_tier, supersedes)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', '[]', 1, ?, ?, 0, NULL, 'cold', ?)
    `).run(
      evolved.id,
      ts, ts,
      existing.type,
      evolved.title,
      newContent,
      JSON.stringify(existing.concepts),
      JSON.stringify(existing.files),
      JSON.stringify(existing.sessionIds),
      existing.strength,
      evolved.version,
      memoryId,
      existing.project || null,
      existing.workspace || currentWorkspace,
      JSON.stringify([existing.id]),
    );

    // Mark old as non-latest
    db.prepare("UPDATE memories SET is_latest = 0 WHERE id = ?").run(memoryId);

    // Create supersedes relation
    const relationId = generateId();
    db.prepare(`
      INSERT INTO relations (id, type, source_id, target_id, confidence, created_at)
      VALUES (?, 'supersedes', ?, ?, 1.0, ?)
    `).run(relationId, evolved.id, memoryId, ts);

    // Update related_ids on new memory
    db.prepare(`UPDATE memories SET related_ids = json_insert(related_ids, '$', ?) WHERE id = ?`).run(
      memoryId, evolved.id,
    );

    return { memory: evolved, previousId: memoryId };
  }

  function removeRelation(relationId: string): boolean {
    const result = db.prepare("DELETE FROM relations WHERE id = ?").run(relationId);
    return result.changes > 0;
  }

  // ── Retention Scoring ────────────────────────────────────────────────

  const DEFAULT_DECAY: DecayConfig = {
    lambda: 0.01,
    sigma: 0.3,
    tierThresholds: { hot: 0.7, warm: 0.4, cold: 0.15 },
  };

  function resolveDecayConfig(input?: DecayConfigInput): { lambda: number; sigma: number; tierThresholds: { hot: number; warm: number; cold: number } } {
    const tierThresholds = {
      hot: input?.tierThresholds?.hot ?? 0.7,
      warm: input?.tierThresholds?.warm ?? 0.4,
      cold: input?.tierThresholds?.cold ?? 0.15,
    };
    return { lambda: input?.lambda ?? 0.01, sigma: input?.sigma ?? 0.3, tierThresholds };
  }

  function computeRetentionScore(
    id: string,
    config: DecayConfigInput = {},
  ): RetentionScore | null {
    const memory = get(id);
    if (!memory) return null;

    const resolved = resolveDecayConfig(config);
    const now = Date.now();

    // Time decay
    const deltaT = (now - new Date(memory.createdAt).getTime()) / (1000 * 60 * 60 * 24);
    const temporalDecay = Math.exp(-resolved.lambda * deltaT);

    // Reinforcement from access count
    const accessStats = getAccessStats(id);
    let reinforcementBoost = 0;
    if (accessStats) {
      // Each access contributes: sigma / daysSinceAccess
      const daysSinceAccess = (now - new Date(accessStats.lastAccessed).getTime()) / (1000 * 60 * 60 * 24);
      if (daysSinceAccess > 0) {
        reinforcementBoost = resolved.sigma / daysSinceAccess;
      }
    }

    // Salience from memory type and access count
    const typeWeights: Record<string, number> = {
      architecture: 0.9,
      bug: 0.7,
      pattern: 0.8,
      preference: 0.85,
      workflow: 0.6,
      fact: 0.5,
    };
    const baseSalience = typeWeights[memory.type] || 0.5;
    const accessBonus = Math.min(0.2, (accessStats?.accessCount || 0) * 0.02);
    const salience = Math.min(1, baseSalience + accessBonus);

    // Final retention score
    const score = Math.min(1, salience * temporalDecay + reinforcementBoost);

    // Determine tier
    let tier: WorkingMemoryTier = "cold";
    if (score >= resolved.tierThresholds.hot) tier = "hot";
    else if (score >= resolved.tierThresholds.warm) tier = "warm";

    return {
      id: memory.id,
      score,
      decayFactor: temporalDecay,
      reinforcementBoost,
      tier,
      type: memory.type,
      strength: memory.strength,
    };
  }

  function rescoreAll(config: DecayConfigInput = {}): RetentionScore[] {
    const allMemories = db.prepare(
      `SELECT * FROM memories WHERE is_latest = 1 AND workspace = ?`,
    ).all(currentWorkspace) as any[];
    const scores: RetentionScore[] = [];

    for (const row of allMemories) {
      const memory = rowToMemory(row);
      const score = computeRetentionScore(memory.id, config);
      if (score) scores.push(score);
    }

    scores.sort((a, b) => b.score - a.score);
    return scores;
  }

  function listByRetentionScore(config: DecayConfigInput = {}, limit: number = 50): RetentionScore[] {
    return rescoreAll(config).slice(0, limit);
  }

  // ── File Context Index ───────────────────────────────────────────────

  function getFileContext(file: string, sessionId?: string): FileContextEntry | null {
    const pattern = `%${file}%`;
    const rows = sessionId
      ? db.prepare(`
          SELECT id, session_id, type, title, narrative, importance, timestamp
          FROM observations
          WHERE workspace = ? AND (title LIKE ? OR narrative LIKE ? OR files LIKE ?) AND session_id = ?
          ORDER BY timestamp DESC
        `).all(currentWorkspace, pattern, pattern, pattern, sessionId) as any[]
      : db.prepare(`
          SELECT id, session_id, type, title, narrative, importance, timestamp
          FROM observations
          WHERE workspace = ? AND (title LIKE ? OR narrative LIKE ? OR files LIKE ?)
          ORDER BY timestamp DESC
        `).all(currentWorkspace, pattern, pattern, pattern) as any[];

    if (rows.length === 0) return null;

    return {
      file,
      observations: rows.map((r) => ({
        sessionId: r.session_id,
        obsId: r.id,
        type: r.type as ObservationType,
        title: r.title || "",
        narrative: r.narrative || "",
        importance: r.importance ?? 5,
        timestamp: r.timestamp,
      })),
    };
  }

  function getFilesContext(files: string[], sessionId?: string): FileContextEntry[] {
    return files
      .map((f) => getFileContext(f, sessionId))
      .filter((e): e is FileContextEntry => e !== null);
  }

  function rebuildFileIndex(): number {
    // Count observations with non-empty files array
    // Since JSON arrays like ["a"] are > 4 chars while [] is exactly 2 chars
    const count = db.prepare(
      `SELECT COUNT(*) as cnt FROM observations WHERE workspace = ? AND LENGTH(files) > 2`
    ).get(currentWorkspace) as { cnt: number };
    return count.cnt || 0;
  }

  // ── Export/Import ────────────────────────────────────────────────────

  function exportData(): ExportData {
    const sessions = listSessions();
    const memories = db.prepare(`SELECT * FROM memories WHERE is_latest = 1`).all().map(rowToMemory);
    const observations = db.prepare(`SELECT * FROM observations ORDER BY timestamp DESC`).all() as any[];
    const relations = db.prepare(`SELECT * FROM relations ORDER BY created_at DESC`).all() as any[];

    return {
      version: 1,
      exportedAt: now(),
      sessions,
      observations: observations.map((r) => ({
        id: r.id,
        sessionId: r.session_id,
        timestamp: r.timestamp,
        type: r.type as ObservationType,
        title: r.title || "",
        subtitle: r.subtitle,
        facts: safeParseJsonArray(r.facts),
        narrative: r.narrative || "",
        concepts: safeParseJsonArray(r.concepts),
        files: safeParseJsonArray(r.files),
        importance: r.importance ?? 5,
        consolidated: r.consolidated === 1 || r.consolidated === true,
      })),
      memories,
      relations: relations.map((r) => ({
        id: r.id,
        type: r.type as MemoryRelationType,
        sourceId: r.source_id,
        targetId: r.target_id,
        confidence: r.confidence ?? 0.5,
        createdAt: r.created_at,
      })),
    };
  }

  function importData(data: ImportData): ImportResult {
    const result: ImportResult = { imported: 0, skipped: 0, errors: [] };
    const mode = data.onConflict || "skip";

    // Import sessions
    for (const session of data.sessions) {
      try {
        const existing = getSession(session.id);
        if (existing && mode === "skip") {
          result.skipped++;
          continue;
        }
        if (existing && mode === "update") {
          updateSession(session.id, session);
        } else {
          createSession(session.id, session);
        }
        result.imported++;
      } catch (e) {
        result.errors.push(`Session ${session.id}: ${(e as Error).message}`);
      }
    }

    // Import observations
    for (const obs of data.observations) {
      try {
        const existing = db.prepare("SELECT id FROM observations WHERE id = ?").get(obs.id) as { id: string } | undefined;
        if (existing && mode === "skip") {
          result.skipped++;
          continue;
        }
        if (existing && mode === "update") {
          db.prepare(`
            UPDATE observations SET session_id=?, timestamp=?, type=?, title=?,
              subtitle=?, narrative=?, facts=?, concepts=?, files=?, importance=?
            WHERE id=?
          `).run(
            obs.sessionId, obs.timestamp, obs.type, obs.title,
            obs.subtitle || null, obs.narrative, JSON.stringify(obs.facts),
            JSON.stringify(obs.concepts), JSON.stringify(obs.files),
            obs.importance, obs.id,
          );
        } else {
          observe(
            { id: obs.id, sessionId: obs.sessionId, timestamp: obs.timestamp, hookType: "import" as any, raw: {} },
            obs,
          );
        }
        result.imported++;
      } catch (e) {
        result.errors.push(`Observation ${obs.id}: ${(e as Error).message}`);
      }
    }

    // Import memories
    for (const mem of data.memories) {
      try {
        const existing = db.prepare("SELECT id FROM memories WHERE id = ?").get(mem.id) as { id: string } | undefined;
        if (existing && mode === "skip") {
          result.skipped++;
          continue;
        }
        if (existing && mode === "update") {
          update(mem.id, { content: mem.content, title: mem.title });
        } else {
          create(mem.content, {
            type: mem.type,
            concepts: mem.concepts,
            files: mem.files,
            strength: mem.strength,
            sessionIds: mem.sessionIds,
            parentId: mem.parentId,
            project: mem.project,
          });
        }
        result.imported++;
      } catch (e) {
        result.errors.push(`Memory ${mem.id}: ${(e as Error).message}`);
      }
    }

    // Import relations
    if (data.relations) {
      for (const rel of data.relations) {
        try {
          const existing = db.prepare("SELECT id FROM relations WHERE id = ?").get(rel.id) as { id: string } | undefined;
          if (existing && mode === "skip") {
            result.skipped++;
            continue;
          }
          db.prepare(`
            INSERT OR IGNORE INTO relations (id, type, source_id, target_id, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
          `).run(rel.id, rel.type, rel.sourceId, rel.targetId, rel.confidence, rel.createdAt);
          result.imported++;
        } catch (e) {
          result.errors.push(`Relation ${rel.id}: ${(e as Error).message}`);
        }
      }
    }

    return result;
  }

  // ── Session ID tracking ────────────────────────────────────────────────

  let currentSessionId: string | null = null;
  function setCurrentSessionId(id: string): void {
    currentSessionId = id;
    // Ensure session exists
    if (!getSession(id)) {
      createSession(id, { project: "" });
    }
    // Sync workspace from session
    const session = getSession(id);
    if (session?.workspace) {
      currentWorkspace = normalizeWorkspacePath(session.workspace);
    }
  }

  function getCurrentSessionId(): string | null {
    return currentSessionId;
  }

  function setCurrentWorkspace(ws: string): void {
    currentWorkspace = normalizeWorkspacePath(ws);
  }

  function getCurrentWorkspace(): string {
    return currentWorkspace;
  }

  // ── Durable semantic extraction queue ────────────────────────────────

  type ExtractionJobRow = {
    id: string; session_id: string; workspace: string; payload: string;
    status: ExtractionJobStatus; attempts: number; created_at: string;
    updated_at: string; next_attempt_at: string; last_error: string | null;
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
  });

  function enqueueExtractionJob(sessionId: string, workspace: string, payload: string): ExtractionJob {
    const id = crypto.randomUUID();
    const timestamp = now();
    const safePayload = (() => {
      try { return JSON.stringify(sanitizePayload(JSON.parse(payload))); }
      catch { return JSON.stringify({ invalidPayload: sanitizeString(payload) }); }
    })();
    db.prepare(`INSERT INTO extraction_jobs
      (id, session_id, workspace, payload, status, attempts, created_at, updated_at, next_attempt_at)
      VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?)`)
      .run(id, sessionId, normalizeWorkspacePath(workspace), safePayload, timestamp, timestamp, timestamp);
    db.prepare("DELETE FROM extraction_jobs WHERE status = 'completed' AND updated_at < ?")
      .run(new Date(Date.now() - 7 * 86_400_000).toISOString());
    return rowToExtractionJob(db.prepare("SELECT * FROM extraction_jobs WHERE id = ?").get(id) as ExtractionJobRow);
  }

  function claimExtractionJob(): ExtractionJob | null {
    const timestamp = now();
    const row = db.prepare(`SELECT * FROM extraction_jobs
      WHERE workspace = ? AND status = 'pending' AND next_attempt_at <= ?
      ORDER BY created_at ASC LIMIT 1`)
      .get(currentWorkspace, timestamp) as ExtractionJobRow | undefined;
    if (!row) return null;
    const updated = db.prepare(`UPDATE extraction_jobs
      SET status = 'running', attempts = attempts + 1, updated_at = ?
      WHERE id = ? AND status = 'pending'`).run(timestamp, row.id);
    if (updated.changes !== 1) return null;
    return rowToExtractionJob({ ...row, status: "running", attempts: row.attempts + 1, updated_at: timestamp });
  }

  function completeExtractionJob(id: string): void {
    const timestamp = now();
    db.prepare("UPDATE extraction_jobs SET status = 'completed', updated_at = ?, last_error = NULL WHERE id = ?")
      .run(timestamp, id);
  }

  function failExtractionJob(id: string, error: string, retryDelayMs: number = 1_000): void {
    const row = db.prepare("SELECT attempts FROM extraction_jobs WHERE id = ?")
      .get(id) as { attempts: number } | undefined;
    if (!row) return;
    const terminal = row.attempts >= 3;
    const timestamp = now();
    const nextAttempt = new Date(Date.now() + Math.max(0, retryDelayMs)).toISOString();
    db.prepare(`UPDATE extraction_jobs SET status = ?, updated_at = ?, next_attempt_at = ?, last_error = ?
      WHERE id = ?`).run(terminal ? "failed" : "pending", timestamp, nextAttempt, error.slice(0, 1000), id);
  }

  function listExtractionJobs(status?: ExtractionJobStatus): ExtractionJob[] {
    const rows = status
      ? db.prepare("SELECT * FROM extraction_jobs WHERE workspace = ? AND status = ? ORDER BY created_at")
          .all(currentWorkspace, status)
      : db.prepare("SELECT * FROM extraction_jobs WHERE workspace = ? ORDER BY created_at")
          .all(currentWorkspace);
    return (rows as ExtractionJobRow[]).map(rowToExtractionJob);
  }

  // ── Public API ─────────────────────────────────────────────────────────

  return {
    createSession,
    getSession,
    listSessions,
    updateSession,
    clearSessions,
    discardEmptySession,
    observe,
    getObservation,
    listObservations,
    listRecentObservations,
    searchObservations,
    expandEntries,
    clearObservations,
    create,
    get,
    getAny,
    list,
    remove: deleteEntry,
    clearMemories,
    update,
    recall,
    consolidate,
    enqueueExtractionJob,
    claimExtractionJob,
    completeExtractionJob,
    failExtractionJob,
    listExtractionJobs,
    getContext,
    upsertEmbedding,
    hasEmbedding,
    searchEmbeddings,
    setCurrentSessionId,
    getCurrentSessionId,
    setCurrentWorkspace,
    getCurrentWorkspace,

    // Dedup
    dedupCheck,
    dedupRecord,

    // Sliding Window
    slidingWindowCap,

    // Access Tracker
    trackAccess,
    getAccessStats,

    // Working Memory Tiers
    getWorkingMemoryTier,
    setWorkingMemoryTier,
    autoTierMemories,

    // Auto-Forget
    autoForget,

    // Memory Relations
    relate,
    getRelations,
    getRelatedMemories,
    evolve,
    removeRelation,

    // Retention Scoring
    computeRetentionScore,
    rescoreAll,
    listByRetentionScore,

    // File Context Index
    getFileContext,
    getFilesContext,
    rebuildFileIndex,

    // Export/Import
    exportData,
    importData,

    close() {
      db.close();
    },
  };
}
