// ── SQLite-backed Memory Store ───────────────────────────────────────────────
import { Database } from "bun:sqlite";
import type {
  MemoryEntry,
  MemoryQuery,
  CreateMemoryOptions,
  MemoryStore,
  RecallOptions,
} from "./types.js";

function generateId(): string {
  return crypto.randomUUID();
}

function now(): string {
  return new Date().toISOString();
}

function extractTags(content: string): string[] {
  const hashtags = content.match(/#(\w+)/g) || [];
  return [...new Set(hashtags.map((t) => t.slice(1)))];
}

function assignImportance(content: string, requested?: number): number {
  if (requested !== undefined && requested >= 1 && requested <= 10)
    return requested;
  const lower = content.toLowerCase();
  if (/^fix|^bug|error|panic|crash/i.test(lower)) return 7;
  if (/^todo|^next|future/i.test(lower)) return 4;
  return 5;
}

export function createMemoryStore(dbPath: string): MemoryStore {
  const db = new Database(dbPath);

  // Create table only — no FTS5 (Bun bundled SQLite has issues with virtual tables in exec())
  db.exec(`
    CREATE TABLE IF NOT EXISTS memories (
      id TEXT PRIMARY KEY,
      content TEXT NOT NULL,
      tags TEXT NOT NULL DEFAULT '[]',
      source TEXT NOT NULL DEFAULT '',
      session_id TEXT NOT NULL DEFAULT '',
      importance INTEGER NOT NULL DEFAULT 5,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
    CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
    CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
  `);

  return {
    create(content: string, options?: CreateMemoryOptions): MemoryEntry {
      const id = generateId();
      const ts = now();
      const tags = options?.tags ?? (options?.autoTags ? extractTags(content) : []);
      const importance = assignImportance(content, options?.importance);

      db.prepare(
        `INSERT INTO memories (id, content, tags, source, session_id, importance, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        id,
        content,
        JSON.stringify(tags),
        options?.source || "",
        options?.sessionId || "",
        importance,
        ts,
        ts,
      );

      return {
        id,
        content,
        tags,
        source: options?.source || "",
        sessionId: options?.sessionId || "",
        importance,
        createdAt: ts,
        updatedAt: ts,
      };
    },

    get(id: string): MemoryEntry | null {
      const row = db.prepare("SELECT * FROM memories WHERE id = ?").get(
        id,
      ) as any;
      return row ? deserialize(row) : null;
    },

    list(query?: MemoryQuery): MemoryEntry[] {
      const conditions: string[] = [];
      const params: (string | number)[] = [];

      if (query?.search) {
        // Plain-text search — no FTS5 dependency
        conditions.push("content LIKE ?");
        params.push(`%${escapeLike(query.search)}%`);
      }

      if (query?.tags?.length) {
        for (const tag of query.tags) {
          // json_each iterates the JSON array stored in tags column
          conditions.push("json_each.value = ?");
          params.push(tag);
        }
      }

      if (query?.source) {
        conditions.push("source = ?");
        params.push(query.source);
      }

      if (query?.sessionId) {
        conditions.push("session_id = ?");
        params.push(query.sessionId);
      }

      if (query?.minImportance !== undefined) {
        conditions.push("importance >= ?");
        params.push(query.minImportance);
      }

      const whereClause = conditions.length > 0
        ? `WHERE ${conditions.join(" AND ")}`
        : "";
      const limit = query?.limit ?? 10;

      // Tag filtering uses HAVING + json_each to enforce all tags match
      if (query?.tags?.length) {
        // Build the tag-matching clause with explicit HAVING count
        const baseConditions = conditions.filter((c, i) => !c.includes("json_each"));
        const baseWhere = baseConditions.length > 0 ? `WHERE ${baseConditions.join(" AND ")}` : "";

        const tags = query.tags!;
        // eslint-disable-next-line no-useless-escape
        const tagParamList = Array(tags.length).fill("?").join(", ");
        const sql = `
          SELECT m.* FROM memories m
          LEFT JOIN json_each(m.tags) je
          ${baseWhere}
          HAVING SUM(je.value IN (${tagParamList})) >= ?
          ORDER BY m.importance DESC, m.created_at DESC
          LIMIT ?
        `;

        // Rebuild params: base conditions first, then tag values, then count + limit
        const baseParams: (string | number)[] = [];
        for (const c of baseConditions) {
          baseParams.push(params.shift()!);
        }
        return db.prepare(sql)
          .all(...baseParams, ...tags, tags.length, limit)
          .map(deserialize) as MemoryEntry[];
      }

      const sql = `SELECT * FROM memories ${whereClause} ORDER BY importance DESC, created_at DESC LIMIT ?`;
      return db.prepare(sql)
        .all(...params, limit)
        .map(deserialize) as MemoryEntry[];
    },

    delete(id: string): boolean {
      const result = db.prepare("DELETE FROM memories WHERE id = ?").run(id);
      return result.changes > 0;
    },

    update(
      id: string,
      updates: Partial<Pick<MemoryEntry, "content" | "tags" | "importance">>,
    ): MemoryEntry | null {
      const sets: string[] = [];
      const params: (string | number)[] = [];

      if (updates.content !== undefined) {
        sets.push("content = ?");
        params.push(updates.content);
      }
      if (updates.tags !== undefined) {
        sets.push("tags = ?");
        params.push(JSON.stringify(updates.tags));
      }
      if (updates.importance !== undefined) {
        sets.push("importance = ?");
        params.push(updates.importance);
      }

      sets.push("updated_at = ?");
      params.push(now());
      params.push(id);

      db.prepare(`UPDATE memories SET ${sets.join(", ")} WHERE id = ?`).run(
        ...params,
      );
      return this.get(id);
    },

    recall(query: MemoryQuery, options?: RecallOptions): string {
      const memories = this.list(query);
      if (!memories.length) return "";

      const format = options?.format || "text";
      const template = options?.template || "{{content}}";

      if (format === "markdown") {
        return memories
          .map(
            (m: MemoryEntry) =>
              `## ${m.id} [${m.importance}/10]\n\n${m.content}\n\n${
                m.tags.length ? "`" + m.tags.join("` `") + "`" : ""
              }`,
          )
          .join("\n\n---\n\n");
      }

      if (format === "system-prompt") {
        return memories
          .map((m: MemoryEntry) => `## ${m.source || "memory"} [${m.importance}/10]\n\n${m.content}`)
          .join("\n\n");
      }

      // text (default)
      return memories
        .map(
          (m: MemoryEntry) =>
            template
              .replace("{{content}}", m.content)
              .replace("{{importance}}", String(m.importance)),
        )
        .join("\n\n");
    },

    close(): void {
      db.close();
    },
  };
}

function escapeLike(val: string): string {
  return val.replace(/([%_\\])/g, "\\$1");
}

function deserialize(row: any): MemoryEntry {
  return {
    id: row.id,
    content: row.content,
    tags: safeParseJsonArray(row.tags) || [],
    source: row.source || "",
    sessionId: row.session_id || "",
    importance: row.importance ?? 5,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

function safeParseJsonArray(val: string): string[] {
  try {
    return JSON.parse(val);
  } catch {
    return [];
  }
}
