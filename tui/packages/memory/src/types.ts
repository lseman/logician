// ── Memory Domain Types ──────────────────────────────────────────────────────

export interface MemoryEntry {
  id: string
  content: string
  tags: string[]
  /** Source event that created this memory (tool name, hook phase, etc.) */
  source: string
  /** Session ID the memory belongs to */
  sessionId: string
  /** Importance score 1-10. Higher = more persistent via retention logic */
  importance: number
  createdAt: string // ISO timestamp
  updatedAt: string // ISO timestamp
}

export interface MemoryQuery {
  /** Full-text or tag search */
  search?: string
  /** Filter by tags (AND semantics) */
  tags?: string[]
  /** Filter by source */
  source?: string
  /** Filter by session */
  sessionId?: string
  /** Minimum importance score */
  minImportance?: number
  /** Max results (default: 10) */
  limit?: number
}

export interface CreateMemoryOptions {
  /** Auto-assign tags from content heuristics if not provided */
  autoTags?: boolean
  /** Explicit tags to set (overrides auto-tagging) */
  tags?: string[]
  /** Importance score, default: 5 */
  importance?: number
  /** Source event that created this memory (tool name, hook phase, etc.) */
  source?: string
  /** Session ID to associate with */
  sessionId?: string
}

export interface MemoryStore {
  create(content: string, options?: CreateMemoryOptions): MemoryEntry
  get(id: string): MemoryEntry | null
  list(query?: MemoryQuery): MemoryEntry[]
  delete(id: string): boolean
  update(id: string, updates: Partial<Pick<MemoryEntry, 'content' | 'tags' | 'importance'>>): MemoryEntry | null
  recall(query: MemoryQuery, options?: RecallOptions): string
  /** Close the underlying database connection */
  close(): void
}

export interface RecallOptions {
  /** Inject memories into a specific context (e.g., system prompt) */
  format?: 'text' | 'system-prompt' | 'markdown'
  /** Prepend a label before each memory */
  template?: string
}
