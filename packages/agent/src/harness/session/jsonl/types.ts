// ── JSONL session storage types ──────────────────────────────────────────
// Ported from pi coding agent's harness/session/jsonl/types.ts, minus the
// legacy v3-on-disk-format migration fields (sourceFormat, legacyParentSessionPath)
// since agent has no prior on-disk format to stay compatible with —
// every session this backend writes is the current (pi's "v4") format.

import type { FileSystem } from "../../../env/execution-env.ts";
import type {
	JsonValue,
	SessionCreateOptions,
	SessionMetadata,
} from "../types.ts";

export type JsonlSessionRepoFileSystem = Pick<
	FileSystem,
	| "absolutePath"
	| "joinPath"
	| "readTextFile"
	| "readTextLines"
	| "writeFile"
	| "appendFile"
	| "renameFile"
	| "fileInfo"
	| "listDir"
	| "exists"
	| "createDir"
	| "remove"
>;

export interface JsonlSessionRepoOptions {
	fs: JsonlSessionRepoFileSystem;
	/** Root containing cwd-encoded session directories. */
	sessionsRoot: string;
}

export interface JsonlSessionMetadata extends SessionMetadata {
	cwd: string;
	path: string;
	/** Filesystem modification time as milliseconds since Unix epoch. */
	modifiedAt: number;
	/** Opaque application-owned metadata. */
	metadata?: Record<string, JsonValue>;
}

export interface JsonlSessionCreateOptions extends SessionCreateOptions {
	cwd: string;
	metadata?: Record<string, JsonValue>;
}

export interface JsonlSessionListOptions {
	cwd?: string;
}

export interface JsonlSessionHeader {
	kind: "header";
	version: 1;
	id: string;
	createdAt: number;
	cwd: string;
	parentSessionId?: string;
	metadata?: Record<string, JsonValue>;
}
