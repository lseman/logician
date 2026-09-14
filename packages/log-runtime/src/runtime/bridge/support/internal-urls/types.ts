// ── Internal URL types ───────────────────────────────────────────────────────
// Lightweight types for the internal URL routing system.
// Registered internal URLs are resolved by tools like read,
// providing access to agent resources
// without exposing filesystem paths.

/** Resource payload returned by protocol handlers. */
export interface InternalResource {
	url: string;
	content: string;
	contentType?: "text/plain" | "text/markdown" | "application/json";
	size?: number;
	sourcePath?: string;
	notes?: string[];
	isDirectory?: boolean;
	/**
	 * True when this resource cannot be edited via write. Stamped by the
	 * router from {@link ProtocolHandler.immutable} when a handler's resolve()
	 * doesn't set it itself; a value set here always wins.
	 */
	immutable?: boolean;
}

/** Parsed resource link. Only the scheme is normalized; the target is opaque. */
export interface InternalUrl {
	readonly scheme: string;
	readonly target: string;
	readonly host: string;
	readonly pathname: string;
	readonly href: string;
}

/** Autocomplete candidate for URL completion. */
export interface UrlCompletion {
	value: string;
	description?: string;
}

/** Context passed to protocol handlers during resolution. */
export interface ResolveContext {
	signal?: AbortSignal | undefined;
	allowedPaths?: string[] | undefined;
	allowAllPaths?: boolean | undefined;
	skills?: Array<{ name: string; content: string; path: string }> | undefined;
	memory?:
		| {
				listObservations: (
					sessionId: string,
					limit: number,
				) => Promise<
					Array<{
						id: string;
						content: string;
						metadata?: Record<string, unknown>;
					}>
				>;
				listMemories: (query?: Record<string, unknown>) => Promise<
					Array<{
						id: string;
						content: string;
						metadata?: Record<string, unknown>;
					}>
				>;
		  }
		| undefined;
	cwd?: string | undefined;
	/**
	 * When set, handlers that would otherwise materialize expensive content
	 * (e.g. reading a multi-MiB artifact just to expose its sourcePath) may
	 * return the resource shape without content. Handlers that cannot
	 * separate path from content ignore this flag.
	 */
	pathOnly?: boolean | undefined;
}

/**
 * Context passed to protocol handlers during write. Mirrors the subset of
 * {@link ResolveContext} a write needs.
 */
export interface WriteContext {
	cwd?: string | undefined;
	allowedPaths?: string[] | undefined;
	allowAllPaths?: boolean | undefined;
	signal?: AbortSignal | undefined;
}

/** Memory entry shape for memory:// resolution. */
export interface MemoryEntry {
	id: string;
	content: string;
	metadata?: Record<string, unknown>;
}

/** Protocol handler for a specific internal URL scheme. */
export interface ProtocolHandler {
	scheme: string;
	/** Whether resources produced by this handler are editable via write. Every handler must declare a stance. */
	immutable: boolean;
	resolve: (
		url: InternalUrl,
		context?: ResolveContext,
	) => Promise<InternalResource>;
	/**
	 * Optional write hook. When present, the write tool dispatches
	 * `write(url, content)` to this handler instead of rejecting the scheme.
	 * Handlers that omit this are read-only for write.
	 */
	write?: (
		url: InternalUrl,
		content: string,
		context?: WriteContext,
	) => Promise<string | void>;
	complete?: (
		query: string,
		context?: ResolveContext,
	) => Promise<UrlCompletion[]>;
}
