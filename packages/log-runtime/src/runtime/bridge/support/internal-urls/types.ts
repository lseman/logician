// ── Internal URL types ───────────────────────────────────────────────────────
// Lightweight types for the internal URL routing system.
// Internal URLs (skill://, rule://, memory://, local://, conflict://, xd://)
// are resolved by tools like read_file, providing access to agent resources
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
}

/** Parsed internal URL with preserved host casing. */
export interface InternalUrl {
	scheme: string;
	host: string;
	rawHost: string;
	pathname: string;
	href: string;
	search: string;
	hash: string;
	protocol: string;
	port: string;
	username: string;
	password: string;
	origin: string;
	toString?: () => string;
	searchParams: URLSearchParams;
	hostname: string;
}

/** Autocomplete candidate for URL completion. */
export interface UrlCompletion {
	value: string;
	description?: string;
}

/** Context passed to protocol handlers during resolution. */
export interface ResolveContext {
	skills?: Array<{ name: string; content: string; path: string }>;
	rules?: Array<{ name: string; content: string; path: string }>;
	memory?: {
		listObservations: (
			sessionId: string,
			limit: number,
		) => Promise<
			Array<{ id: string; content: string; metadata?: Record<string, unknown> }>
		>;
		listMemories: (
			query?: Record<string, unknown>,
		) => Promise<
			Array<{ id: string; content: string; metadata?: Record<string, unknown> }>
		>;
	};
	cwd?: string;
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
	immutable?: boolean;
	resolve: (
		url: InternalUrl,
		context?: ResolveContext,
	) => Promise<InternalResource>;
	complete?: (
		query: string,
		context?: ResolveContext,
	) => Promise<UrlCompletion[]>;
}
