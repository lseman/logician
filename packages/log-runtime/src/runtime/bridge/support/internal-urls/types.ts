// ── Internal URL types ───────────────────────────────────────────────────────
// Enhanced types for the internal URL routing system, inspired by oh-my-pi's
// protocol specification model. Handlers declare their capabilities via SchemeSpec
// so tools consult the router instead of branching on scheme names.

/**
 * How a scheme's resources exist.
 * - `file`: resolved content is the byte-identical content of the file locate() returns.
 * - `virtual`: content is rendered by the handler (may still locate a backing file).
 * - `remote`: content lives on another host or service; never locatable.
 */
export type SchemeBacking = "file" | "virtual" | "remote";

/**
 * Read-selector grammar after the URL.
 * - `lines`: any trailing :<selector> chain is a read selector (artifact://3:raw:1-50).
 * - `none`: never peel; the URL is passed through as written.
 */
export type SchemeSelectors = "lines" | "none";

/** Write policy for a writable scheme. Absent on read-only schemes. */
export interface SchemeWritePolicy {
	/** Who performs the write: file (standard tools) or handler (custom logic). */
	via: "file" | "handler";
}

/**
 * Rich scheme declaration that tells tools how to consume a protocol.
 * Declared on ProtocolHandler via spec().
 */
export interface SchemeSpec {
	/** How the scheme's resources exist (file, virtual, remote). */
	backing?: SchemeBacking;
	/** Read-selector grammar after the URL. */
	selectors?: SchemeSelectors;
}

/**
 * Per-scheme metadata for system prompt generation and tool introspection.
 * Populated by ProtocolHandler.describe().
 */
export interface SchemeHost {
	/** The scheme name (e.g., "memory", "skill"). */
	scheme: string;
	/** Whether this scheme is currently addressable (has content). */
	addressable?: boolean;
	/** Count of addressable resources (rules, skills, etc.) — used for promptDoc gating. */
	count?: number;
}

// ── Resource & URL types ──────────────────────────────────────────────────────

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
	 * router from ProtocolHandler.immutable when a handler's resolve()
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
 * ResolveContext a write needs.
 */
export interface WriteContext {
	cwd?: string | undefined;
	allowedPaths?: string[] | undefined;
	allowAllPaths?: boolean | undefined;
	signal?: AbortSignal | undefined;
}

// ── Protocol Handler Interface ────────────────────────────────────────────────

/** Protocol handler for a specific internal URL scheme. */
export interface ProtocolHandler {
	/** The URL scheme (e.g., "memory", "skill"). Lowercase only. */
	scheme: string;
	/** Whether resources produced by this handler are editable via write. */
	immutable: boolean;
	/** Rich scheme declaration — tells tools how to consume this protocol. */
	spec?: SchemeSpec | Record<string, unknown>;
	/**
	 * Resolve a URL to its resource content. Called by the router when a tool
	 * reads an internal URL.
	 */
	resolve: (
		url: InternalUrl,
		context?: ResolveContext,
	) => Promise<InternalResource>;
	/**
	 * Optional write hook. When present, the write tool dispatches
	 * `write(url, content)` to this handler instead of rejecting the scheme.
	 */
	write?: (
		url: InternalUrl,
		content: string,
		context?: WriteContext,
	) => Promise<string | void>;
	/**
	 * Optional autocomplete for URL completion in the TUI.
	 */
	complete?: (
		query: string,
		context?: ResolveContext,
	) => Promise<UrlCompletion[]>;
	/**
	 * Optional prompt document describing valid URL forms for this scheme.
	 * The router includes this in the system prompt when addressable resources exist.
	 * Return undefined when no resources are available (scheme is a no-op).
	 */
	promptDoc?: (host: SchemeHost) => string | undefined;
}
