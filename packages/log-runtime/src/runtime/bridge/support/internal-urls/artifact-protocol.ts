// ── artifact:// protocol handler ──────────────────────────────────────────────
// Resolves session-scoped artifact files.
// URL forms:
//   artifact://          — lists available artifact IDs
//   artifact://<id>      — reads artifact by ID
//   artifact://<id>:50   — reads artifact with line-range selector
//   artifact://<id>:raw  — reads artifact verbatim
//   artifact://<id>:5-200 — reads lines 5 to 200 of artifact
//
// Pagination via offset/limit is handled by the read tool.

import { ArtifactRegistry } from "./artifact-manager";
import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
	UrlCompletion,
} from "./types";

const MAX_ARTIFACT_BYTES = 8 * 1024 * 1024; // 8 MiB

/**
 * Check if a selector string looks like a pagination selector. Comma-separated
 * multi-range lists (`1-20,30+5`) are intentionally NOT accepted — they were
 * never documented (`:50`, `:raw`, `:5-200` only) and `parseRange` below only
 * ever handled a single pair, silently mis-parsing anything else. A selector
 * this doesn't match falls through to a full-content read instead.
 */
function isSelector(value: string): boolean {
	return /^(raw|conflicts|-?\d+(?:[-+]\d+)?)$/i.test(value);
}

/**
 * Parse `<id>` or `<id>:<selector>` from the URL. The shared parser only puts
 * the first path segment into `url.host`; the rest lands in `url.pathname`
 * (and never stops at `:` either) — reassemble the full address before
 * splitting, the same pattern log-protocol.ts uses for `/`-separated paths.
 */
function parseIdAndSelector(url: InternalUrl): { id: string; selector: string | null } {
	const full = url.pathname === "/" ? url.host : `${url.host}${url.pathname}`;
	if (!full) {
		throw new Error("artifact:// URL requires a numeric ID: artifact://<id>");
	}
	const colonIdx = full.indexOf(":");
	const id = colonIdx < 0 ? full : full.slice(0, colonIdx);
	if (!/^\d+$/.test(id)) {
		throw new Error(`artifact:// ID must be numeric, got: ${id}`);
	}
	if (colonIdx < 0) return { id, selector: null };
	const suffix = full.slice(colonIdx + 1);
	return { id, selector: isSelector(suffix) ? suffix : null };
}

/** Extract a range from a selector string. Returns null for non-range selectors. */
function parseRange(
	selector: string,
): { offset?: number; limit?: number } | null {
	if (selector === "raw") return null;
	const plusMatch = selector.match(/^(\d+)\+(\d+)$/);
	if (plusMatch) {
		const [, offset, count] = plusMatch;
		return { offset: parseInt(offset, 10), limit: parseInt(count, 10) };
	}
	const dashMatch = selector.match(/^(\d+)-(\d+)$/);
	if (dashMatch) {
		const [, start, end] = dashMatch;
		const offset = Math.max(0, parseInt(start, 10) - 1);
		const limit = parseInt(end, 10) - offset + 1;
		return { offset, limit };
	}
	const num = parseInt(selector, 10);
	if (Number.isNaN(num)) return null;
	return { limit: num };
}

export class ArtifactProtocolHandler implements ProtocolHandler {
	readonly scheme = "artifact";
	readonly immutable = true;

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const registry = ArtifactRegistry.instance();

		return this.#resolveWithRegistry(registry, url, context);
	}

	async complete(
		_query: string,
		_context?: ResolveContext,
	): Promise<UrlCompletion[]> {
		const registry = ArtifactRegistry.instance();
		const ids = await registry.listIds();
		return ids.map(id => ({ value: id, description: `Artifact ${id}` }));
	}

	async #resolveWithRegistry(
		registry: ArtifactRegistry,
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		// Bare artifact:// — list available IDs
		if (!url.host) {
			const ids = await registry.listIds();
			const content =
				ids.length > 0
					? `# Artifacts\n\n${ids.map((id: string) => `  - ${id}`).join("\n")}`
					: "# Artifacts\n\nNo artifacts have been saved yet.";
			return {
				url: url.href,
				content,
				contentType: "text/markdown",
			};
		}

		const { id, selector } = parseIdAndSelector(url);
		if (selector) {
			return this.#resolveWithSelector(registry, id, url, selector);
		}

		// No selector — return full content
		return this.#resolveFull(registry, id, url, context);
	}

	async #resolveFull(
		registry: ArtifactRegistry,
		id: string,
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		if (context?.pathOnly) {
			const sourcePath = await registry.getPath(id);
			if (sourcePath === null) {
				const available = await registry.listIds();
				const hint =
					available.length > 0
						? `\nAvailable: ${available.join(", ")}`
						: "\nNo artifacts have been saved yet.";
				throw new Error(`Unknown artifact: ${id}${hint}`);
			}
			return {
				url: url.href,
				content: "",
				contentType: "text/plain",
				sourcePath,
			};
		}

		const content = await registry.read(id);
		if (content === null) {
			const available = await registry.listIds();
			const hint =
				available.length > 0
					? `\nAvailable: ${available.join(", ")}`
					: "\nNo artifacts have been saved yet.";
			throw new Error(`Unknown artifact: ${id}${hint}`);
		}

		if (content.length > MAX_ARTIFACT_BYTES) {
			throw new Error(
				`artifact://${id} exceeds ${MAX_ARTIFACT_BYTES / 1024 / 1024} MiB limit; use \`read\` with line selectors or the backing path for large files`,
			);
		}

		return {
			url: url.href,
			content,
			contentType: "text/plain",
			size: Buffer.byteLength(content, "utf-8"),
		};
	}

	async #resolveWithSelector(
		registry: ArtifactRegistry,
		id: string,
		url: InternalUrl,
		selector: string,
	): Promise<InternalResource> {
		const content = await registry.read(id);
		if (content === null) {
			throw new Error(`Unknown artifact: ${id}`);
		}

		const range = parseRange(selector);
		if (range) {
			// Slice by line directly — offset/limit are already line values
			// from parseRange. (Previously approximated a byte range from an
			// average bytes-per-line and sliced with substring(), which indexes
			// by UTF-16 code unit: wrong for non-ASCII content and not actually
			// a line boundary either way.)
			const lines = content.split("\n");
			const offset = range.offset ?? 0;
			const limit = range.limit ?? lines.length;
			const slice = lines.slice(offset, offset + limit).join("\n");
			return {
				url: url.href,
				content: slice,
				contentType: "text/plain",
				size: Buffer.byteLength(slice, "utf-8"),
			};
		}

		// 'raw' — return verbatim
		if (selector === "raw") {
			return {
				url: url.href,
				content,
				contentType: "text/plain",
				size: Buffer.byteLength(content, "utf-8"),
			};
		}

		// Unknown selector — fall back to full
		return this.#resolveFull(registry, id, url);
	}
}
