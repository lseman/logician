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

/** Parse a numeric artifact ID from the URL host. */
function parseArtifactId(url: InternalUrl): string {
	const id = url.rawHost || url.hostname;
	if (!id) {
		throw new Error("artifact:// URL requires a numeric ID: artifact://<id>");
	}
	if (!/^\d+$/.test(id)) {
		throw new Error(`artifact:// ID must be numeric, got: ${id}`);
	}
	return id;
}

/** Check if a selector string looks like a pagination selector. */
function isSelector(value: string): boolean {
	return /^(raw|conflicts|-?\d+(?:[-+]\d+)?(?:,\d+(?:[-+]\d+)?)*)$/i.test(
		value,
	);
}

/** Extract a selector from the pathname (e.g. /3:50 → 50). */
function extractSelector(pathname: string): string | null {
	const colonIdx = pathname.lastIndexOf(":");
	if (colonIdx < 0) return null;
	const suffix = pathname.slice(colonIdx + 1);
	return isSelector(suffix) ? suffix : null;
}

/** Extract a range from a selector string. Returns null for non-range selectors. */
function parseRange(
	selector: string,
): { offset?: number; limit?: number } | null {
	if (selector === "raw") return null;
	if (selector.includes(",")) {
		const parts = selector.split(",").map(p => parseInt(p, 10));
		if (parts.some(Number.isNaN)) return null;
		return { offset: parts[0], limit: parts[1] - parts[0] + 1 };
	}
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
		_context?: ResolveContext,
	): Promise<InternalResource> {
		const registry = ArtifactRegistry.instance();

		return this.#resolveWithRegistry(registry, url);
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
	): Promise<InternalResource> {
		// Bare artifact:// — list available IDs
		if (!(url.rawHost || url.hostname)) {
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

		const id = parseArtifactId(url);

		// Check for line-range selector in pathname
		const selector = extractSelector(url.pathname);
		if (selector) {
			return this.#resolveWithSelector(registry, id, url, selector);
		}

		// No selector — return full content
		return this.#resolveFull(registry, id, url);
	}

	async #resolveFull(
		registry: ArtifactRegistry,
		id: string,
		url: InternalUrl,
	): Promise<InternalResource> {
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
			// Apply pagination
			const offset = range.offset ?? 0;
			const limit = range.limit ?? content.length;
			const bytesPerLine = Math.max(
				1,
				Math.floor(
					Buffer.byteLength(content, "utf-8") /
						Math.max(1, content.split("\n").length),
				),
			);
			const byteOffset = Math.min(
				offset * bytesPerLine,
				Buffer.byteLength(content, "utf-8"),
			);
			const byteLimit = Math.min(
				limit * bytesPerLine,
				Buffer.byteLength(content, "utf-8") - byteOffset,
			);
			const slice = content.substring(byteOffset, byteOffset + byteLimit);
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
