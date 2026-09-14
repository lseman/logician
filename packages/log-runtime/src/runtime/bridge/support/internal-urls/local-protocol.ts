// ── '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local' protocol handler ────────────────────────────────────────────────
// Resolves session-local artifacts (JSONL journals, session meta, etc.).
// URL forms:
//   '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local'<path>     — reads a file under the session artifacts directory
//   '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local/0'          — reads artifact by numeric ID
//   '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local/0:10-30'    — artifact with line-range selector
//   '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local/0:raw'      — artifact verbatim

import * as fs from "node:fs/promises";
import * as path from "node:path";
import { ArtifactRegistry } from "./artifact-manager";
import { ensureInsideCwd } from "../../../../capabilities/tools/support/utils/path-utils.ts";
import { atomicWriteFile } from "../../../../capabilities/tools/support/utils/atomic-write.ts";
import { formatSize } from "../../../../capabilities/tools/support/utils/truncate.ts";

import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
	WriteContext,
} from "./types";

/** Path to a local session artifact. */
interface LocalEntry {
	path: string;
	name: string;
	type: "file" | "directory";
}

const LOCAL_TEXT_SNIFF_BYTES = 8192;
const LOCAL_TEXT_RESOURCE_MAX_BYTES = 1024 * 1024; // 1 MiB
const MAX_ARTIFACT_BYTES = 8 * 1024 * 1024; // 8 MiB

/** Check if a selector string looks like a range selector. */
function isRangeSelector(value: string): boolean {
	return /^(raw|conflicts|-?\d+(?:[-+]\d+)?)$/i.test(value);
}

/** Parse `<id>` or `<id>:<selector>` from the URL host+pathname. */
function parseIdAndSelector(url: InternalUrl): { id: string; selector: string | null } {
	const full = url.pathname === "/" ? url.host : `${url.host}${url.pathname}`;
	if (!full) return { id: "", selector: null };
	const colonIdx = full.indexOf(":");
	const id = colonIdx < 0 ? full : full.slice(0, colonIdx);
	if (colonIdx < 0) return { id, selector: null };
	const suffix = full.slice(colonIdx + 1);
	return { id, selector: isRangeSelector(suffix) ? suffix : null };
}

/** Extract a range from a selector string. */
function parseRange(selector: string): { offset?: number; limit?: number } | null {
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

export class LocalProtocolHandler implements ProtocolHandler {
	readonly scheme = "local";
	readonly immutable = false;

	/** Resolve `<cwd>/.logician/artifacts` plus the target path for a URL, enforcing the sandbox both hops. */
	#resolveTargetPath(
		url: InternalUrl,
		cwd: string,
		allowedPaths?: string[],
		allowAllPaths?: boolean,
	): { artifactDir: string; resolved: string | null } {
		const artifactDir = path.join(cwd, ".logician", "artifacts");
		ensureInsideCwd(cwd, artifactDir, allowedPaths, allowAllPaths);

		const hostname = url.host;
		if (!hostname) return { artifactDir, resolved: null };

		const filePath = path.join(artifactDir, hostname, url.pathname.slice(1));
		const resolved = path.resolve(filePath);
		ensureInsideCwd(artifactDir, resolved);
		return { artifactDir, resolved };
	}

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const cwd = context?.cwd ?? process.cwd();
		const { artifactDir, resolved } = this.#resolveTargetPath(
			url,
			cwd,
			context?.allowedPaths,
			context?.allowAllPaths,
		);

		// '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local' — list available artifacts
		if (!resolved || (url.host === "local" && url.pathname === "/")) {
			return this.listArtifacts(artifactDir, url);
		}

		// '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local'<numeric-id> — read artifact by ID (supports selectors)
		const { id, selector } = parseIdAndSelector(url);
		if (id && /^\d+$/.test(id)) {
			return this.#resolveArtifact(url, id, selector, cwd, context);
		}

		// '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local'<path> — read file under artifacts directory
		try {
			const stat = await fs.stat(resolved);
			if (stat.isDirectory()) {
				return this.listDirectory(resolved, url);
			}
			if (stat.isFile()) {
				if (context?.pathOnly) {
					return {
						url: url.href,
						content: "",
						contentType: detectContentType(resolved),
						size: stat.size,
						sourcePath: resolved,
					};
				}
				if (stat.size > LOCAL_TEXT_RESOURCE_MAX_BYTES) {
					throw new Error(
						`${url.href} is ${formatSize(stat.size)}, exceeding the ${formatSize(LOCAL_TEXT_RESOURCE_MAX_BYTES)} limit for '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local' text resources. Use bash tools to inspect it, or the backing path directly.`,
					);
				}
				const buffer = await fs.readFile(resolved);
				if (buffer.subarray(0, LOCAL_TEXT_SNIFF_BYTES).includes(0)) {
					throw new Error(
						`${url.href} appears to be a binary file (${formatSize(stat.size)}). Use bash tools (file, xxd, strings) to inspect it.`,
					);
				}
				return {
					url: url.href,
					content: buffer.toString("utf-8"),
					contentType: detectContentType(resolved),
					size: stat.size,
					sourcePath: resolved,
				};
			}
		} catch (err) {
			if (
				err instanceof Error &&
				"code" in err &&
				(err as { code: string }).code === "ENOENT"
			) {
				throw new Error(`No such local artifact: ${url.href}`);
			}
			throw err;
		}

		throw new Error(`Unknown artifact type: ${resolved}`);
	}

	async write(
		url: InternalUrl,
		content: string,
		context?: WriteContext,
	): Promise<void> {
		const cwd = context?.cwd ?? process.cwd();
		const { resolved } = this.#resolveTargetPath(
			url,
			cwd,
			context?.allowedPaths,
			context?.allowAllPaths,
		);
		if (!resolved) {
			throw new Error(
				`'/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local' write requires a target: '/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local'<name> (got ${url.href})`,
			);
		}

		const stat = await fs.stat(resolved).catch(() => null);
		if (stat?.isDirectory()) {
			throw new Error(`Cannot write: ${url.href} is a directory`);
		}

		await fs.mkdir(path.dirname(resolved), { recursive: true });
		await atomicWriteFile(resolved, content);
	}

	async complete(
		query: string,
		context?: ResolveContext,
	): Promise<Array<{ value: string; description?: string }>> {
		// If query looks like a number, suggest artifact IDs
		if (/^\d*$/.test(query)) {
			const registry = ArtifactRegistry.instance();
			const ids = await registry.listIds();
			return ids
				.filter(id => id.startsWith(query) || query === "")
				.map(id => ({ value: id, description: `Artifact ${id}` }));
		}
		return [];
	}

	async #resolveArtifact(
		url: InternalUrl,
		id: string,
		selector: string | null,
		cwd: string,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const registry = ArtifactRegistry.instance();

		if (selector) {
			return this.#resolveArtifactWithSelector(id, url, selector, context);
		}

		// No selector — return full content
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
				size: 0,
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
				`'/home/seman/.omp/agent/sessions/-logician/2026-09-14T15-47-18-748Z_01a0a099-f25c-7093-9375-1f584092948f/local'${id} exceeds ${MAX_ARTIFACT_BYTES / 1024 / 1024} MiB limit; use \`read\` with line selectors or the backing path for large files`,
			);
		}

		return {
			url: url.href,
			content,
			contentType: "text/plain",
			size: Buffer.byteLength(content, "utf-8"),
		};
	}

	async #resolveArtifactWithSelector(
		id: string,
		url: InternalUrl,
		selector: string,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const registry = ArtifactRegistry.instance();
		const content = await registry.read(id);
		if (content === null) {
			throw new Error(`Unknown artifact: ${id}`);
		}

		const range = parseRange(selector);
		if (range) {
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
		return this.#resolveArtifact(url, id, null, context?.cwd, context);
	}

	private async listArtifacts(
		artifactDir: string,
		_url: InternalUrl,
	): Promise<InternalResource> {
		const entries: LocalEntry[] = [];
		try {
			const items = await fs.readdir(artifactDir, { withFileTypes: true });
			for (const item of items) {
				if (item.isDirectory()) {
					entries.push({
						path: item.name,
						name: item.name,
						type: "directory",
					});
				}
			}
		} catch {
			// Directory doesn't exist or is empty
		}

		const lines = entries.map(
			e => `  ${e.type === "directory" ? "📁" : "📄"} ${e.name}`,
		);
		return {
			url: _url.href,
			content:
				entries.length > 0
					? `# Local Artifacts\n\n${lines.join("\n")}`
					: "# Local Artifacts\n\nNo artifacts found. Artifacts are stored in .logician/artifacts/",
			contentType: "text/markdown",
		};
	}

	private async listDirectory(
		dir: string,
		_url: InternalUrl,
	): Promise<InternalResource> {
		const entries = await fs.readdir(dir, { withFileTypes: true });
		const lines = entries.map(
			e => `  ${e.isDirectory() ? "📁" : "📄"} ${e.name}`,
		);
		return {
			url: _url.href,
			content: `# ${path.basename(dir)}\n\n${lines.join("\n")}`,
			contentType: "text/markdown",
		};
	}
}

function detectContentType(
	filePath: string,
): "text/plain" | "text/markdown" | "application/json" {
	if (filePath.endsWith(".md")) return "text/markdown";
	if (filePath.endsWith(".json")) return "application/json";
	return "text/plain";
}
