// ── log:// protocol handler ────────────────────────────────────────────────────
// Resolves documentation files from the workspace docs/ directory.
// URL forms:
//   log://               — lists available docs
//   log://<path>         — reads a doc file (relative to docs/)
//   log://<dir>/         — lists directory contents
//
// Example: log://guides/agent-evaluation, log://architecture/overview, log://

import * as fs from "node:fs/promises";
import * as path from "node:path";
import { ensureInsideCwd } from "../shared/path-utils.ts";

import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
	UrlCompletion,
} from "./types";

const DOCS_DIR_NAME = "docs";
const MAX_DOCS_FILE_BYTES = 512 * 1024;

/** Detect content type from file extension. */
function detectContentType(filePath: string): InternalResource["contentType"] {
	if (filePath.endsWith(".md")) return "text/markdown";
	if (filePath.endsWith(".json")) return "application/json";
	return "text/plain";
}

/** Resolve the docs directory under cwd, or throw with guidance. */
function resolveDocsDir(cwd: string): string {
	const docsDir = path.join(cwd, DOCS_DIR_NAME);
	return docsDir;
}

/** Format a directory listing as markdown. */
function formatDirectoryListing(
	entries: Array<{ name: string; isDir: boolean }>,
	url: InternalUrl,
): string {
	if (entries.length === 0)
		return `# ${url.pathname === "/" ? "Docs" : url.pathname.slice(1)}\n\n(empty directory)\n`;
	const lines = entries
		.map(e => {
			const prefix = e.isDir ? "📁" : "📄";
			return `${prefix} ${e.name}${e.isDir ? "/" : ""}`;
		})
		.join("\n");
	return `# ${url.pathname === "/" ? "Docs" : url.pathname.slice(1)}\n\n${lines}\n`;
}

export class LogProtocolHandler implements ProtocolHandler {
	readonly scheme = "log";
	readonly immutable = true;

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const cwd = context?.cwd ?? process.cwd();
		const docsDir = resolveDocsDir(cwd);
		const hostname = url.host;
		const pathname = url.pathname;

		// Bare log:// — show root index
		if (!hostname || (hostname === "log" && pathname === "/")) {
			return {
				url: url.href,
				content: await this.#formatIndex(docsDir),
				contentType: "text/markdown",
			};
		}

		// Build the resolved path within docs/
		let resolved: string;
		if (hostname === "log" || !hostname) {
			resolved = path.join(docsDir, pathname.slice(1));
		} else {
			resolved = path.join(docsDir, hostname, pathname.slice(1));
		}

		const resolvedAbs = path.resolve(resolved);

		// Security: must be within docsDir
		ensureInsideCwd(
			cwd,
			docsDir,
			context?.allowedPaths,
			context?.allowAllPaths,
		);
		ensureInsideCwd(docsDir, resolvedAbs);

		try {
			const stat = await fs.stat(resolvedAbs);
			if (stat.isDirectory()) {
				return this.#listDirectory(resolvedAbs, url);
			}
			return this.#readFile(resolvedAbs, url);
		} catch (err) {
			if (
				err instanceof Error &&
				"code" in err &&
				(err as { code: string }).code === "ENOENT"
			) {
				// Extension-less docs convention: log://guides/foo → guides/foo.md
				const mdFallback = `${resolvedAbs}.md`;
				try {
					const mdStat = await fs.stat(mdFallback);
					if (mdStat.isFile()) {
						return this.#readFile(mdFallback, url);
					}
				} catch {
					// no .md fallback — fall through to not-found below
				}
				// Provide helpful listing of available docs
				const available = await this.#listAvailable(docsDir, url);
				throw new Error(`Not found: ${url.href}\n${available}`);
			}
			throw err;
		}
	}

	async complete(
		query: string,
		context?: ResolveContext,
	): Promise<UrlCompletion[]> {
		const cwd = context?.cwd ?? process.cwd();
		const docsDir = resolveDocsDir(cwd);

		// If query is empty or ends with /, list top-level entries
		if (!query || query.endsWith("/")) {
			const entries = await this.#listEntries(docsDir);
			return entries.map(e => ({
				value: `${e.name}${e.isDir ? "/" : ""}`,
				description: e.isDir ? "Directory" : "File",
			}));
		}

		// Otherwise search for matching files
		const partial = query.toLowerCase();
		const entries = await this.#listEntries(docsDir);
		const matches: UrlCompletion[] = [];
		for (const entry of entries) {
			if (entry.name.toLowerCase().includes(partial)) {
				matches.push({
					value: `${entry.name}${entry.isDir ? "/" : ""}`,
					description: entry.isDir ? "Directory" : "File",
				});
			}
		}
		return matches;
	}

	/** List a directory within docs/. */
	async #listDirectory(
		dir: string,
		_url: InternalUrl,
	): Promise<InternalResource> {
		const entries = await this.#listEntries(dir);
		const content = formatDirectoryListing(entries, _url);
		return {
			url: _url.href,
			content,
			contentType: "text/plain",
			size: Buffer.byteLength(content, "utf-8"),
		};
	}

	/** Read a file within docs/. */
	async #readFile(
		filePath: string,
		_url: InternalUrl,
	): Promise<InternalResource> {
		const stat = await fs.stat(filePath);
		if (stat.size > MAX_DOCS_FILE_BYTES) {
			throw new Error(
				`log://: ${filePath} exceeds ${MAX_DOCS_FILE_BYTES / 1024 / 1024} MiB limit; use \`read\` with line selectors for large files`,
			);
		}
		const content = await fs.readFile(filePath, "utf-8");
		return {
			url: _url.href,
			content,
			contentType: detectContentType(filePath),
			size: stat.size,
			sourcePath: filePath,
		};
	}

	/** List available docs for error messages. */
	async #listAvailable(docsDir: string, _url: InternalUrl): Promise<string> {
		const entries = await this.#listEntries(docsDir);
		if (entries.length === 0) return "No documentation files found.";
		const lines = entries.map(e => `  - ${e.name}${e.isDir ? "/" : ""}`);
		return `\nAvailable under ${DOCS_DIR_NAME}/:\n${lines.join("\n")}`;
	}

	/** List directory entries (dirs and files) with isDir flag. */
	async #listEntries(
		dir: string,
	): Promise<Array<{ name: string; isDir: boolean }>> {
		try {
			const items = await fs.readdir(dir, { withFileTypes: true });
			return items
				.filter(e => !e.name.startsWith("."))
				.map(e => ({ name: e.name, isDir: e.isDirectory() }))
				.sort((a, b) => {
					// Dirs first, then alphabetical
					if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
					return a.name.localeCompare(b.name);
				});
		} catch {
			return [];
		}
	}

	/** Format the root index from the actual docs/ contents. */
	async #formatIndex(docsDir: string): Promise<string> {
		try {
			await fs.access(docsDir);
		} catch {
			return `# Logician Docs\n\nNo \`docs/\` directory found under \`${docsDir}\`.\n`;
		}
		const entries = await this.#listEntries(docsDir);
		if (entries.length === 0) {
			return `# Logician Docs\n\nThe \`docs/\` directory is empty.\n`;
		}
		const lines = entries
			.map(e => `- \`${e.name}${e.isDir ? "/" : ""}\``)
			.join("\n");
		return `# Logician Docs\n\nBrowse available documentation with \`log://<path>\`.\n\nTop-level entries under docs/:\n\n${lines}\n`;
	}
}
