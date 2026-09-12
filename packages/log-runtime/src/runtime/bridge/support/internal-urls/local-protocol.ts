// ── local:// protocol handler ────────────────────────────────────────────────
// Resolves session-local artifacts (JSONL journals, session meta, etc.).
// URL forms:
//   local://<path> — reads a file under the session artifacts directory

import * as fs from "node:fs/promises";
import * as path from "node:path";

import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
} from "./types";

/** Path to a local session artifact. */
interface LocalEntry {
	path: string;
	name: string;
	type: "file" | "directory";
}

export class LocalProtocolHandler implements ProtocolHandler {
	readonly scheme = "local";

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const cwd = context?.cwd ?? process.cwd();
		const artifactDir = path.join(cwd, ".logician", "artifacts");

		const hostname = url.rawHost || url.hostname;
		const pathname = url.pathname;

		// local:// — list available artifacts
		if (!hostname || (hostname === "local" && pathname === "/")) {
			return this.listArtifacts(artifactDir, url);
		}

		// local://<name> or local://<dir>/<path>
		const filePath = path.join(artifactDir, hostname, pathname.slice(1));
		const resolved = path.resolve(filePath);

		// Security: must be within artifactDir
		if (!resolved.startsWith(artifactDir)) {
			throw new Error(`Path traversal blocked: local://${hostname}${pathname}`);
		}

		try {
			const stat = await fs.stat(resolved);
			if (stat.isDirectory()) {
				return this.listDirectory(resolved, url);
			}
			if (stat.isFile()) {
				const content = await fs.readFile(resolved, "utf-8");
				return {
					url: url.href,
					content,
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
