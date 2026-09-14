// ── local:// protocol handler ────────────────────────────────────────────────
// Resolves session-local artifacts (JSONL journals, session meta, etc.).
// URL forms:
//   local://<path> — reads a file under the session artifacts directory

import * as fs from "node:fs/promises";
import * as path from "node:path";
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

		// local:// — list available artifacts
		if (!resolved || (url.host === "local" && url.pathname === "/")) {
			return this.listArtifacts(artifactDir, url);
		}

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
						`${url.href} is ${formatSize(stat.size)}, exceeding the ${formatSize(LOCAL_TEXT_RESOURCE_MAX_BYTES)} limit for local:// text resources. Use bash tools to inspect it, or the backing path directly.`,
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
				`local:// write requires a target: local://<name> (got ${url.href})`,
			);
		}

		const stat = await fs.stat(resolved).catch(() => null);
		if (stat?.isDirectory()) {
			throw new Error(`Cannot write: ${url.href} is a directory`);
		}

		await fs.mkdir(path.dirname(resolved), { recursive: true });
		await atomicWriteFile(resolved, content);
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
