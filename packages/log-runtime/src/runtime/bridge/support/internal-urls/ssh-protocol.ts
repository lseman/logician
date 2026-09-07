// ── ssh:// protocol handler ────────────────────────────────────────────────────
// Reads files and lists directories on remote hosts via ssh/scp.
// URL forms:
//   ssh://                              — lists configured/known hosts
//   ssh://<host>/path/to/file           — reads a file on remote host
//   ssh://<host>/path/to/dir/           — lists directory contents
//
// Host resolution:
//   1. ~/.logician/ssh.json (if present) — named host entries
//   2. ~/.ssh/config — OpenSSH config aliases
//   3. Any hostname resolvable by system ssh/scp
//
// Auth: key/agent-based only (no password support).
// Max file size: 1 MiB for inline reading.

import { spawn } from "node:child_process";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

import type { InternalResource, InternalUrl, ProtocolHandler, ResolveContext, UrlCompletion } from "./types";

const SSH_TEXT_MAX_BYTES = 1 * 1024 * 1024;
const SSH_CONFIG_PATH = path.join(os.homedir(), ".logician", "ssh.json");

/** SSH host entry from ~/.logician/ssh.json */
interface SshHostEntry {
	name: string;
	host: string;
	port?: number;
	username?: string;
	keyPath?: string;
	description?: string;
}

/** Decoded SSH connection target. */
interface SshTarget {
	name: string;
	host: string;
	username?: string;
	port?: number;
	keyPath?: string;
}

/** One-line address for a host, e.g. `deploy@10.0.0.1:2222`. */
function hostAddress(host: SshHostEntry): string {
	return `${host.username ? `${host.username}@` : ""}${host.host}${host.port ? `:${host.port}` : ""}`;
}

/** Render the configured-host index for a bare `ssh://` read. */
function formatHostIndex(hosts: readonly SshHostEntry[]): string {
	if (hosts.length === 0) {
		return "# SSH hosts\n\nNo SSH hosts are configured. Add hosts to `~/.logician/ssh.json`, or use `ssh://<host>/<path>` with any destination your OpenSSH client can resolve (e.g. a `~/.ssh/config` alias).\n";
	}
	const lines = hosts.map(host => {
		const addr = hostAddress(host);
		const suffix = addr === host.name ? "" : ` — \`${addr}\``;
		const desc = host.description ? ` (${host.description})` : "";
		return `- ${host.name}${suffix}${desc}`;
	});
	return `# SSH hosts\n\n${hosts.length} configured host${hosts.length === 1 ? "" : "s"}:\n\n${lines.join("\n")}\n`;
}

/** Load configured hosts from ~/.logician/ssh.json. */
async function loadConfiguredHosts(): Promise<SshHostEntry[]> {
	try {
		const raw = await fs.readFile(SSH_CONFIG_PATH, "utf-8");
		const parsed = JSON.parse(raw);
		if (!Array.isArray(parsed)) return [];
		return parsed.filter((h: unknown) => typeof h === "object" && h !== null && "name" in h && "host" in h) as SshHostEntry[];
	} catch {
		// No config file or parse error — fall back to ~.ssh/config resolution
		return [];
	}
}

/** Execute a command via spawn, returning { stdout, stderr, code }. */
function execSsh(
	command: string,
	args: string[],
	timeoutMs: number = 30_000,
): Promise<{ stdout: string; stderr: string; code: number | null }> {
	const { promise, resolve, reject } = Promise.withResolvers<{ stdout: string; stderr: string; code: number | null }>();
	const proc = spawn(command, args, {
		timeout: timeoutMs,
		stdio: ["pipe", "pipe", "pipe"],
		shell: false,
	});
	const stdoutChunks: Buffer[] = [];
	const stderrChunks: Buffer[] = [];
	proc.stdout?.on("data", (chunk: Buffer) => stdoutChunks.push(chunk));
	proc.stderr?.on("data", (chunk: Buffer) => stderrChunks.push(chunk));
	proc.on("error", (err: NodeJS.ErrnoException) => {
		if (err.code === "ETIMEDOUT" || err.code === "SIGTERM") {
			resolve({ stdout: Buffer.concat(stdoutChunks).toString("utf-8"), stderr: Buffer.concat(stderrChunks).toString("utf-8"), code: null });
		} else {
			reject(err);
		}
	});
	proc.on("close", code => {
		resolve({ stdout: Buffer.concat(stdoutChunks).toString("utf-8"), stderr: Buffer.concat(stderrChunks).toString("utf-8"), code });
	});
	return promise;
}

/** Build scp arguments for a target. */
function buildScpArgs(target: SshTarget, remotePath: string, args: string[]): string[] {
	const { username, port, host, keyPath } = target;
	const scpArgs: string[] = [...args];
	if (keyPath) scpArgs.push("-i", keyPath);
	if (port) scpArgs.push("-P", String(port));
	const remote = username ? `${username}@${host}` : host;
	scpArgs.push(`${remote}:${remotePath}`);
	return scpArgs;
}
/** Build ssh arguments for a target. */
function buildSshArgs(target: SshTarget, command: string): string[] {
	const { username, port, host, keyPath } = target;
	const sshArgs: string[] = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"];
	if (username) sshArgs.push("-l", username);
	if (port) sshArgs.push("-p", String(port));
	if (keyPath) sshArgs.push("-i", keyPath);
	sshArgs.push(host, command);
	return sshArgs;
}

/** Decode the remote path from URL. */
function remotePathFromUrl(url: InternalUrl): string {
	const raw = url.pathname;
	const pathPart = raw.startsWith("/") ? raw.slice(1) : raw;
	if (!pathPart || pathPart === ".") return ".";
	return pathPart;
}

/** Resolve URL authority to an SSH target. */
async function resolveTarget(url: InternalUrl, _cwd?: string): Promise<SshTarget> {
	const bareHost = url.hostname;
	const rawAuthority = url.rawHost || bareHost;
	const username = url.username || undefined;
	const port = url.port ? Number(url.port) : undefined;

	if (!bareHost && !rawAuthority) {
		throw new Error("ssh:// requires a host: ssh://<host>/<path>");
	}

	if (url.password) {
		throw new Error("ssh://: password authentication is not supported — use key/agent auth");
	}

	// Parse user@host:port or bare host
	const isIpv6Literal = bareHost?.startsWith("[") && bareHost?.endsWith("]");
	const sshHost = isIpv6Literal ? bareHost?.slice(1, -1) : bareHost;

	// Try configured hosts first
	const configured = await loadConfiguredHosts();
	const match = configured.find(h => h.name === (url.rawHost || url.hostname)) ?? configured.find(h => h.name === bareHost);
	if (match) {
		return {
			name: match.name,
			host: match.host,
			username: username || match.username,
			port: port || match.port,
			keyPath: match.keyPath,
		};
	}

	// Opaque OpenSSH destination
	return {
		name: rawAuthority,
		host: isIpv6Literal ? (sshHost ?? rawAuthority) : rawAuthority,
		username,
		port,
	};
}

/** Format a remote directory listing. */
function formatDirListing(output: string): string {
	const trimmed = output.trim();
	if (!trimmed) return "(empty directory)\n";
	const lines = trimmed.split("\n").map(line => `  ${line}`).join("\n");
	return `${lines}\n`;
}

/** Detect content type from remote file path. */
function contentTypeFor(remotePath: string): InternalResource["contentType"] {
	if (remotePath.endsWith(".md")) return "text/markdown";
	if (remotePath.endsWith(".json")) return "application/json";
	if (remotePath.endsWith(".ts") || remotePath.endsWith(".js") || remotePath.endsWith(".tsx") || remotePath.endsWith(".jsx")) return "text/plain";
	if (remotePath.endsWith(".yaml") || remotePath.endsWith(".yml") || remotePath.endsWith(".toml") || remotePath.endsWith(".ini") || remotePath.endsWith(".cfg")) return "text/plain";
	if (remotePath.endsWith(".log") || remotePath.endsWith(".txt")) return "text/plain";
	return "text/plain";
}

/** Check if output is likely binary (contains NUL bytes). */
function isLikelyBinary(buffer: Buffer): boolean {
	return buffer.includes(Buffer.from([0]));
}

export class SshProtocolHandler implements ProtocolHandler {
	readonly scheme = "ssh";
	readonly immutable = false;

	async resolve(url: InternalUrl, context?: ResolveContext): Promise<InternalResource> {
		// Bare ssh:// with no host — list configured hosts
		if (!(url.rawHost || url.hostname)) {
			const rawPath = url.pathname;
			if (rawPath && rawPath !== "/") {
				throw new Error(`ssh:// requires a host: ssh://<host>${rawPath}`);
			}
			const hosts = await loadConfiguredHosts();
			const content = formatHostIndex(hosts);
			return {
				url: url.href,
				content,
				contentType: "text/markdown",
				size: Buffer.byteLength(content, "utf-8"),
			};
		}

		const target = await resolveTarget(url, context?.cwd);
		const remotePath = remotePathFromUrl(url);
		const isDirectory = remotePath.endsWith("/");

		if (isDirectory) {
			// Directory listing via ssh + ls
			const { stdout, stderr, code } = await execSsh("ssh", buildSshArgs(target, `ls -1A "${remotePath}"`));
			if (code !== 0) {
				throw new Error(`ssh://: ${stderr || `ls failed (exit ${code})`}`);
			}
			const content = formatDirListing(stdout);
			return {
				url: url.href,
				content,
				contentType: "text/plain",
				size: Buffer.byteLength(content, "utf-8"),
				isDirectory: true,
			};
		}

		// File read via scp
		const { stdout, stderr, code } = await execSsh("scp", buildScpArgs(target, remotePath, ["-q", "-C"]));
		if (code !== 0) {
			throw new Error(`ssh://: ${stderr || `scp failed (exit ${code})`}`);
		}
		if (stdout.length === 0) {
			throw new Error(`ssh://: empty file — ${remotePath}`);
		}

		// Check for binary content
		const buffer = Buffer.from(stdout, "binary");
		if (isLikelyBinary(buffer)) {
			throw new Error(`ssh://: ${remotePath} appears to be binary; ssh:// supports UTF-8 text only`);
		}
		if (buffer.length > SSH_TEXT_MAX_BYTES) {
			throw new Error(`ssh://: ${remotePath} exceeds ${SSH_TEXT_MAX_BYTES / 1024 / 1024} MiB limit`);
		}

		const content = buffer.toString("utf-8");
		return {
			url: url.href,
			content,
			contentType: contentTypeFor(remotePath),
			size: buffer.length,
		};
	}

	async complete(query?: string, _context?: ResolveContext): Promise<UrlCompletion[]> {
		const hosts = await loadConfiguredHosts();
		if (!query) {
			return hosts.map(host => ({
				value: encodeURIComponent(host.name),
				description: `${host.name} — ${hostAddress(host)}${host.description ? ` (${host.description})` : ""}`,
			}));
		}
		const q = query.toLowerCase();
		return hosts
			.filter(h => h.name.toLowerCase().includes(q))
			.map(host => ({
				value: encodeURIComponent(host.name),
				description: `${host.name} — ${hostAddress(host)}`,
			}));
	}
}
