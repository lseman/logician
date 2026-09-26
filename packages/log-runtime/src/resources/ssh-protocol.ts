// ── ssh:// protocol handler ────────────────────────────────────────────────────
// Reads files, lists directories, and writes files on remote hosts via ssh/scp.
// URL forms:
//   ssh://                              — lists configured/known hosts
//   ssh://<host>/path/to/file           — reads a file on remote host
//   ssh://<host>/path/to/dir/           — lists directory contents
//
// Write: byte-exact remote file write (staged through a temp in the destination
// directory; in-place overwrite preserves inode and permission bits, new paths
// commit by atomic rename). Directories, FIFOs, sockets, and devices are refused.
//
// Host resolution:
//   1. ~/.logician/ssh.json (if present) — named host entries
//   2. ~/.ssh/config — OpenSSH config aliases
//   3. Any hostname resolvable by system ssh/scp
//
// Auth: key/agent-based only (no password support).
// Max file size: 1 MiB for inline reading.

import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
	UrlCompletion,
	WriteContext,
} from "./types";

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
		return parsed.filter(
			(h: unknown) =>
				typeof h === "object" && h !== null && "name" in h && "host" in h,
		) as SshHostEntry[];
	} catch {
		// No config file or parse error — fall back to ~.ssh/config resolution
		return [];
	}
}

/**
 * Execute a command via spawn, returning { stdout, stderr, code }.
 * When `stdin` is given it is piped to the child (used by the staged write);
 * the child's stdin is always closed so remote commands that read stdin
 * (e.g. `cat`) cannot hang on an open pipe.
 */
function execSsh(
	command: string,
	args: string[],
	timeoutMs: number = 30_000,
	stdin?: Buffer,
	signal?: AbortSignal,
): Promise<{ stdout: string; stderr: string; code: number | null }> {
	const { promise, resolve, reject } = Promise.withResolvers<{
		stdout: string;
		stderr: string;
		code: number | null;
	}>();
	const proc = spawn(command, args, {
		timeout: timeoutMs,
		stdio: ["pipe", "pipe", "pipe"],
		shell: false,
		...(signal ? { signal } : {}),
	});
	const stdoutChunks: Buffer[] = [];
	const stderrChunks: Buffer[] = [];
	proc.stdout?.on("data", (chunk: Buffer) => stdoutChunks.push(chunk));
	proc.stderr?.on("data", (chunk: Buffer) => stderrChunks.push(chunk));
	// A child killed by the timeout or an aborted signal exits with code null
	// and no output of its own; give the caller a readable reason.
	const noteKill = () => {
		if (stdoutChunks.length === 0 && stderrChunks.length === 0) {
			stderrChunks.push(Buffer.from("command timed out or was aborted"));
		}
	};
	proc.on("error", (err: NodeJS.ErrnoException) => {
		if (
			err.code === "ETIMEDOUT" ||
			err.code === "SIGTERM" ||
			err.code === "ABORT"
		) {
			noteKill();
			resolve({
				stdout: Buffer.concat(stdoutChunks).toString("utf-8"),
				stderr: Buffer.concat(stderrChunks).toString("utf-8"),
				code: null,
			});
		} else {
			reject(err);
		}
	});
	proc.on("close", code => {
		if (code === null) noteKill();
		resolve({
			stdout: Buffer.concat(stdoutChunks).toString("utf-8"),
			stderr: Buffer.concat(stderrChunks).toString("utf-8"),
			code,
		});
	});
	if (stdin) {
		// EPIPE is expected when the remote command exits before reading all
		// input (e.g. it refuses the destination); the exit status carries
		// the real outcome, so suppress the error and let close resolve it.
		proc.stdin?.on("error", () => {});
		proc.stdin?.write(stdin);
	}
	proc.stdin?.end();
	return promise;
}

/** Build scp arguments for a target. */
function buildScpArgs(
	target: SshTarget,
	remotePath: string,
	args: string[],
): string[] {
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
	const sshArgs: string[] = [
		"-o",
		"StrictHostKeyChecking=accept-new",
		"-o",
		"BatchMode=yes",
	];
	if (username) sshArgs.push("-l", username);
	if (port) sshArgs.push("-p", String(port));
	if (keyPath) sshArgs.push("-i", keyPath);
	const destination =
		host.startsWith("[") && host.endsWith("]") ? host.slice(1, -1) : host;
	sshArgs.push(destination, command);
	return sshArgs;
}

/** Decode the remote path from URL. */
function remotePathFromUrl(url: InternalUrl): string {
	const raw = url.pathname;
	const pathPart = raw.startsWith("/") ? raw.slice(1) : raw;
	if (!pathPart || pathPart === ".") return ".";
	return pathPart;
}
/** Wrap a POSIX path in single quotes so the remote shell treats it as one word. */
function quotePosixPath(p: string): string {
	return `'${p.replace(/'/g, `'\\''`)}'`;
}

/**
 * Write bytes to a remote file byte-exact. Stdin is always staged first into
 * a uniquely named temp in the destination directory (so the remote never
 * blocks on an unread pipe and a dropped connection lands in the temp, never
 * the destination). The destination then dictates the commit:
 *  - a directory — or a symlink to one, since the `-d` test follows links — is
 *    refused (a plain `mv tmp dir` would move the temp INTO it).
 *  - an existing non-symlink regular file is rewritten IN PLACE from the
 *    staged temp, preserving its inode and therefore its ordinary permission
 *    bits (a `0600` secret stays `0600` on overwrite), ACLs, xattrs, and
 *    hardlinks. The setuid/setgid bits may be cleared by the write (per POSIX).
 *  - an existing special file (FIFO/socket/device) is refused, not replaced.
 *  - anything else (a new path, a symlink to a non-directory, a dangling
 *    symlink) is committed with an atomic rename, which REPLACES a symlink
 *    with a regular file rather than writing through it.
 * The EXIT trap removes the staged temp on every exit path.
 */
async function writeRemoteFile(
	target: SshTarget,
	remotePath: string,
	content: Buffer,
	signal?: AbortSignal,
): Promise<void> {
	const dest = quotePosixPath(remotePath);
	const tmp = quotePosixPath(`${remotePath}.logician-tmp.${randomUUID()}`);
	const command =
		`t=${tmp}; trap 'rm -f -- "$t"' 0; ` +
		`mkdir -p -- "$(dirname "$t")" && ` +
		`cat > "$t" && { ` +
		`if [ -d ${dest} ]; then echo 'ssh://: destination is a directory' >&2; exit 1; ` +
		`elif [ -f ${dest} ] && [ ! -L ${dest} ]; then cat "$t" > ${dest} || exit 1; ` +
		`elif [ -e ${dest} ] && [ ! -L ${dest} ]; then echo 'ssh://: destination is a special file (not a regular file)' >&2; exit 1; ` +
		`else mv "$t" ${dest}; fi; ` +
		`}`;
	const { stderr, code } = await execSsh(
		"ssh",
		buildSshArgs(target, command),
		30_000,
		content,
		signal,
	);
	if (code !== 0) {
		throw new Error(
			`ssh://: write to ${remotePath} failed: ${stderr.trim() || `exit ${code}`}`,
		);
	}
}

/** SSH alone interprets user, host and port in a resource-link authority. */
function parseSshAuthority(authority: string): SshTarget {
	const match = authority.match(
		/^(?:([^@\s]+)@)?(\[[^\]]+\]|[^:\s@[\]]+)(?::(\d+))?$/,
	);
	if (!match) {
		throw new Error(
			"Invalid SSH authority: expected [user@]host[:port], with IPv6 hosts in brackets",
		);
	}
	const [, username, host, portText] = match;
	if (username?.includes(":")) {
		throw new Error(
			"ssh://: password authentication is not supported — use key/agent auth",
		);
	}
	const port = portText === undefined ? undefined : Number(portText);
	if (
		port !== undefined &&
		(!Number.isInteger(port) || port < 1 || port > 65535)
	) {
		throw new Error("ssh:// port must be an integer from 1 to 65535");
	}
	return { name: authority, host, username, port };
}

/** Resolve the parsed destination against configured hosts. */
async function resolveTarget(url: InternalUrl): Promise<SshTarget> {
	const target = parseSshAuthority(url.host);

	// Try configured hosts first
	const configured = await loadConfiguredHosts();
	const match =
		configured.find(h => h.name === url.host) ??
		configured.find(h => h.name === target.host);
	if (match) {
		return {
			name: match.name,
			host: match.host,
			username: target.username ?? match.username,
			port: target.port ?? match.port,
			keyPath: match.keyPath,
		};
	}

	return target;
}

/** Format a remote directory listing. */
function formatDirListing(output: string): string {
	const trimmed = output.trim();
	if (!trimmed) return "(empty directory)\n";
	const lines = trimmed
		.split("\n")
		.map(line => `  ${line}`)
		.join("\n");
	return `${lines}\n`;
}

/** Detect content type from remote file path. */
function contentTypeFor(remotePath: string): InternalResource["contentType"] {
	if (remotePath.endsWith(".md")) return "text/markdown";
	if (remotePath.endsWith(".json")) return "application/json";
	if (
		remotePath.endsWith(".ts") ||
		remotePath.endsWith(".js") ||
		remotePath.endsWith(".tsx") ||
		remotePath.endsWith(".jsx")
	)
		return "text/plain";
	if (
		remotePath.endsWith(".yaml") ||
		remotePath.endsWith(".yml") ||
		remotePath.endsWith(".toml") ||
		remotePath.endsWith(".ini") ||
		remotePath.endsWith(".cfg")
	)
		return "text/plain";
	if (remotePath.endsWith(".log") || remotePath.endsWith(".txt"))
		return "text/plain";
	return "text/plain";
}

/** Check if output is likely binary (contains NUL bytes). */
function isLikelyBinary(buffer: Buffer): boolean {
	return buffer.includes(Buffer.from([0]));
}

export class SshProtocolHandler implements ProtocolHandler {
	readonly scheme = "ssh";
	readonly immutable = false;

	async resolve(
		url: InternalUrl,
		_context?: ResolveContext,
	): Promise<InternalResource> {
		// Bare ssh:// with no host — list configured hosts
		if (!url.host) {
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

		const target = await resolveTarget(url);
		const remotePath = remotePathFromUrl(url);
		const isDirectory = remotePath.endsWith("/");

		if (isDirectory) {
			// Directory listing via ssh + ls
			const { stdout, stderr, code } = await execSsh(
				"ssh",
				buildSshArgs(target, `ls -1A "${remotePath}"`),
			);
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

		// Stat before transferring: fail fast when the target is missing or not
		// a regular file (FIFOs and devices would hang the transfer) and skip
		// oversized files without downloading them.
		const {
			stdout: statOut,
			stderr: statErr,
			code: statCode,
		} = await execSsh(
			"ssh",
			buildSshArgs(
				target,
				`test -f "${remotePath}" && wc -c < "${remotePath}"`,
			),
		);
		if (statCode !== 0) {
			throw new Error(
				`ssh://: not a readable regular file: ${remotePath}${statErr ? ` — ${statErr.trim()}` : ""}`,
			);
		}
		const remoteSize = Number.parseInt(statOut.trim(), 10);
		if (!Number.isFinite(remoteSize) || remoteSize < 0) {
			throw new Error(`ssh://: could not determine size of ${remotePath}`);
		}
		if (remoteSize > SSH_TEXT_MAX_BYTES) {
			throw new Error(
				`ssh://: ${remotePath} is ${remoteSize} bytes, exceeds ${SSH_TEXT_MAX_BYTES / 1024 / 1024} MiB limit`,
			);
		}
		if (remoteSize === 0) {
			return {
				url: url.href,
				content: "",
				contentType: contentTypeFor(remotePath),
				size: 0,
			};
		}

		// File read via scp
		const { stdout, stderr, code } = await execSsh(
			"scp",
			buildScpArgs(target, remotePath, ["-q", "-C"]),
		);
		if (code !== 0) {
			throw new Error(`ssh://: ${stderr || `scp failed (exit ${code})`}`);
		}

		// Check for binary content
		const buffer = Buffer.from(stdout, "binary");
		if (isLikelyBinary(buffer)) {
			throw new Error(
				`ssh://: ${remotePath} appears to be binary; ssh:// supports UTF-8 text only`,
			);
		}
		if (buffer.length > SSH_TEXT_MAX_BYTES) {
			throw new Error(
				`ssh://: ${remotePath} exceeds ${SSH_TEXT_MAX_BYTES / 1024 / 1024} MiB limit`,
			);
		}

		const content = buffer.toString("utf-8");
		return {
			url: url.href,
			content,
			contentType: contentTypeFor(remotePath),
			size: buffer.length,
		};
	}

	async complete(
		query?: string,
		_context?: ResolveContext,
	): Promise<UrlCompletion[]> {
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

	/**
	 * Byte-exact remote write. Stages stdin into a temp in the destination
	 * directory, then commits in place (existing regular file — preserves
	 * inode and permission bits) or by atomic rename (new path / symlink).
	 * Refuses directories and special files; see writeRemoteFile.
	 */
	async write(
		url: InternalUrl,
		content: string,
		context?: WriteContext,
	): Promise<void> {
		context?.signal?.throwIfAborted();
		const remotePath = remotePathFromUrl(url);
		if (remotePath.endsWith("/")) {
			throw new Error(
				"ssh:// write requires a file path, not a directory (remove the trailing '/')",
			);
		}
		const target = await resolveTarget(url);
		await writeRemoteFile(
			target,
			remotePath,
			Buffer.from(content, "utf-8"),
			context?.signal,
		);
		context?.signal?.throwIfAborted();
	}
}
