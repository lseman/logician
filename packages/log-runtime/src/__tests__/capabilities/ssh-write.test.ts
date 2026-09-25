// ssh:// write path: staged remote write through a fake `ssh` on PATH.
// The fake runs the last argument (the remote command) locally with stdin
// intact, so the real staging/commit shell logic is exercised end to end.

import { afterEach, beforeEach, expect, test } from "bun:test";
import { execSync } from "node:child_process";
import {
	chmodSync,
	existsSync,
	lstatSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	rmSync,
	statSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { parseInternalUrl } from "../../runtime/bridge/support/internal-urls/parse.ts";
import { SshProtocolHandler } from "../../runtime/bridge/support/internal-urls/ssh-protocol.ts";

const FAKE_HOST = "logician-test-host";

let workDir: string;
let fakeBin: string;
let originalPath: string;
let originalCwd: string;

beforeEach(() => {
	workDir = mkdtempSync(path.join(tmpdir(), "ssh-write-"));
	fakeBin = mkdtempSync(path.join(tmpdir(), "ssh-fake-bin-"));
	const script = [
		"#!/bin/sh",
		"# Fake ssh: run the last argument (the remote command) locally; stdin passes through.",
		'last=""',
		'for a in "$@"; do last="$a"; done',
		'exec sh -c "$last"',
		"",
	].join("\n");
	writeFileSync(path.join(fakeBin, "ssh"), script, { mode: 0o755 });
	originalPath = process.env.PATH ?? "";
	process.env.PATH = `${fakeBin}:${originalPath}`;
	originalCwd = process.cwd();
	// Remote paths from ssh:// URLs are relative to the remote home, so the
	// fake "remote" (this process) must live in the scratch directory.
	process.chdir(workDir);
});

afterEach(() => {
	process.chdir(originalCwd);
	process.env.PATH = originalPath;
	rmSync(workDir, { recursive: true, force: true });
	rmSync(fakeBin, { recursive: true, force: true });
});

function writeUrl(relPath: string): string {
	return `ssh://${FAKE_HOST}/${relPath}`;
}

async function write(relPath: string, content: string): Promise<void> {
	await new SshProtocolHandler().write(
		parseInternalUrl(writeUrl(relPath)),
		content,
		{},
	);
}

function stagedTempResidue(): string[] {
	return readdirSync(workDir, { recursive: true })
		.map(String)
		.filter(entry => entry.includes(".logician-tmp."));
}

test("ssh:// write creates a new file byte-exact (non-ASCII included) and leaves no staged temp", async () => {
	const content = "héllo wörld — 日本語\nsecond line\n";
	await write("new.txt", content);
	expect(readFileSync(path.join(workDir, "new.txt"), "utf-8")).toBe(content);
	expect(stagedTempResidue()).toEqual([]);
});

test("ssh:// write with empty content creates an empty file", async () => {
	await write("empty.txt", "");
	expect(existsSync(path.join(workDir, "empty.txt"))).toBe(true);
	expect(statSync(path.join(workDir, "empty.txt")).size).toBe(0);
});

test("ssh:// write overwrites an existing regular file in place (inode and permission bits preserved)", async () => {
	const file = path.join(workDir, "secret.txt");
	writeFileSync(file, "old");
	chmodSync(file, 0o600);
	const before = statSync(file).ino;

	await write("secret.txt", "new");

	expect(readFileSync(file, "utf-8")).toBe("new");
	expect(statSync(file).ino).toBe(before);
	expect(statSync(file).mode & 0o777).toBe(0o600);
	expect(stagedTempResidue()).toEqual([]);
});

test("ssh:// write to a directory is refused, with or without the trailing slash", async () => {
	mkdirSync(path.join(workDir, "subdir"));

	// Trailing slash is caught client-side before any ssh is spawned.
	await expect(write("subdir/", "x")).rejects.toThrow(
		/file path, not a directory/,
	);
	// No trailing slash: the remote commit-by-kind refuses the directory.
	await expect(write("subdir", "x")).rejects.toThrow(/directory/);

	expect(readdirSync(path.join(workDir, "subdir"))).toEqual([]);
	expect(stagedTempResidue()).toEqual([]);
});

test("ssh:// write to a FIFO is refused, not replaced", async () => {
	const fifo = path.join(workDir, "pipe");
	execSync(`mkfifo ${JSON.stringify(fifo)}`);

	await expect(write("pipe", "x")).rejects.toThrow(/special file/);

	expect(lstatSync(fifo).isFIFO()).toBe(true);
	expect(stagedTempResidue()).toEqual([]);
});

test("ssh:// write to a symlink replaces the link with a regular file instead of writing through it", async () => {
	const target = path.join(workDir, "target.txt");
	const link = path.join(workDir, "link.txt");
	writeFileSync(target, "original");
	symlinkSync("target.txt", link);

	await write("link.txt", "via-link");

	expect(lstatSync(link).isSymbolicLink()).toBe(false);
	expect(readFileSync(link, "utf-8")).toBe("via-link");
	expect(readFileSync(target, "utf-8")).toBe("original");
});

test("ssh:// write handles paths with spaces, quotes, and a nested (auto-created) directory", async () => {
	const name = `it's "quoted".txt`;
	await write(`nested/dir/${name}`, "quoted ok");
	expect(readFileSync(path.join(workDir, "nested", "dir", name), "utf-8")).toBe(
		"quoted ok",
	);
	expect(stagedTempResidue()).toEqual([]);
});
