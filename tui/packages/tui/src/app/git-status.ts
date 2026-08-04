// ── Git status helpers ────────────────────────────────────────────────────
// Read-only git plumbing for the status bar / startup banner. Pure functions
// over the current working directory — no UI state. Extracted from tui.ts.

import { execSync } from "node:child_process";

export function getGitBranch(): string {
	try {
		return execSync("git branch --show-current", {
			cwd: process.cwd(),
			encoding: "utf8",
			stdio: ["ignore", "pipe", "ignore"],
		}).trim();
	} catch {
		return "";
	}
}

export function getGitStatus(): {
	branch: string;
	modified: number;
	staged: number;
	untracked: number;
} {
	// A single `status --porcelain=v2 --branch` call replaces what used to be
	// 4 sequential execSync shell spawns (branch + 3 more piped through `wc
	// -l`) — each spawn is a blocking subprocess on the startup path, before
	// the first frame paints.
	try {
		const output = execSync("git status --porcelain=v2 --branch", {
			cwd: process.cwd(),
			encoding: "utf8",
			stdio: ["ignore", "pipe", "ignore"],
		});
		let branch = "";
		let modified = 0;
		let staged = 0;
		let untracked = 0;
		for (const line of output.split("\n")) {
			if (line.startsWith("# branch.head ")) {
				const head = line.slice("# branch.head ".length).trim();
				branch = head === "(detached)" ? "" : head;
			} else if (line.startsWith("1 ") || line.startsWith("2 ")) {
				// Ordinary/renamed entry: "1 XY ...path" — X is the index (staged)
				// status, Y is the worktree (unstaged) status; "." means no change.
				const xy = line.slice(2, 4);
				if (xy[0] !== ".") staged++;
				if (xy[1] !== ".") modified++;
			} else if (line.startsWith("? ")) {
				untracked++;
			}
		}
		return { branch, modified, staged, untracked };
	} catch {
		return { branch: "", modified: 0, staged: 0, untracked: 0 };
	}
}

export function getGitVersion(): string {
	try {
		const branch =
			getGitBranch() ||
			execSync("git rev-parse --short HEAD", {
				cwd: process.cwd(),
				encoding: "utf8",
				stdio: ["ignore", "pipe", "ignore"],
			}).trim();
		const sha = execSync("git rev-parse --short HEAD", {
			cwd: process.cwd(),
			encoding: "utf8",
			stdio: ["ignore", "pipe", "ignore"],
		}).trim();
		let dirty = "";
		try {
			execSync("git diff --quiet && git diff --cached --quiet", {
				cwd: process.cwd(),
				stdio: "ignore",
			});
		} catch {
			dirty = " dirty";
		}
		return `${branch}@${sha}${dirty}`;
	} catch {
		return "";
	}
}
