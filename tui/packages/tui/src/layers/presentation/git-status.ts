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
	const branch = getGitBranch();
	let modified = 0;
	let staged = 0;
	let untracked = 0;
	try {
		modified =
			parseInt(
				execSync("git diff --quiet || git diff --name-only | wc -l", {
					cwd: process.cwd(),
					encoding: "utf8",
					stdio: ["ignore", "pipe", "ignore"],
				}).trim(),
			) || 0;
		staged =
			parseInt(
				execSync(
					"git diff --cached --quiet || git diff --cached --name-only | wc -l",
					{
						cwd: process.cwd(),
						encoding: "utf8",
						stdio: ["ignore", "pipe", "ignore"],
					},
				).trim(),
			) || 0;
		untracked =
			parseInt(
				execSync("git ls-files --others --exclude-standard | wc -l", {
					cwd: process.cwd(),
					encoding: "utf8",
					stdio: ["ignore", "pipe", "ignore"],
				}).trim(),
			) || 0;
	} catch {
		// ignore
	}
	return { branch, modified, staged, untracked };
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
