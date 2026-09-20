import { afterEach, expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { bash } from "../../capabilities/tools/bash.ts";
import { ArtifactRegistry } from "../../runtime/bridge/support/internal-urls/artifact-manager.ts";

// The bash output minimizer (pi-minimize crate, vendored from oh-my-pi)
// rewrites chatty command output before it reaches the model. These tests
// exercise the wiring end-to-end through the real bash tool: minimization
// fires, the full original is persisted as an artifact referenced by the
// tool result, and the kill switch disables the feature. They require the
// native addon to be built (as in CI: `bun run --filter @logician/log-natives build`).

let dir: string | undefined;

afterEach(() => {
	if (dir) {
		rmSync(dir, { recursive: true, force: true });
		dir = undefined;
	}
	delete process.env.LOGICIAN_MINIMIZER;
});

function tempRepo(): string {
	const d = mkdtempSync(path.join(tmpdir(), "logician-bash-min-"));
	dir = d;
	execFileSync("git", ["init", "-q"], { cwd: d });
	// Ten commits: the pi-minimize engine skips output below
	// MIN_MINIMIZE_CHARS (1,000 chars), and a single-commit log is
	// passed through by the git log filter.
	for (let i = 1; i <= 10; i += 1) {
		execFileSync(
			"git",
			[
				"-c",
				"user.email=t@example.com",
				"-c",
				"user.name=t",
				"commit",
				"-q",
				"--allow-empty",
				"-m",
				`commit ${i}`,
			],
			{ cwd: d },
		);
	}
	ArtifactRegistry.resetForTests();
	ArtifactRegistry.instance().init({ cwd: d, sessionId: "min-test" });
	return d;
}

async function run(command: string, cwd: string): Promise<string> {
	const result = await bash.execute({ command }, { cwd });
	return typeof result === "string" ? result : result.content;
}

test("chatty git log is minimized and the original is retrievable via the artifact ref", async () => {
	const cwd = tempRepo();
	const content = await run("git log --no-patch -10", cwd);
	const match = content.match(
		/\[minimized by git filter: \S+ → \S+\. Full output: local:\/\/(\S+?)\.\]/,
	);
	if (!match) throw new Error(`expected a minimization note, got: ${content}`);
	const artifact = await ArtifactRegistry.instance().read(match[1]);
	expect(artifact).toContain("commit 1");
	expect(artifact).toContain("Author:");
});

test("minimized content is the condensed form, not the raw log", async () => {
	const cwd = tempRepo();
	const content = await run("git log --no-patch -10", cwd);
	expect(content).toContain("commit 1");
	expect(content).not.toContain("Author:");
});

test("commands without a matching filter pass through unchanged", async () => {
	const cwd = tempRepo();
	const content = await run("echo hello-min-test", cwd);
	expect(content).toBe("hello-min-test\n");
});

test("LOGICIAN_MINIMIZER=0 disables minimization", async () => {
	const cwd = tempRepo();
	process.env.LOGICIAN_MINIMIZER = "0";
	const content = await run("git log --no-patch -10", cwd);
	expect(content).toContain("Author:");
	expect(content).not.toContain("[minimized by");
});
