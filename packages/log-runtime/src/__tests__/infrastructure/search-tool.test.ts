import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ensureTool } from "../../capabilities/tools/external-tools.ts";
import { createGrepTool, grep } from "../../capabilities/tools/search.ts";
import { InternalUrlRouter } from "../../runtime/bridge/support/internal-urls/router.ts";
import { SkillProtocolHandler } from "../../runtime/bridge/support/internal-urls/skill-protocol.ts";
import { MemoryProtocolHandler } from "../../runtime/bridge/support/internal-urls/memory-protocol.ts";

const rgTest = (await ensureTool("rg")) ? test : test.skip;

void test("grep prepareArguments does not turn the search pattern into a glob", () => {
	const args = grep.prepareArguments?.({ pattern: "needle" }) ?? {};
	assert.equal(args.pattern, "needle");
	assert.equal(args.glob, undefined);
});

void rgTest(
	"grep finds plain prepared searches across normal filenames",
	async () => {
		const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
		writeFileSync(join(cwd, "notes.txt"), "alpha\nneedle\nomega\n", "utf8");

		const args = grep.prepareArguments?.({ pattern: "needle" }) ?? {
			pattern: "needle",
		};
		const result = await grep.execute(args, { cwd });
		const content = typeof result === "string" ? result : result.content;
		assert.match(content, /notes\.txt:2: needle/);
	},
);

void rgTest(
	"grep reports ripgrep pattern errors instead of no matches",
	async () => {
		const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
		writeFileSync(join(cwd, "notes.txt"), "alpha\n", "utf8");

		const result = await grep.execute({ pattern: "[" }, { cwd });
		const content = typeof result === "string" ? result : result.content;
		assert.match(content, /^Error: /);
		assert.notEqual(content, "No matches found.");
	},
);

void rgTest(
	"grep rebases an internal URL with a sourcePath onto the real file and greps it",
	async () => {
		const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
		const skillDir = join(cwd, "elsewhere");
		mkdirSync(skillDir, { recursive: true });
		const skillFile = join(skillDir, "SKILL.md");
		writeFileSync(skillFile, "heading\nneedle here\nbody", "utf8");

		const router = new InternalUrlRouter();
		router.register(new SkillProtocolHandler());
		const grepTool = createGrepTool(router);

		const result = await grepTool.execute(
			{ pattern: "needle", path: "skill://demo" },
			{
				cwd,
				skills: [{ name: "demo", path: skillFile, content: "heading\nneedle here\nbody" }],
			},
		);
		const content = typeof result === "string" ? result : result.content;
		assert.match(content, /^skill:\/\/demo:2: needle here/);
	},
);

void test("grep searches an internal resource with no backing file in-process", async () => {
	const router = new InternalUrlRouter();
	router.register(new MemoryProtocolHandler());
	const grepTool = createGrepTool(router);

	const memory = {
		listObservations: async () => [{ id: "obs-1", content: "needle in memory" }],
		listMemories: async () => [],
	};
	const result = await grepTool.execute(
		{ pattern: "needle" as const, path: "memory://list" },
		{ memory },
	);
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /memory:\/\/list:\d+: .*needle in memory/);
});

void test("grep surfaces a clear error for a malformed internal URL instead of crashing", async () => {
	const router = new InternalUrlRouter();
	router.register(new SkillProtocolHandler());
	const grepTool = createGrepTool(router);

	const result = await grepTool.execute(
		{ pattern: "x", path: "skill://" },
		{ skills: [] },
	);
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /^Error: /);
});

void test("grep rejects a directory-shaped resource that has no backing sourcePath", async () => {
	const router = new InternalUrlRouter();
	router.register({
		scheme: "listing",
		immutable: true,
		resolve: async url => ({
			url: url.href,
			content: "a\nb\nc",
			isDirectory: true,
		}),
	});
	const grepTool = createGrepTool(router);

	const result = await grepTool.execute({ pattern: "x", path: "listing://" }, {});
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /^Error: grep cannot recurse/);
});
