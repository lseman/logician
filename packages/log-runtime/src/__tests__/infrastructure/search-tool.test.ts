import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ensureTool } from "../../capabilities/tools/external-tools.ts";
import { createGrepTool, grep } from "../../capabilities/tools/search.ts";
import { MemoryProtocolHandler } from "../../runtime/bridge/support/internal-urls/memory-protocol.ts";
import { InternalUrlRouter } from "../../runtime/bridge/support/internal-urls/router.ts";
import { SkillProtocolHandler } from "../../runtime/bridge/support/internal-urls/skill-protocol.ts";

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
	"grep falls back to a literal match for a malformed regex instead of erroring",
	async () => {
		// Native grep's build_matcher() treats an unparseable pattern as a
		// literal string rather than failing the whole search (grep.rs's
		// "final fallback"). "[" isn't in the file, so this is a no-match,
		// not an error — an intentional upstream reliability tradeoff.
		const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
		writeFileSync(join(cwd, "notes.txt"), "alpha\n", "utf8");

		const result = await grep.execute({ pattern: "[" }, { cwd });
		const content = typeof result === "string" ? result : result.content;
		assert.equal(content, "No matches found.");
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
				skills: [
					{
						name: "demo",
						path: skillFile,
						content: "heading\nneedle here\nbody",
					},
				],
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
		listObservations: async () => [
			{ id: "obs-1", content: "needle in memory" },
		],
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

	const result = await grepTool.execute(
		{ pattern: "x", path: "listing://" },
		{},
	);
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /^Error: grep cannot recurse/);
});

// ── skip parameter ──────────────────────────────────────────────────────────
void rgTest("grep skip skips the first N matches", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(
		join(cwd, "notes.txt"),
		"alpha\nfirst needle\nbeta\nsecond needle\ngamma\n",
		"utf8",
	);
	const result = await grep.execute(
		{ pattern: "needle", path: join(cwd, "notes.txt"), limit: 1, skip: 1 },
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /second needle/);
	assert.doesNotMatch(content, /first needle/);
});

void rgTest("grep skip with zero skip returns all matches", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(
		join(cwd, "notes.txt"),
		"alpha\nneedle\nbeta\nneedle\ngamma\n",
		"utf8",
	);
	const result = await grep.execute(
		{ pattern: "needle", path: join(cwd, "notes.txt"), skip: 0 },
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	const lines = content.split("\n").filter(l => l.includes("needle"));
	assert.equal(lines.length, 2);
});

// ── case sensitivity ────────────────────────────────────────────────────────
void rgTest("grep case:true is case-sensitive", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(join(cwd, "notes.txt"), "Alpha\nALPHA\nalpha\n", "utf8");
	const result = await grep.execute(
		{ pattern: "Alpha", path: join(cwd, "notes.txt"), case: true },
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /1: Alpha/);
	assert.doesNotMatch(content, /ALPHA/);
	assert.doesNotMatch(content, /3: alpha/);
});

void rgTest("grep case:false is case-insensitive", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(join(cwd, "notes.txt"), "Alpha\nALPHA\nalpha\n", "utf8");
	const result = await grep.execute(
		{ pattern: "alpha", path: join(cwd, "notes.txt"), case: false },
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	const lines = content
		.split("\n")
		.filter(l => /: (Alpha|ALPHA|alpha)/.test(l));
	assert.equal(lines.length, 3);
});

void rgTest("grep case takes priority over ignoreCase", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(join(cwd, "notes.txt"), "Alpha\nALPHA\nalpha\n", "utf8");
	// case:true overrides ignoreCase:true
	const result = await grep.execute(
		{
			pattern: "alpha",
			path: join(cwd, "notes.txt"),
			case: true,
			ignoreCase: true,
		},
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	const lines = content.split("\n").filter(l => /: alpha/.test(l));
	assert.equal(lines.length, 1);
	assert.match(content, /3: alpha/);
});

// ── cross-line patterns ────────────────────────────────────────────────────
void rgTest("grep detects cross-line patterns via literal \\n", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(join(cwd, "notes.txt"), "hello\nworld\nfoo\n", "utf8");
	const result = await grep.execute(
		{ pattern: "hello\\nworld", path: join(cwd, "notes.txt") },
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	assert.doesNotMatch(content, /No matches/);
});

// ── semicolon-delimited paths ──────────────────────────────────────────────
void rgTest("grep supports semicolon-delimited multiple paths", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(join(cwd, "a.txt"), "from A\n", "utf8");
	writeFileSync(join(cwd, "b.txt"), "from B\n", "utf8");
	const result = await grep.execute(
		{ pattern: "from", path: join(cwd, "a.txt") + ";" + join(cwd, "b.txt") },
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /a\.txt/);
	assert.match(content, /b\.txt/);
});

// ── line-range selector ────────────────────────────────────────────────────
void rgTest("grep filters by file:LINE1-LINE2 selector", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(
		join(cwd, "notes.txt"),
		"line one\nline two\nneedle here\nline four\nline five\n",
		"utf8",
	);
	const result = await grep.execute(
		{
			pattern: "needle",
			path: join(cwd, "notes.txt") + ":1-2",
		},
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /No matches/);
});

void rgTest(
	"grep line-range selector returns match when in range",
	async () => {
		const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
		writeFileSync(
			join(cwd, "notes.txt"),
			"line one\nneedle here\nline three\n",
			"utf8",
		);
		const result = await grep.execute(
			{
				pattern: "needle",
				path: join(cwd, "notes.txt") + ":2-2",
			},
			{ cwd },
		);
		const content = typeof result === "string" ? result : result.content;
		assert.match(content, /needle/);
	},
);

// ── prepareArguments ───────────────────────────────────────────────────────
void test("grep prepareArguments passes through skip and case", () => {
	const args =
		grep.prepareArguments?.({
			pattern: "test",
			skip: 5,
			case: false,
		}) ?? {};
	assert.equal(args.pattern, "test");
	assert.equal(args.skip, 5);
	assert.equal(args.case, false);
});

void test("grep prepareArguments defaults skip to 0", () => {
	const args = grep.prepareArguments?.({ pattern: "test" }) ?? {};
	assert.equal(args.skip, 0);
});
